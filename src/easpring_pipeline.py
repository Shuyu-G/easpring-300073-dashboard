from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import numpy as np
import re
import time as time_module
import urllib.parse
from urllib.parse import parse_qs, urlparse
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from html import unescape
from pathlib import Path
from typing import Any

import akshare as ak
import feedparser
from googlenewsdecoder import gnewsdecoder
import joblib
import pandas as pd
import requests
from snownlp import SnowNLP
import trafilatura
import pdfplumber
from pypdf import PdfReader
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COMPANY_NAME = "北京当升材料科技股份有限公司"
DEFAULT_SYMBOL = "300073"
DEFAULT_BENCHMARK = "sz399006"
GOOGLE_NEWS_QUERIES = [
    '"当升科技" OR "北京当升材料科技股份有限公司" OR "300073"',
    '"北京当升材料科技股份有限公司"',
]
GOOGLE_NEWS_ENDPOINT = "https://news.google.com/rss/search"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
COMPANY_KEYWORDS = ["当升科技", "北京当升材料科技股份有限公司"]
GOLD_STYLE_RF_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "daily_return",
    "news_sentiment_mean",
    "news_count",
]
DISCLOSURE_INCLUDE_KEYWORDS = [
    "年报",
    "年度报告",
    "半年度报告",
    "半年报",
    "季报",
    "季度报告",
    "业绩预告",
    "业绩快报",
    "合作",
    "协议",
    "增持",
    "减持",
    "回购",
    "发行",
    "定增",
    "募集资金",
    "募投",
    "融资",
    "担保",
    "补助",
    "投资",
    "减值",
    "关联交易",
    "子公司",
    "股权",
    "股份",
    "诉讼",
    "仲裁",
    "项目",
    "扩产",
    "产能",
    "签署",
    "供货协议",
]
DISCLOSURE_EXCLUDE_KEYWORDS = [
    "制度",
    "工作细则",
    "法律意见书",
    "决议公告",
    "提示性公告",
    "通知公告",
    "述职报告",
    "核查意见",
    "可持续发展报告",
    "审计报告",
    "鉴证报告",
    "专项审计说明",
    "汇总表",
    "培训制度",
    "管理办法",
    "募集说明书",
    "上市保荐书",
    "发行保荐书",
    "回复报告",
    "发行情况报告书",
    "上市公告书",
    "验资报告",
    "专项报告",
]
FINANCIAL_INDICATOR_MAP = {
    "营业总收入": "fin_revenue",
    "归母净利润": "fin_net_profit",
    "经营现金流量净额": "fin_operating_cashflow",
    "基本每股收益": "fin_eps",
    "每股经营现金流": "fin_operating_cashflow_per_share",
    "净资产收益率(ROE)": "fin_roe",
    "毛利率": "fin_gross_margin",
    "销售净利率": "fin_net_margin",
    "资产负债率": "fin_debt_ratio",
    "营业总收入增长率": "fin_revenue_yoy",
    "归属母公司净利润增长率": "fin_profit_yoy",
}

POSITIVE_KEYWORDS = [
    "增长",
    "提升",
    "增持",
    "回购",
    "合作",
    "签署",
    "战略",
    "中标",
    "突破",
    "投产",
    "获批",
    "通过",
    "盈利",
    "预增",
    "补助",
    "完成",
    "落地",
    "买入",
    "推荐",
    "出海",
    "扩产",
    "新高",
    "修复",
]
NEGATIVE_KEYWORDS = [
    "下滑",
    "下降",
    "亏损",
    "预亏",
    "减持",
    "终止",
    "延期",
    "处罚",
    "问询",
    "诉讼",
    "仲裁",
    "风险",
    "减值",
    "失败",
    "取消",
    "波动",
    "承压",
    "下修",
    "冻结",
    "质押",
    "担保",
    "违约",
]
RATING_MAP = {"买入": 1.0, "增持": 0.5, "中性": 0.0, "减持": -0.5, "卖出": -1.0}


@dataclass
class PipelineConfig:
    project_root: Path
    symbol: str = DEFAULT_SYMBOL
    benchmark_symbol: str = DEFAULT_BENCHMARK
    start_date: str = "20240311"
    end_date: str = field(default_factory=lambda: date.today().strftime("%Y%m%d"))
    company_name: str = COMPANY_NAME


@dataclass
class PipelinePaths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    model_dir: Path
    report_dir: Path
    stock_prices_csv: Path
    benchmark_prices_csv: Path
    disclosures_csv: Path
    research_reports_csv: Path
    stock_news_csv: Path
    google_news_csv: Path
    financial_abstract_csv: Path
    financial_report_csv: Path
    pdf_text_cache_csv: Path
    company_news_csv: Path
    daily_news_features_csv: Path
    quarterly_financials_csv: Path
    training_data_csv: Path
    latest_features_csv: Path
    direction_model_path: Path
    return_model_path: Path
    metrics_json_path: Path
    report_md_path: Path


def make_paths(project_root: Path) -> PipelinePaths:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    model_dir = project_root / "models"
    report_dir = project_root / "reports"

    for path in [raw_dir, processed_dir, model_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        project_root=project_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        stock_prices_csv=raw_dir / "stock_prices.csv",
        benchmark_prices_csv=raw_dir / "benchmark_prices.csv",
        disclosures_csv=raw_dir / "disclosures.csv",
        research_reports_csv=raw_dir / "research_reports.csv",
        stock_news_csv=raw_dir / "recent_stock_news.csv",
        google_news_csv=raw_dir / "google_news_archive.csv",
        financial_abstract_csv=raw_dir / "financial_abstract.csv",
        financial_report_csv=raw_dir / "financial_report.csv",
        pdf_text_cache_csv=raw_dir / "pdf_text_cache.csv",
        company_news_csv=processed_dir / "company_news.csv",
        daily_news_features_csv=processed_dir / "daily_news_features.csv",
        quarterly_financials_csv=processed_dir / "quarterly_financials.csv",
        training_data_csv=processed_dir / "training_data.csv",
        latest_features_csv=processed_dir / "latest_features.csv",
        direction_model_path=model_dir / "direction_model.joblib",
        return_model_path=model_dir / "return_model.joblib",
        metrics_json_path=report_dir / "metrics.json",
        report_md_path=report_dir / "analysis.md",
    )


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_cached_frame(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in parse_dates or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def call_with_retries(fetcher: Any, label: str, attempts: int = 3, base_delay_seconds: float = 1.5) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fetcher()
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time_module.sleep(base_delay_seconds * attempt)
    if last_error is not None:
        raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error
    raise RuntimeError(f"{label} failed without raising an exception")


def fetch_frame_with_cache(
    fetcher: Any,
    cache_path: Path,
    label: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    try:
        frame = call_with_retries(fetcher, label=label)
        write_frame(frame, cache_path)
        return frame
    except Exception:
        if cache_path.exists():
            return read_cached_frame(cache_path, parse_dates=parse_dates)
        raise


def to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def month_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_ts.replace(day=1)
    end_exclusive = end_ts + pd.Timedelta(days=1)

    while current < end_exclusive:
        next_month = (current + pd.offsets.MonthBegin(1)).normalize()
        window_start = max(current, start_ts)
        window_end = min(next_month, end_exclusive)
        windows.append((window_start, window_end))
        current = next_month
    return windows


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_google_title(title: str, source_name: str) -> str:
    title = clean_text(title)
    suffix = f" - {source_name}".strip()
    if source_name and title.endswith(suffix):
        return title[: -len(suffix)].strip()
    return title


def fetch_google_news_archive(config: PipelineConfig) -> pd.DataFrame:
    start_ts = pd.Timestamp(datetime.strptime(config.start_date, "%Y%m%d"))
    end_ts = pd.Timestamp(datetime.strptime(config.end_date, "%Y%m%d"))
    windows = month_windows(start_ts, end_ts)

    rows: list[dict[str, Any]] = []
    for query in GOOGLE_NEWS_QUERIES:
        for window_start, window_end in windows:
            search_query = (
                f"{query} after:{(window_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')} "
                f"before:{window_end.strftime('%Y-%m-%d')}"
            )
            params = {
                "hl": "zh-CN",
                "gl": "CN",
                "ceid": "CN:zh-Hans",
                "q": search_query,
            }
            response = requests.get(
                GOOGLE_NEWS_ENDPOINT,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                published_at = pd.to_datetime(entry.get("published"), utc=True, errors="coerce")
                if pd.isna(published_at):
                    continue
                published_at = published_at.tz_convert("Asia/Shanghai").tz_localize(None)
                if not (start_ts <= published_at.normalize() <= end_ts):
                    continue

                source = entry.get("source", {})
                source_name = clean_text(source.get("title", ""))
                title = clean_google_title(entry.get("title", ""), source_name)
                body = clean_text(entry.get("summary", ""))

                rows.append(
                    {
                        "published_at": published_at,
                        "title": title,
                        "summary": body,
                        "source_name": source_name,
                        "source_site": source.get("href", ""),
                        "news_link": entry.get("link", ""),
                        "query": query,
                        "window_start": window_start,
                        "window_end": window_end - pd.Timedelta(days=1),
                    }
                )

    columns = [
        "published_at",
        "title",
        "summary",
        "source_name",
        "source_site",
        "news_link",
        "query",
        "window_start",
        "window_end",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame

    frame["dedupe_date"] = frame["published_at"].dt.normalize()
    frame["dedupe_title"] = frame["title"].str.replace(r"\s+", "", regex=True)
    frame = frame.drop_duplicates(subset=["dedupe_date", "dedupe_title", "source_name"]).copy()
    frame = frame.drop(columns=["dedupe_date", "dedupe_title"])
    frame = frame.sort_values("published_at", ascending=False).reset_index(drop=True)
    return frame


def extract_article_text(decoded_url: str) -> tuple[str, str, int]:
    response = requests.get(decoded_url, headers=REQUEST_HEADERS, timeout=(5, 10))
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    extracted = trafilatura.extract(
        response.text,
        url=response.url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        deduplicate=True,
    )
    cleaned = clean_text(extracted or "")
    return response.url, cleaned, len(cleaned)


def company_relevance_score(text: str) -> int:
    text = clean_text(text)
    strong_hits = sum(text.count(keyword) for keyword in COMPANY_KEYWORDS)
    code_hits = text.count(DEFAULT_SYMBOL)
    return strong_hits * 2 + code_hits


def is_material_disclosure(title: str) -> bool:
    title = str(title)
    include_hit = any(keyword in title for keyword in DISCLOSURE_INCLUDE_KEYWORDS)
    exclude_hit = any(keyword in title for keyword in DISCLOSURE_EXCLUDE_KEYWORDS)
    return include_hit and not exclude_hit


def build_cninfo_pdf_url(source_link: str, announcement_time: str | pd.Timestamp) -> str:
    query = parse_qs(urlparse(str(source_link)).query)
    announcement_id = query.get("announcementId", [""])[0]
    if not announcement_id:
        return ""
    announcement_date = pd.to_datetime(announcement_time).strftime("%Y-%m-%d")
    return f"https://static.cninfo.com.cn/finalpage/{announcement_date}/{announcement_id}.PDF"


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 12, max_chars: int = 16000) -> tuple[str, int, int]:
    pages_processed = 0
    page_count = 0
    text = ""

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        parts: list[str] = []
        for page in reader.pages[:max_pages]:
            pages_processed += 1
            page_text = (page.extract_text() or "").strip()
            if page_text:
                parts.append(page_text)
            if sum(len(part) for part in parts) >= max_chars:
                break
        text = "\n".join(parts).strip()
    except Exception:
        text = ""

    if len(text) < 200:
        pages_processed = 0
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)
                parts = []
                for page in pdf.pages[:max_pages]:
                    pages_processed += 1
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        parts.append(page_text)
                    if sum(len(part) for part in parts) >= max_chars:
                        break
                text = "\n".join(parts).strip()
        except Exception:
            text = ""

    text = clean_text(text)[:max_chars]
    return text, page_count, pages_processed


def build_pdf_source_index(paths: PipelinePaths) -> pd.DataFrame:
    disclosures = pd.read_csv(paths.disclosures_csv)
    disclosures = disclosures[disclosures["公告标题"].apply(is_material_disclosure)].copy()
    disclosures["公告时间"] = pd.to_datetime(disclosures["公告时间"])
    disclosures = disclosures.sort_values("公告时间", ascending=False).head(80).copy()
    disclosures["pdf_url"] = disclosures.apply(
        lambda row: build_cninfo_pdf_url(row["公告链接"], row["公告时间"]),
        axis=1,
    )
    disclosures["source_type"] = "disclosure"
    disclosures["source_link"] = disclosures["公告链接"]
    disclosures["title"] = disclosures["公告标题"]
    disclosures["published_at"] = disclosures["公告时间"]

    research = pd.read_csv(paths.research_reports_csv)
    research["pdf_url"] = research["报告PDF链接"]
    research["source_type"] = "research"
    research["source_link"] = research["报告PDF链接"]
    research["title"] = research["报告名称"]
    research["published_at"] = research["日期"]

    columns = ["source_type", "source_link", "pdf_url", "title", "published_at"]
    frame = pd.concat([disclosures[columns], research[columns]], ignore_index=True)
    frame = frame.dropna(subset=["pdf_url"]).drop_duplicates(subset=["source_type", "source_link"]).reset_index(drop=True)
    return frame


def fetch_pdf_text_cache(paths: PipelinePaths) -> dict[str, int]:
    source_index = build_pdf_source_index(paths)

    if paths.pdf_text_cache_csv.exists():
        cache = pd.read_csv(paths.pdf_text_cache_csv)
    else:
        cache = pd.DataFrame(
            columns=[
                "source_type",
                "source_link",
                "pdf_url",
                "title",
                "published_at",
                "extract_status",
                "text_len",
                "page_count",
                "pages_processed",
                "pdf_text",
                "fetched_at",
            ]
        )

    cached_keys = set(zip(cache["source_type"], cache["source_link"])) if not cache.empty else set()
    pending = source_index[
        ~source_index.apply(lambda row: (row["source_type"], row["source_link"]) in cached_keys, axis=1)
    ].copy()

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        extract_status = "failed"
        text_len = 0
        page_count = 0
        pages_processed = 0
        pdf_text = ""

        try:
            response = requests.get(row["pdf_url"], headers=REQUEST_HEADERS, timeout=(5, 20))
            response.raise_for_status()
            pdf_text, page_count, pages_processed = extract_pdf_text(response.content)
            text_len = len(pdf_text)
            extract_status = "ok" if text_len >= 200 else "short"
        except Exception:
            extract_status = "failed"

        return {
            "source_type": row["source_type"],
            "source_link": row["source_link"],
            "pdf_url": row["pdf_url"],
            "title": row["title"],
            "published_at": row["published_at"],
            "extract_status": extract_status,
            "text_len": text_len,
            "page_count": page_count,
            "pages_processed": pages_processed,
            "pdf_text": pdf_text,
            "fetched_at": datetime.now().isoformat(),
        }

    new_rows: list[dict[str, Any]] = []
    records = pending.to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_row, row) for row in records]
        for future in as_completed(futures):
            new_rows.append(future.result())

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates(subset=["source_type", "source_link"], keep="last")

    cache = source_index.merge(
        cache.drop(columns=["pdf_url", "title", "published_at"], errors="ignore"),
        on=["source_type", "source_link"],
        how="left",
    )
    cache = cache.sort_values(["source_type", "published_at"], ascending=[True, False]).reset_index(drop=True)
    write_frame(cache, paths.pdf_text_cache_csv)

    return {
        "pdf_sources_indexed": len(source_index),
        "pdf_cache_rows": len(cache),
        "pdf_extract_ok": int((cache["extract_status"] == "ok").sum()) if not cache.empty else 0,
        "pdf_extract_short": int((cache["extract_status"] == "short").sum()) if not cache.empty else 0,
    }


def enrich_google_news_archive(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        enriched = frame.copy()
        for column in [
            "decoded_url",
            "final_url",
            "article_text",
            "article_text_len",
            "decode_status",
            "extract_status",
            "keyword_score",
        ]:
            enriched[column] = []
        return enriched

    def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
        decoded_url = ""
        final_url = ""
        article_text = ""
        article_text_len = 0
        decode_status = "failed"
        extract_status = "failed"

        try:
            decoded_payload = gnewsdecoder(row["news_link"], interval=0)
            if decoded_payload.get("status") and decoded_payload.get("decoded_url"):
                decoded_url = decoded_payload["decoded_url"]
                decode_status = "ok"
        except Exception:
            decode_status = "failed"

        if decoded_url:
            try:
                final_url, article_text, article_text_len = extract_article_text(decoded_url)
                extract_status = "ok" if article_text_len >= 80 else "short"
            except Exception:
                extract_status = "failed"

        combined_text = f"{row.get('title', '')} {row.get('summary', '')} {article_text}"
        keyword_score = company_relevance_score(combined_text)

        return {
            **row,
            "decoded_url": decoded_url,
            "final_url": final_url,
            "article_text": article_text,
            "article_text_len": article_text_len,
            "decode_status": decode_status,
            "extract_status": extract_status,
            "keyword_score": keyword_score,
        }

    enriched_rows: list[dict[str, Any]] = []
    records = frame.to_dict("records")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(enrich_row, row) for row in records]
        for future in as_completed(futures):
            enriched_rows.append(future.result())

    enriched = pd.DataFrame(enriched_rows)
    enriched = enriched[enriched["keyword_score"] > 0].copy()
    return enriched.sort_values("published_at", ascending=False).reset_index(drop=True)


def to_exchange_prefixed_symbol(symbol: str) -> str:
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    if symbol.startswith("6"):
        return f"sh{symbol}"
    return symbol


def fetch_financial_data(config: PipelineConfig, paths: PipelinePaths) -> dict[str, int]:
    financial_abstract = fetch_frame_with_cache(
        lambda: ak.stock_financial_abstract(symbol=config.symbol).copy(),
        paths.financial_abstract_csv,
        label="financial abstract",
    )

    financial_report = fetch_frame_with_cache(
        lambda: ak.stock_financial_report_sina(
            stock=to_exchange_prefixed_symbol(config.symbol),
            symbol="利润表",
        )
        .copy()
        .assign(
            报告日=lambda frame: pd.to_datetime(frame["报告日"], errors="coerce"),
            公告日期=lambda frame: pd.to_datetime(frame["公告日期"], format="%Y%m%d", errors="coerce"),
        )
        .sort_values(["报告日", "公告日期"])
        .reset_index(drop=True),
        paths.financial_report_csv,
        label="financial report",
        parse_dates=["报告日", "公告日期"],
    )

    return {
        "financial_abstract_rows": len(financial_abstract),
        "financial_report_rows": len(financial_report),
    }


def build_financial_features(paths: PipelinePaths, prices: pd.DataFrame) -> pd.DataFrame:
    trade_dates = prices["date"].tolist()
    trade_set = set(trade_dates)

    abstract = pd.read_csv(paths.financial_abstract_csv)
    report_frame = pd.read_csv(paths.financial_report_csv, parse_dates=["报告日", "公告日期"])

    selected_indicators = list(FINANCIAL_INDICATOR_MAP.keys())
    abstract = abstract[abstract["指标"].isin(selected_indicators)].copy()
    abstract = abstract.drop_duplicates(subset=["指标"], keep="first")
    date_columns = [column for column in abstract.columns if re.fullmatch(r"\d{8}", str(column))]

    long_frame = abstract.melt(
        id_vars=["指标"],
        value_vars=date_columns,
        var_name="report_date",
        value_name="value",
    )
    long_frame["report_date"] = pd.to_datetime(long_frame["report_date"], format="%Y%m%d", errors="coerce")
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
    long_frame = long_frame.dropna(subset=["report_date"])
    long_frame["feature_name"] = long_frame["指标"].map(FINANCIAL_INDICATOR_MAP)
    financial_metrics = (
        long_frame.pivot_table(index="report_date", columns="feature_name", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={"report_date": "报告日"})
    )

    release_dates = (
        report_frame[["报告日", "公告日期"]]
        .dropna()
        .sort_values(["报告日", "公告日期"])
        .drop_duplicates(subset=["报告日"], keep="first")
    )
    quarterly = financial_metrics.merge(release_dates, on="报告日", how="left")
    quarterly["公告日期"] = quarterly["公告日期"].fillna(quarterly["报告日"])
    quarterly["effective_timestamp"] = quarterly["公告日期"] + pd.Timedelta(hours=18)
    quarterly["effective_date"] = quarterly["effective_timestamp"].apply(
        lambda value: map_to_trade_date(value, trade_dates, trade_set)
    )
    quarterly = quarterly.dropna(subset=["effective_date"]).sort_values("effective_date").reset_index(drop=True)

    scale_columns = ["fin_revenue", "fin_net_profit", "fin_operating_cashflow"]
    for column in scale_columns:
        if column in quarterly.columns:
            quarterly[column] = quarterly[column] / 1e9

    write_frame(quarterly, paths.quarterly_financials_csv)
    return quarterly


def fetch_stock_prices(config: PipelineConfig, paths: PipelinePaths) -> pd.DataFrame:
    def get_prices() -> pd.DataFrame:
        frame = ak.stock_zh_a_hist(
            symbol=config.symbol,
            period="daily",
            start_date=config.start_date,
            end_date=config.end_date,
            adjust="qfq",
        ).copy()
        frame = frame.rename(
            columns={
                "日期": "date",
                "股票代码": "symbol",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "chg",
                "换手率": "turnover",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.sort_values("date").reset_index(drop=True)

    def get_benchmark() -> pd.DataFrame:
        frame = ak.stock_zh_index_daily_em(
            symbol=config.benchmark_symbol,
            start_date=config.start_date,
            end_date=config.end_date,
        ).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date").reset_index(drop=True)
        return frame.rename(
            columns={"open": "benchmark_open", "close": "benchmark_close", "volume": "benchmark_volume"}
        )

    try:
        prices = call_with_retries(get_prices, label="stock prices")
        benchmark = call_with_retries(get_benchmark, label="benchmark prices")
        write_frame(prices, paths.stock_prices_csv)
        write_frame(benchmark, paths.benchmark_prices_csv)
        return prices
    except Exception:
        if paths.stock_prices_csv.exists() and paths.benchmark_prices_csv.exists():
            return read_cached_frame(paths.stock_prices_csv, parse_dates=["date"])
        raise


def fetch_company_news(config: PipelineConfig, paths: PipelinePaths) -> dict[str, int]:
    start_ts = pd.Timestamp(datetime.strptime(config.start_date, "%Y%m%d"))
    end_ts = pd.Timestamp(datetime.strptime(config.end_date, "%Y%m%d"))

    disclosures = fetch_frame_with_cache(
        lambda: ak.stock_zh_a_disclosure_report_cninfo(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        .copy()
        .assign(公告时间=lambda frame: pd.to_datetime(frame["公告时间"]))
        .sort_values("公告时间", ascending=False)
        .reset_index(drop=True),
        paths.disclosures_csv,
        label="cninfo disclosures",
        parse_dates=["公告时间"],
    )

    research = fetch_frame_with_cache(
        lambda: ak.stock_research_report_em(symbol=config.symbol)
        .copy()
        .assign(日期=lambda frame: pd.to_datetime(frame["日期"]))
        .loc[lambda frame: (frame["日期"] >= start_ts) & (frame["日期"] <= end_ts)]
        .sort_values("日期", ascending=False)
        .reset_index(drop=True),
        paths.research_reports_csv,
        label="research reports",
        parse_dates=["日期"],
    )

    recent_news = fetch_frame_with_cache(
        lambda: ak.stock_news_em(symbol=config.symbol)
        .copy()
        .assign(发布时间=lambda frame: pd.to_datetime(frame["发布时间"]))
        .sort_values("发布时间", ascending=False)
        .reset_index(drop=True),
        paths.stock_news_csv,
        label="recent stock news",
        parse_dates=["发布时间"],
    )

    google_news = fetch_frame_with_cache(
        lambda: enrich_google_news_archive(fetch_google_news_archive(config)),
        paths.google_news_csv,
        label="google news archive",
        parse_dates=["published_at", "window_start", "window_end"],
    )

    return {
        "disclosures": len(disclosures),
        "research_reports": len(research),
        "recent_news": len(recent_news),
        "google_news_archive": len(google_news),
    }


def map_to_trade_date(timestamp: pd.Timestamp, trade_dates: list[pd.Timestamp], trade_set: set[pd.Timestamp]) -> pd.Timestamp | pd.NaT:
    event_time = pd.Timestamp(timestamp)
    trade_day = event_time.normalize()

    if trade_day in trade_set and event_time.time() <= time(15, 0):
        return trade_day

    if trade_day not in trade_set:
        idx = bisect.bisect_left(trade_dates, trade_day)
    else:
        idx = bisect.bisect_right(trade_dates, trade_day)

    if idx >= len(trade_dates):
        return pd.NaT
    return trade_dates[idx]


def lexicon_score(text: str) -> float:
    text = str(text)
    positive_hits = sum(text.count(keyword) for keyword in POSITIVE_KEYWORDS)
    negative_hits = sum(text.count(keyword) for keyword in NEGATIVE_KEYWORDS)
    if positive_hits + negative_hits == 0:
        return 0.0
    return (positive_hits - negative_hits) / math.sqrt(positive_hits + negative_hits)


def snow_score(text: str) -> float:
    text = str(text).strip()
    if not text:
        return 0.0
    try:
        return SnowNLP(text).sentiments - 0.5
    except Exception:
        return 0.0


def load_and_prepare_news(paths: PipelinePaths, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_dates = prices["date"].tolist()
    trade_set = set(trade_dates)
    pdf_cache = pd.read_csv(paths.pdf_text_cache_csv) if paths.pdf_text_cache_csv.exists() else pd.DataFrame()
    if not pdf_cache.empty:
        pdf_cache = pdf_cache[["source_type", "source_link", "extract_status", "text_len", "pdf_text"]].copy()

    disclosures = pd.read_csv(paths.disclosures_csv, parse_dates=["公告时间"])
    disclosures["timestamp"] = disclosures["公告时间"]
    disclosures["title"] = disclosures["公告标题"]
    disclosures["body"] = ""
    disclosures["source_type"] = "disclosure"
    disclosures["rating_score"] = 0.0
    disclosures["source_link"] = disclosures["公告链接"]
    disclosures["source_name"] = "CNInfo"
    if not pdf_cache.empty:
        disclosure_cache = pdf_cache[pdf_cache["source_type"] == "disclosure"].copy()
        disclosures = disclosures.merge(
            disclosure_cache,
            on=["source_type", "source_link"],
            how="left",
            suffixes=("", "_pdf"),
        )
        disclosures["body"] = disclosures["pdf_text"].fillna(disclosures["body"])

    research = pd.read_csv(paths.research_reports_csv, parse_dates=["日期"])
    research["timestamp"] = research["日期"] + pd.Timedelta(hours=12)
    research["title"] = research["报告名称"]
    research["body"] = research["机构"].astype(str) + " " + research["行业"].astype(str)
    research["source_type"] = "research"
    research["rating_score"] = research["东财评级"].map(RATING_MAP).fillna(0.0)
    research["source_link"] = research["报告PDF链接"]
    research["source_name"] = research["机构"].fillna("研究报告")
    if not pdf_cache.empty:
        research_cache = pdf_cache[pdf_cache["source_type"] == "research"].copy()
        research = research.merge(
            research_cache,
            on=["source_type", "source_link"],
            how="left",
            suffixes=("", "_pdf"),
        )
        research["body"] = research["pdf_text"].fillna(research["body"])

    stock_news = pd.read_csv(paths.stock_news_csv, parse_dates=["发布时间"])
    stock_news["timestamp"] = stock_news["发布时间"]
    stock_news["title"] = stock_news["新闻标题"]
    stock_news["body"] = stock_news["新闻内容"].fillna("")
    stock_news["source_type"] = "news"
    stock_news["rating_score"] = 0.0
    stock_news["source_link"] = stock_news["新闻链接"]
    stock_news["source_name"] = stock_news["文章来源"].fillna("东方财富")

    google_news = pd.read_csv(
        paths.google_news_csv,
        parse_dates=["published_at", "window_start", "window_end"],
    )
    if google_news.empty:
        google_news = pd.DataFrame(
            columns=[
                "published_at",
                "title",
                "summary",
                "source_name",
                "source_site",
                "news_link",
                "article_text",
                "article_text_len",
                "extract_status",
            ]
        )
    google_news["timestamp"] = google_news["published_at"]
    google_news["body"] = google_news["article_text"].fillna("")
    google_news.loc[google_news["body"].str.len().fillna(0) < 80, "body"] = google_news["summary"].fillna("")
    google_news["source_type"] = "web_news"
    google_news["rating_score"] = 0.0
    google_news["source_link"] = google_news["news_link"]

    company_news = pd.concat(
        [
            disclosures[["timestamp", "title", "body", "source_type", "rating_score", "source_link", "source_name"]],
            research[["timestamp", "title", "body", "source_type", "rating_score", "source_link", "source_name"]],
            stock_news[["timestamp", "title", "body", "source_type", "rating_score", "source_link", "source_name"]],
            google_news[["timestamp", "title", "body", "source_type", "rating_score", "source_link", "source_name"]],
        ],
        ignore_index=True,
    )
    company_news["dedupe_date"] = pd.to_datetime(company_news["timestamp"]).dt.normalize()
    company_news["dedupe_title"] = company_news["title"].fillna("").str.replace(r"\s+", "", regex=True)
    company_news["dedupe_source"] = company_news["source_name"].fillna("").str.replace(r"\s+", "", regex=True)
    company_news = company_news.drop_duplicates(
        subset=["dedupe_date", "dedupe_title", "dedupe_source"]
    ).reset_index(drop=True)
    company_news = company_news.drop(columns=["dedupe_date", "dedupe_title", "dedupe_source"])
    company_news["effective_date"] = company_news["timestamp"].apply(
        lambda value: map_to_trade_date(value, trade_dates, trade_set)
    )
    company_news = company_news.dropna(subset=["effective_date"]).copy()

    text_blob = (company_news["title"].fillna("") + " " + company_news["body"].fillna("")).str.slice(0, 1600)
    company_news["snow_score"] = text_blob.apply(snow_score)
    company_news["lexicon_score"] = text_blob.apply(lexicon_score)
    company_news["sentiment_score"] = (
        0.6 * company_news["snow_score"]
        + 0.4 * company_news["lexicon_score"]
        + 0.3 * company_news["rating_score"]
    )
    company_news["positive_flag"] = (company_news["sentiment_score"] > 0.15).astype(int)
    company_news["negative_flag"] = (company_news["sentiment_score"] < -0.15).astype(int)

    source_dummies = pd.get_dummies(company_news["source_type"], prefix="src")
    company_news = pd.concat([company_news, source_dummies], axis=1)

    aggregations: dict[str, Any] = {
        "sentiment_score": ["mean", "sum", "std"],
        "rating_score": ["mean", "sum"],
        "positive_flag": "sum",
        "negative_flag": "sum",
        "title": "count",
    }
    for column in source_dummies.columns:
        aggregations[column] = "sum"

    daily_news = company_news.groupby("effective_date").agg(aggregations)
    daily_news.columns = [
        "_".join([part for part in column if part]).strip("_")
        for column in daily_news.columns.to_flat_index()
    ]
    daily_news = daily_news.rename(
        columns={
            "sentiment_score_mean": "news_sentiment_mean",
            "sentiment_score_sum": "news_sentiment_sum",
            "sentiment_score_std": "news_sentiment_std",
            "rating_score_mean": "rating_mean",
            "rating_score_sum": "rating_sum",
            "positive_flag_sum": "positive_news_count",
            "negative_flag_sum": "negative_news_count",
            "title_count": "news_count",
            "src_disclosure_sum": "disclosure_count",
            "src_research_sum": "research_count",
            "src_news_sum": "article_count",
            "src_web_news_sum": "web_news_count",
        }
    ).reset_index().rename(columns={"effective_date": "date"})
    return company_news, daily_news


def build_training_dataset(config: PipelineConfig, paths: PipelinePaths) -> dict[str, Any]:
    prices = pd.read_csv(paths.stock_prices_csv, parse_dates=["date"])
    benchmark = pd.read_csv(paths.benchmark_prices_csv, parse_dates=["date"])
    company_news, daily_news = load_and_prepare_news(paths, prices)
    quarterly_financials = build_financial_features(paths, prices)

    benchmark = benchmark[["date", "benchmark_close", "benchmark_volume"]].copy()
    benchmark["benchmark_ret_1d"] = benchmark["benchmark_close"].pct_change()
    benchmark["benchmark_ret_5d"] = benchmark["benchmark_close"].pct_change(5)

    features = prices.merge(
        benchmark[["date", "benchmark_close", "benchmark_ret_1d", "benchmark_ret_5d"]],
        on="date",
        how="left",
    )
    features["ret_1d"] = features["close"].pct_change()
    features["ret_3d"] = features["close"].pct_change(3)
    features["ret_5d"] = features["close"].pct_change(5)
    features["daily_return"] = features["ret_1d"]
    features["ma_5"] = features["close"].rolling(5).mean()
    features["ma_10"] = features["close"].rolling(10).mean()
    features["ma_20"] = features["close"].rolling(20).mean()
    features["ma_60"] = features["close"].rolling(60).mean()
    features["ma_ratio_5"] = features["close"] / features["ma_5"] - 1
    features["ma_ratio_10"] = features["close"] / features["ma_10"] - 1
    features["ma_ratio_20"] = features["close"] / features["ma_20"] - 1
    features["ma_ratio_60"] = features["close"] / features["ma_60"] - 1
    features["volatility_5"] = features["ret_1d"].rolling(5).std()
    features["volatility_20"] = features["ret_1d"].rolling(20).std()
    features["volume_ma_5"] = features["volume"].rolling(5).mean()
    features["volume_ratio_5"] = features["volume"] / features["volume_ma_5"] - 1

    financial_feature_columns = [
        column
        for column in quarterly_financials.columns
        if column not in {"报告日", "公告日期", "effective_timestamp", "effective_date"}
    ]
    financial_merge = quarterly_financials[["effective_date"] + financial_feature_columns].copy()
    financial_merge = financial_merge.sort_values("effective_date")
    features = pd.merge_asof(
        features.sort_values("date"),
        financial_merge.rename(columns={"effective_date": "financial_effective_date"}),
        left_on="date",
        right_on="financial_effective_date",
        direction="backward",
    )
    features["financial_freshness_days"] = (
        features["date"] - features["financial_effective_date"]
    ).dt.days

    features = features.merge(daily_news, on="date", how="left")
    news_columns = [
        "news_sentiment_mean",
        "news_sentiment_sum",
        "news_sentiment_std",
        "rating_mean",
        "rating_sum",
        "positive_news_count",
        "negative_news_count",
        "news_count",
        "disclosure_count",
        "research_count",
        "article_count",
        "web_news_count",
    ]
    for column in news_columns:
        if column not in features.columns:
            features[column] = 0.0
    features[news_columns] = features[news_columns].fillna(0.0)

    features["news_count_5d"] = features["news_count"].rolling(5, min_periods=1).sum()
    features["sentiment_mean_5d"] = features["news_sentiment_mean"].rolling(5, min_periods=1).mean()
    features["disclosure_count_5d"] = features["disclosure_count"].rolling(5, min_periods=1).sum()
    features["research_count_20d"] = features["research_count"].rolling(20, min_periods=1).sum()
    features["web_news_count_5d"] = features["web_news_count"].rolling(5, min_periods=1).sum()
    features["next_return"] = features["close"].shift(-1) / features["close"] - 1
    features["target_up"] = (features["next_return"] > 0).astype(int)

    extended_feature_columns = [
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "turnover",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_20",
        "ma_ratio_60",
        "volatility_5",
        "volatility_20",
        "volume_ratio_5",
        "benchmark_ret_1d",
        "benchmark_ret_5d",
        "fin_revenue",
        "fin_net_profit",
        "fin_operating_cashflow",
        "fin_eps",
        "fin_operating_cashflow_per_share",
        "fin_roe",
        "fin_gross_margin",
        "fin_net_margin",
        "fin_debt_ratio",
        "fin_revenue_yoy",
        "fin_profit_yoy",
        "financial_freshness_days",
        "news_sentiment_mean",
        "news_sentiment_sum",
        "news_sentiment_std",
        "rating_mean",
        "rating_sum",
        "positive_news_count",
        "negative_news_count",
        "news_count",
        "disclosure_count",
        "research_count",
        "article_count",
        "web_news_count",
        "news_count_5d",
        "sentiment_mean_5d",
        "disclosure_count_5d",
        "research_count_20d",
        "web_news_count_5d",
    ]
    feature_columns = list(dict.fromkeys(GOLD_STYLE_RF_FEATURES + extended_feature_columns))

    latest_features = features.iloc[[-1]][["date"] + feature_columns].copy()
    training_data = features.dropna(subset=["next_return"]).copy()
    training_data = training_data[["date"] + feature_columns + ["target_up", "next_return"]]

    write_frame(company_news.sort_values("timestamp"), paths.company_news_csv)
    write_frame(daily_news.sort_values("date"), paths.daily_news_features_csv)
    write_frame(training_data, paths.training_data_csv)
    write_frame(latest_features, paths.latest_features_csv)

    return {
        "feature_columns": feature_columns,
        "extended_feature_columns": extended_feature_columns,
        "simple_rf_feature_columns": GOLD_STYLE_RF_FEATURES,
        "price_rows": len(prices),
        "news_items": len(company_news),
        "news_days": int((daily_news["news_count"] > 0).sum()) if "news_count" in daily_news.columns else 0,
        "financial_reports": len(quarterly_financials),
        "trainable_rows": len(training_data),
    }


def select_best_model(results: list[dict[str, Any]], sort_keys: list[str], descending: bool = True) -> dict[str, Any]:
    sorted_results = sorted(
        results,
        key=lambda item: tuple(item[key] for key in sort_keys),
        reverse=descending,
    )
    return sorted_results[0]


def extract_feature_importance(fitted_model: Pipeline, feature_columns: list[str]) -> list[dict[str, Any]]:
    model = fitted_model.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = model.coef_[0]
    else:
        return []

    importance = pd.DataFrame({"feature": feature_columns, "importance": values})
    importance["abs_importance"] = importance["importance"].abs()
    importance = importance.sort_values("abs_importance", ascending=False)
    return [
        {"feature": row.feature, "importance": float(row.importance)}
        for row in importance.head(12).itertuples()
    ]


def positive_class_probabilities(fitted_model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = fitted_model.predict_proba(frame)
    classes = getattr(fitted_model, "classes_", None)
    if classes is None and hasattr(fitted_model, "named_steps"):
        classes = getattr(fitted_model.named_steps.get("model"), "classes_", None)

    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.shape[1] == 1:
        if classes is not None and len(classes) == 1 and int(classes[0]) == 1:
            return np.ones(len(frame), dtype=float)
        return np.zeros(len(frame), dtype=float)
    if classes is not None and 1 in list(classes):
        return probabilities[:, list(classes).index(1)].astype(float)
    return probabilities[:, -1].astype(float)


def choose_best_threshold(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    best_result: dict[str, float] | None = None
    candidate_thresholds = np.round(np.arange(0.30, 0.701, 0.01), 2)

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        result = {
            "decision_threshold": float(threshold),
            "accuracy": float(accuracy_score(targets, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        }
        rank = (
            result["balanced_accuracy"],
            result["accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best_result is None or rank > (
            best_result["balanced_accuracy"],
            best_result["accuracy"],
            -abs(best_result["decision_threshold"] - 0.5),
        ):
            best_result = result

    return best_result or {"decision_threshold": 0.5, "accuracy": 0.0, "balanced_accuracy": 0.0}


def tune_threshold_with_time_series_cv(
    candidate_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> dict[str, Any]:
    if len(X_train) < 80 or y_train.nunique() < 2:
        return {
            "decision_threshold": 0.5,
            "threshold_source": "default",
            "cv_folds": 0,
            "cv_accuracy": None,
            "cv_balanced_accuracy": None,
            "cv_roc_auc": None,
        }

    max_splits = min(n_splits, len(X_train) - 1)
    if max_splits < 2:
        return {
            "decision_threshold": 0.5,
            "threshold_source": "default",
            "cv_folds": 0,
            "cv_accuracy": None,
            "cv_balanced_accuracy": None,
            "cv_roc_auc": None,
        }

    splitter = TimeSeriesSplit(n_splits=max_splits)
    oof_probabilities: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []
    valid_folds = 0

    for fold_train_idx, fold_valid_idx in splitter.split(X_train):
        fold_X_train = X_train.iloc[fold_train_idx]
        fold_y_train = y_train.iloc[fold_train_idx]
        fold_X_valid = X_train.iloc[fold_valid_idx]
        fold_y_valid = y_train.iloc[fold_valid_idx]
        if fold_X_train.empty or fold_X_valid.empty:
            continue

        fold_model = clone(candidate_pipeline)
        fold_model.fit(fold_X_train, fold_y_train)
        fold_probabilities = positive_class_probabilities(fold_model, fold_X_valid)
        oof_probabilities.append(fold_probabilities)
        oof_targets.append(fold_y_valid.to_numpy())
        valid_folds += 1

    if valid_folds < 2:
        return {
            "decision_threshold": 0.5,
            "threshold_source": "default",
            "cv_folds": valid_folds,
            "cv_accuracy": None,
            "cv_balanced_accuracy": None,
            "cv_roc_auc": None,
        }

    cv_probabilities = np.concatenate(oof_probabilities)
    cv_targets = np.concatenate(oof_targets)
    threshold_result = choose_best_threshold(cv_targets, cv_probabilities)
    cv_roc_auc = (
        float(roc_auc_score(cv_targets, cv_probabilities))
        if len(np.unique(cv_targets)) > 1
        else None
    )

    return {
        "decision_threshold": threshold_result["decision_threshold"],
        "threshold_source": "time_series_cv",
        "cv_folds": valid_folds,
        "cv_accuracy": threshold_result["accuracy"],
        "cv_balanced_accuracy": threshold_result["balanced_accuracy"],
        "cv_roc_auc": cv_roc_auc,
    }


def train_models(config: PipelineConfig, paths: PipelinePaths) -> dict[str, Any]:
    training_data = pd.read_csv(paths.training_data_csv, parse_dates=["date"])
    latest_features = pd.read_csv(paths.latest_features_csv, parse_dates=["date"])
    company_news = pd.read_csv(paths.company_news_csv, parse_dates=["timestamp", "effective_date"])
    quarterly_financials = pd.read_csv(
        paths.quarterly_financials_csv,
        parse_dates=["报告日", "公告日期", "effective_timestamp", "effective_date"],
    )
    google_news = pd.read_csv(paths.google_news_csv)
    pdf_cache = pd.read_csv(paths.pdf_text_cache_csv) if paths.pdf_text_cache_csv.exists() else pd.DataFrame()
    feature_columns = [column for column in training_data.columns if column not in {"date", "target_up", "next_return"}]
    extended_feature_columns = [column for column in feature_columns if column not in {"open", "high", "low", "daily_return"}]
    simple_rf_feature_columns = [column for column in GOLD_STYLE_RF_FEATURES if column in feature_columns]

    train_size = int(len(training_data) * 0.8)
    train_df = training_data.iloc[:train_size].copy()
    test_df = training_data.iloc[train_size:].copy()
    y_train = train_df["target_up"]
    y_test = test_df["target_up"]

    classification_candidates = {
        "logistic": {
            "features": extended_feature_columns,
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                ]
            ),
        },
        "random_forest": {
            "features": extended_feature_columns,
            "threshold_mode": "fixed",
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=6,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        },
        "random_forest_tuned": {
            "features": extended_feature_columns,
            "threshold_mode": "time_series_cv",
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=6,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        },
        "rf_simple_baseline": {
            "features": simple_rf_feature_columns,
            "threshold_mode": "fixed",
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
                ]
            ),
        },
        "hist_gradient_boosting": {
            "features": extended_feature_columns,
            "threshold_mode": "fixed",
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=200,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        },
    }

    classification_results: list[dict[str, Any]] = []
    for name, candidate in classification_candidates.items():
        candidate_features = candidate["features"]
        model = candidate["pipeline"]
        threshold_details = {"decision_threshold": 0.5, "threshold_source": "fixed"}
        if candidate.get("threshold_mode") == "time_series_cv":
            threshold_details = tune_threshold_with_time_series_cv(
                model,
                train_df[candidate_features],
                y_train,
            )
        model.fit(train_df[candidate_features], y_train)
        probabilities = positive_class_probabilities(model, test_df[candidate_features])
        predictions = (probabilities >= threshold_details["decision_threshold"]).astype(int)
        classification_results.append(
            {
                "name": name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "latest_up_probability": float(positive_class_probabilities(model, latest_features[candidate_features])[0]),
                "feature_count": len(candidate_features),
                **threshold_details,
            }
        )

    best_classification = select_best_model(classification_results, ["balanced_accuracy", "roc_auc"])
    direction_candidate = classification_candidates[best_classification["name"]]
    direction_model = direction_candidate["pipeline"]
    direction_feature_columns = direction_candidate["features"]
    direction_model.fit(training_data[direction_feature_columns], training_data["target_up"])
    regression_feature_columns = extended_feature_columns

    regression_candidates = {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    regression_results: list[dict[str, Any]] = []
    for name, model in regression_candidates.items():
        model.fit(train_df[regression_feature_columns], train_df["next_return"])
        predictions = model.predict(test_df[regression_feature_columns])
        regression_results.append(
            {
                "name": name,
                "mae": float(mean_absolute_error(test_df["next_return"], predictions)),
                "rmse": float(mean_squared_error(test_df["next_return"], predictions) ** 0.5),
                "r2": float(r2_score(test_df["next_return"], predictions)),
                "latest_predicted_return": float(model.predict(latest_features[regression_feature_columns])[0]),
            }
        )

    best_regression = sorted(regression_results, key=lambda item: (item["mae"], item["rmse"]))[0]
    return_model = regression_candidates[best_regression["name"]]
    return_model.fit(training_data[regression_feature_columns], training_data["next_return"])

    ridge_equation_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge_equation_model.fit(training_data[regression_feature_columns], training_data["next_return"])
    ridge_coefficients = ridge_equation_model.named_steps["model"].coef_
    ridge_intercept = ridge_equation_model.named_steps["model"].intercept_

    next_session_date = (latest_features["date"].iloc[0] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    latest_up_probability = float(positive_class_probabilities(direction_model, latest_features[direction_feature_columns])[0])
    latest_predicted_return = float(return_model.predict(latest_features[regression_feature_columns])[0])

    if latest_up_probability >= 0.6 and latest_predicted_return > 0.005:
        forecast_label = "偏强"
    elif latest_up_probability <= 0.4 and latest_predicted_return < -0.005:
        forecast_label = "偏弱"
    else:
        forecast_label = "震荡偏中性"

    direction_payload = {
        "model": direction_model,
        "features": direction_feature_columns,
        "selected_model": best_classification["name"],
        "evaluation": best_classification,
        "generated_at": datetime.now().isoformat(),
    }
    return_payload = {
        "model": return_model,
        "features": regression_feature_columns,
        "selected_model": best_regression["name"],
        "evaluation": best_regression,
        "generated_at": datetime.now().isoformat(),
        "ridge_equation": {
            "intercept": float(ridge_intercept),
            "coefficients": {
                feature: float(value) for feature, value in zip(regression_feature_columns, ridge_coefficients)
            },
        },
    }
    joblib.dump(direction_payload, paths.direction_model_path)
    joblib.dump(return_payload, paths.return_model_path)

    prices = pd.read_csv(paths.stock_prices_csv, parse_dates=["date"])
    benchmark = pd.read_csv(paths.benchmark_prices_csv, parse_dates=["date"])
    benchmark = benchmark.rename(columns={"benchmark_close": "close"})
    benchmark = benchmark if "close" in benchmark.columns else benchmark.rename(columns={"close": "close"})

    stock_start = prices.iloc[0]
    stock_end = prices.iloc[-1]
    benchmark_end = benchmark.iloc[-1]
    benchmark_start = benchmark.iloc[0]
    rolling_peak = prices["close"].cummax()
    max_drawdown = float((prices["close"] / rolling_peak - 1).min())
    source_type_counts = {key: int(value) for key, value in company_news["source_type"].value_counts().to_dict().items()}
    google_decode_ok = int((google_news["decode_status"] == "ok").sum()) if "decode_status" in google_news.columns else 0
    google_extract_ok = int((google_news["extract_status"] == "ok").sum()) if "extract_status" in google_news.columns else 0
    pdf_extract_ok = int((pdf_cache["extract_status"] == "ok").sum()) if not pdf_cache.empty else 0
    pdf_extract_short = int((pdf_cache["extract_status"] == "short").sum()) if not pdf_cache.empty else 0
    pdf_source_counts = (
        {key: int(value) for key, value in pdf_cache["source_type"].value_counts().to_dict().items()}
        if not pdf_cache.empty
        else {}
    )

    metrics = {
        "config": {
            "company_name": config.company_name,
            "symbol": config.symbol,
            "benchmark_symbol": config.benchmark_symbol,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "generated_at": datetime.now().isoformat(),
        },
        "data_summary": {
            "price_rows": len(prices),
            "training_rows": len(training_data),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
            "train_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
            "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
            "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
            "news_items": len(company_news),
            "news_days": int(company_news["effective_date"].nunique()),
            "source_type_counts": source_type_counts,
            "quarterly_financial_rows": len(quarterly_financials),
            "google_decode_ok": google_decode_ok,
            "google_extract_ok": google_extract_ok,
            "pdf_extract_ok": pdf_extract_ok,
            "pdf_extract_short": pdf_extract_short,
            "pdf_source_counts": pdf_source_counts,
        },
        "descriptive_summary": {
            "stock_close_start": float(stock_start["close"]),
            "stock_close_end": float(stock_end["close"]),
            "stock_total_return": float(stock_end["close"] / stock_start["close"] - 1),
            "benchmark_total_return": float(benchmark_end["close"] / benchmark_start["close"] - 1),
            "excess_return_vs_benchmark": float(
                (stock_end["close"] / stock_start["close"] - 1)
                - (benchmark_end["close"] / benchmark_start["close"] - 1)
            ),
            "max_drawdown": max_drawdown,
            "latest_20d_return": float(prices["close"].iloc[-1] / prices["close"].iloc[-21] - 1)
            if len(prices) > 20
            else None,
            "latest_60d_return": float(prices["close"].iloc[-1] / prices["close"].iloc[-61] - 1)
            if len(prices) > 60
            else None,
        },
        "classification": {
            "selected_model": best_classification["name"],
            "baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
            "candidates": classification_results,
            "selected_metrics": best_classification,
            "feature_importance": extract_feature_importance(direction_model, direction_feature_columns),
            "feature_sets": {
                "extended": extended_feature_columns,
                "rf_simple_baseline": simple_rf_feature_columns,
            },
        },
        "regression": {
            "selected_model": best_regression["name"],
            "candidates": regression_results,
            "selected_metrics": best_regression,
            "ridge_equation": {
                "intercept": float(ridge_intercept),
                "coefficients": {
                    feature: float(value) for feature, value in zip(regression_feature_columns, ridge_coefficients)
                },
            },
        },
        "forecast": {
            "signal_date": latest_features["date"].iloc[0].strftime("%Y-%m-%d"),
            "next_session_date": next_session_date,
            "latest_close": float(latest_features["close"].iloc[0]),
            "next_up_probability": latest_up_probability,
            "predicted_return": latest_predicted_return,
            "label": forecast_label,
        },
    }
    save_json(metrics, paths.metrics_json_path)
    return metrics


def recent_month_summary(company_news: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    monthly = (
        company_news.assign(month=company_news["effective_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(news_count=("title", "count"), avg_sentiment=("sentiment_score", "mean"))
        .reset_index()
        .sort_values("month")
    )
    return monthly.tail(months)


def write_report(config: PipelineConfig, paths: PipelinePaths) -> Path:
    metrics = json.loads(paths.metrics_json_path.read_text(encoding="utf-8"))
    company_news = pd.read_csv(paths.company_news_csv, parse_dates=["timestamp", "effective_date"])
    monthly_summary = recent_month_summary(company_news, months=8)

    positive_events = company_news.sort_values("sentiment_score", ascending=False).head(3)
    negative_events = company_news.sort_values("sentiment_score", ascending=True).head(3)
    latest_events = company_news.sort_values("timestamp", ascending=False).head(5)

    feature_importance = metrics["classification"]["feature_importance"][:8]
    importance_lines = [
        f"- `{item['feature']}`: {item['importance']:.4f}" for item in feature_importance
    ] or ["- 无可解释特征重要性输出"]

    monthly_lines = [
        f"- `{row.month}`: 新闻/公告 {int(row.news_count)} 条, 平均情绪 {row.avg_sentiment:.3f}"
        for row in monthly_summary.itertuples()
    ]

    positive_lines = [
        f"- `{row.timestamp:%Y-%m-%d}` [{row.title}]({row.source_link}) | 情绪分数 `{row.sentiment_score:.3f}`"
        for row in positive_events.itertuples()
    ]
    negative_lines = [
        f"- `{row.timestamp:%Y-%m-%d}` [{row.title}]({row.source_link}) | 情绪分数 `{row.sentiment_score:.3f}`"
        for row in negative_events.itertuples()
    ]
    latest_lines = [
        f"- `{row.timestamp:%Y-%m-%d %H:%M}` [{row.title}]({row.source_link}) | 来源 `{row.source_type}` / `{row.source_name}`"
        for row in latest_events.itertuples()
    ]
    source_counts = metrics["data_summary"]["source_type_counts"]
    source_lines = [
        f"- `{source}`: {count} 条"
        for source, count in sorted(source_counts.items(), key=lambda item: item[0])
    ]
    google_decode_ok = metrics["data_summary"]["google_decode_ok"]
    google_extract_ok = metrics["data_summary"]["google_extract_ok"]
    quarterly_financial_rows = metrics["data_summary"]["quarterly_financial_rows"]
    pdf_extract_ok = metrics["data_summary"]["pdf_extract_ok"]
    pdf_extract_short = metrics["data_summary"]["pdf_extract_short"]
    pdf_source_counts = metrics["data_summary"]["pdf_source_counts"]
    pdf_source_lines = [
        f"- `{source}`: {count} 份"
        for source, count in sorted(pdf_source_counts.items(), key=lambda item: item[0])
    ] or ["- 无 PDF 正文缓存"]

    stock_total_return = metrics["descriptive_summary"]["stock_total_return"] * 100
    benchmark_total_return = metrics["descriptive_summary"]["benchmark_total_return"] * 100
    excess_return = metrics["descriptive_summary"]["excess_return_vs_benchmark"] * 100
    max_drawdown = metrics["descriptive_summary"]["max_drawdown"] * 100
    classifier_metrics = metrics["classification"]["selected_metrics"]
    regression_metrics = metrics["regression"]["selected_metrics"]
    forecast = metrics["forecast"]
    baseline_accuracy = metrics["classification"]["baseline_accuracy"]
    classification_candidates = metrics["classification"]["candidates"]
    simple_rf_candidate = next(
        (item for item in classification_candidates if item["name"] == "rf_simple_baseline"),
        None,
    )
    model_display_names = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "random_forest_tuned": "Random Forest (TimeSeriesSplit 调阈值)",
        "rf_simple_baseline": "Random Forest (简化基线)",
        "hist_gradient_boosting": "Hist Gradient Boosting",
        "ridge": "Ridge Regression",
        "random_forest_regressor": "Random Forest Regressor",
    }
    classification_candidate_lines = [
        (
            f"- `{model_display_names.get(item['name'], item['name'])}`:"
            f" accuracy `{item['accuracy']:.4f}` |"
            f" balanced_accuracy `{item['balanced_accuracy']:.4f}` |"
            f" roc_auc `{item['roc_auc']:.4f}` |"
            f" 阈值 `{item.get('decision_threshold', 0.5):.2f}` |"
            f" 特征数 `{item.get('feature_count', 0)}`"
        )
        for item in classification_candidates
    ]
    random_forest_candidate = next(
        (item for item in classification_candidates if item["name"] == "random_forest"),
        None,
    )
    random_forest_tuned_candidate = next(
        (item for item in classification_candidates if item["name"] == "random_forest_tuned"),
        None,
    )

    if classifier_metrics["accuracy"] > baseline_accuracy:
        classifier_read = "分类模型在 plain accuracy 上略高于简单多数基线，但优势仍然很弱。"
    elif classifier_metrics["accuracy"] == baseline_accuracy:
        classifier_read = "分类模型在 plain accuracy 上只是追平简单多数基线，但平衡准确率和 ROC AUC 略优于随机。"
    else:
        classifier_read = (
            "分类模型在 plain accuracy 上没有跑赢简单多数基线，只能说平衡准确率和 ROC AUC 略高于随机。"
        )

    if forecast["next_up_probability"] >= 0.6 and forecast["predicted_return"] > 0.005:
        forecast_read = "上涨概率和预测收益都偏正，这更像偏强信号，但还远谈不上高确信度。"
    elif forecast["next_up_probability"] <= 0.4 and forecast["predicted_return"] < -0.005:
        forecast_read = "上涨概率和预测收益都偏弱，短线更像偏谨慎信号。"
    else:
        forecast_read = "概率和预测收益都靠近中轴，这不是明确突破或转弱，更像震荡偏中性。"

    if simple_rf_candidate is None:
        simple_rf_read = "这次没有成功写出简化随机森林基线结果。"
    elif simple_rf_candidate["accuracy"] >= classifier_metrics["accuracy"]:
        simple_rf_read = (
            "黄金项目启发出的简化随机森林基线并不比当前最优分类模型差，说明少量价量 + 新闻聚合特征已经带出了一部分信号。"
        )
    else:
        simple_rf_read = (
            "黄金项目风格的简化随机森林基线能作为对照，但当前还是扩展特征模型更强一些，说明财务和多源新闻统计仍然提供了补充信息。"
        )

    if random_forest_candidate and random_forest_tuned_candidate:
        if random_forest_tuned_candidate["balanced_accuracy"] > random_forest_candidate["balanced_accuracy"]:
            rf_threshold_read = (
                f"对扩展特征随机森林做 TimeSeriesSplit 阈值调优后，balanced accuracy 从 "
                f"`{random_forest_candidate['balanced_accuracy']:.4f}` 提高到 "
                f"`{random_forest_tuned_candidate['balanced_accuracy']:.4f}`，说明它不完全是模型本身无效，"
                f"阈值确实影响了方向判断。"
            )
        elif random_forest_tuned_candidate["balanced_accuracy"] == random_forest_candidate["balanced_accuracy"]:
            rf_threshold_read = (
                "对扩展特征随机森林做 TimeSeriesSplit 阈值调优后，balanced accuracy 没有变化，说明问题不只是 `0.5` 阈值。"
            )
        else:
            rf_threshold_read = (
                f"对扩展特征随机森林做 TimeSeriesSplit 阈值调优后，balanced accuracy 从 "
                f"`{random_forest_candidate['balanced_accuracy']:.4f}` 变为 "
                f"`{random_forest_tuned_candidate['balanced_accuracy']:.4f}`，没有变好，"
                "说明当前随机森林的主要问题不只是 `0.5` 阈值。"
            )
    else:
        rf_threshold_read = "这次没有成功产出随机森林阈值调优结果。"

    report = f"""# 当升科技(300073) 两年新闻与股价 ETL + ML 分析

生成时间: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

## 结论先看

- 这套模型可以跑通，而且可以每周更新，但**预测能力只有弱信号水平**。
- 在 `2024-03-11` 到 `2026-03-11` 的样本内，当升科技前复权收盘价从 `{metrics["descriptive_summary"]["stock_close_start"]:.2f}` 走到 `{metrics["descriptive_summary"]["stock_close_end"]:.2f}`，区间涨幅约 `{stock_total_return:.2f}%`。
- 同期创业板指近似涨幅约 `{benchmark_total_return:.2f}%`，当升科技相对超额收益约 `{excess_return:.2f}%`。
- 区间最大回撤约 `{max_drawdown:.2f}%`，波动不低，说明它更适合列入**重点跟踪池**，不适合只靠单一新闻模型做重仓判断。

## 数据与方法

- 股价来源: `AKShare -> 东方财富 A 股历史行情`
- 新闻/信息来源:
  - `CNInfo` 公司公告
  - `CNInfo` 材料性公告 PDF 正文
  - `东方财富` 个股研报
  - `东方财富` 券商研报 PDF 正文
  - `东方财富` 个股新闻(最近一段)
  - `Google News RSS` 按月回溯抓取的外部媒体报道
- 时间范围: `{config.start_date}` 到 `{config.end_date}`
- 合并后的有效新闻/公告样本数: `{metrics["data_summary"]["news_items"]}`
- 发生在交易日映射内的有效消息日: `{metrics["data_summary"]["news_days"]}`
- Google News 解码成功数: `{google_decode_ok}`
- Google News 正文抽取成功数: `{google_extract_ok}`
- PDF 正文抽取成功数: `{pdf_extract_ok}`
- PDF 正文抽取较短但可用数: `{pdf_extract_short}`
- 季度财务快照数: `{quarterly_financial_rows}`
- 来源分布:
{chr(10).join(source_lines)}
- PDF 来源分布:
{chr(10).join(pdf_source_lines)}
- 建模目标:
  - 分类: 预测下一交易日涨/跌
  - 回归: 预测下一交易日收益率
- 特征:
  - 价格/成交量/换手率/振幅
  - 1/3/5 日收益率
  - 5/10/20/60 日均线偏离
  - 5/20 日波动率
  - 创业板指收益率
  - 公告日期生效的季度财务指标(收入、利润、ROE、毛利率、负债率、经营现金流、同比增速)
  - 公告/研报/新闻数量与情绪聚合

## 模型结果

- 分类模型最终选择: `{metrics["classification"]["selected_model"]}`
- 测试集准确率: `{classifier_metrics["accuracy"]:.4f}`
- 测试集平衡准确率: `{classifier_metrics["balanced_accuracy"]:.4f}`
- 测试集 ROC AUC: `{classifier_metrics["roc_auc"]:.4f}`
- 同期简单基线(永远猜多数方向): `{metrics["classification"]["baseline_accuracy"]:.4f}`
- 分类候选对比:
{chr(10).join(classification_candidate_lines)}
- 回归模型最终选择: `{metrics["regression"]["selected_model"]}`
- 回归 MAE: `{regression_metrics["mae"]:.4f}`
- 回归 RMSE: `{regression_metrics["rmse"]:.4f}`
- 回归 R²: `{regression_metrics["r2"]:.4f}`

解释:

- {classifier_read}
- {rf_threshold_read}
- {simple_rf_read}
- 回归模型的 `R²` 仍为负值，说明它对下一日精确收益率的解释力不足。
- 所以这套结果更适合用来做**辅助观察**，不适合单独作为买卖依据。

## 最新预测

- 信号日期: `{forecast["signal_date"]}`
- 基准收盘价: `{forecast["latest_close"]:.2f}`
- 下一交易时点(近似): `{forecast["next_session_date"]}`
- 上涨概率: `{forecast["next_up_probability"]:.4f}`
- 预测收益率: `{forecast["predicted_return"]:.4f}`
- 模型标签: `{forecast["label"]}`

我对这个结果的解释:

- {forecast_read}
- 如果你是做投资跟踪，这意味着它值得继续放在观察名单里，但还不支持仅凭模型发出高确信交易结论。

## 最近 8 个月信息流

{chr(10).join(monthly_lines)}

从最近几个月看:

- `2025-10`、`2026-01`、`2026-02` 的平均情绪更正面。
- `2025-12` 虽然消息多，但情绪均值回落，说明信息密集不等于确定性更高。
- 外部媒体补充进来后，消息覆盖明显更完整，能更接近“公司被市场如何讨论”的真实样子。
- 公告和研报接入 PDF 正文后，情绪特征不再只靠标题，信息密度更高。
- 财务因子并入后，模型不再只看短线价格噪声，而是把季度基本面状态也考虑进去。

## 情绪最强的正面事件

{chr(10).join(positive_lines)}

## 情绪最强的负面事件

{chr(10).join(negative_lines)}

## 最近披露/新闻

{chr(10).join(latest_lines)}

## 模型最看重的特征

{chr(10).join(importance_lines)}

## 业务合作 / 投资关注判断

- 作为**业务合作候选**: 可以关注，但要做更深尽调。理由是它公告披露频率高、研报覆盖持续、技术路线仍在被市场跟踪，说明公开透明度和行业关注度都不低。
- 作为**投资关注对象**: 值得持续跟踪，但不适合只凭这套模型的单次输出就下强结论。真正更关键的变量仍然是业绩兑现、正极材料价格、下游电池需求、海外客户订单和固态电池合作落地情况。
- 如果你要做合作，建议额外核查:
  - 近四个季度毛利与现金流波动
  - 大客户集中度和回款条款
  - 海外子公司、补助、扩产和合作协议的实际执行进度
  - 固态电池合作是否已进入有收入贡献的阶段

## 限制

- 这套样本已经比最初版本更完整，但仍不是完整全网历史库，尤其对未被搜索引擎稳定索引的站点覆盖有限。
- 即便加入 Google News 月度回溯，也仍然不是完整全网历史库，部分站点会缺失或被 Google 索引节奏影响。
- 中文金融情绪分析存在噪声，尤其对公告标题这种偏正式文本。
- 下一日涨跌本来就高度随机，所以 `50%` 到 `55%` 左右的准确率只能视为弱信号。

## 输出文件

- `data/raw/stock_prices.csv`
- `data/raw/disclosures.csv`
- `data/raw/research_reports.csv`
- `data/raw/recent_stock_news.csv`
- `data/raw/google_news_archive.csv`
- `data/raw/financial_abstract.csv`
- `data/raw/financial_report.csv`
- `data/raw/pdf_text_cache.csv`
- `data/processed/company_news.csv`
- `data/processed/daily_news_features.csv`
- `data/processed/quarterly_financials.csv`
- `data/processed/training_data.csv`
- `models/direction_model.joblib`
- `models/return_model.joblib`
- `reports/metrics.json`
"""
    paths.report_md_path.write_text(report, encoding="utf-8")
    return paths.report_md_path


def run_pipeline(project_root: Path, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    config = PipelineConfig(
        project_root=project_root,
        start_date=start_date or "20240311",
        end_date=end_date or date.today().strftime("%Y%m%d"),
    )
    paths = make_paths(project_root)
    fetch_stock_prices(config, paths)
    fetch_company_news(config, paths)
    fetch_pdf_text_cache(paths)
    fetch_financial_data(config, paths)
    build_training_dataset(config, paths)
    metrics = train_models(config, paths)
    write_report(config, paths)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Easpring stock/news ETL + ML pipeline.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root for outputs.",
    )
    parser.add_argument("--start-date", default="20240311", help="Start date in YYYYMMDD format.")
    parser.add_argument(
        "--end-date",
        default=date.today().strftime("%Y%m%d"),
        help="End date in YYYYMMDD format.",
    )
    args = parser.parse_args()

    metrics = run_pipeline(args.project_root, start_date=args.start_date, end_date=args.end_date)
    print(
        json.dumps(
            {
                "selected_direction_model": metrics["classification"]["selected_model"],
                "selected_return_model": metrics["regression"]["selected_model"],
                "forecast": metrics["forecast"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
