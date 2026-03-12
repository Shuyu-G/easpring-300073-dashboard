from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pandas as pd
from plotly.offline import get_plotlyjs


SOURCE_LABELS = {
    "disclosure": "公司披露",
    "research": "卖方研报",
    "news": "市场新闻",
    "web_news": "媒体报道",
}


def classify_impact(score: float) -> str:
    if pd.isna(score):
        return "中性"
    if score >= 0.35:
        return "偏利多"
    if score <= -0.35:
        return "偏利空"
    return "中性"


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def impact_key_from_label(label: str | None) -> str:
    mapping = {
        "偏利多": "positive",
        "中性": "neutral",
        "偏利空": "negative",
    }
    return mapping.get(label or "", "neutral")


def signal_key_from_label(label: str | None) -> str:
    mapping = {
        "偏强": "strong",
        "偏弱": "weak",
        "震荡偏中性": "neutral",
    }
    return mapping.get(label or "", "neutral")


def render_nav(current_page: str) -> str:
    items = [
        ("index.html", "总览", "home", "nav.home"),
        ("news.html", "新闻", "news", "nav.news"),
        ("model.html", "模型", "model", "nav.model"),
        ("fundamentals.html", "基本面", "fundamentals", "nav.fundamentals"),
    ]
    links: list[str] = []
    for href, label, page, i18n_key in items:
        active = " active" if page == current_page else ""
        links.append(
            f'<a class="nav-link{active}" href="{href}" data-i18n="{i18n_key}">{label}</a>'
        )
    return "".join(links)


def render_head(title: str, page: str, asset_version: str) -> str:
    return dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{title}</title>
          <link rel="stylesheet" href="assets/styles.css?v={asset_version}" />
          <script>window.__DASHBOARD_VERSION__ = "{asset_version}";</script>
          <script defer src="assets/plotly.min.js?v={asset_version}"></script>
          <script defer src="assets/app.js?v={asset_version}"></script>
        </head>
        <body data-page="{page}">
          <div class="app-shell">
            <header class="topbar">
              <div>
                <p class="eyebrow">EASPRING 300073</p>
                <h1 data-i18n="site.title">当升科技跟踪 Dashboard</h1>
                <p class="subtle" data-i18n="site.subtitle">股价、新闻、模型与基本面，每个工作日自动刷新一次。</p>
              </div>
              <div class="topbar-meta">
                <nav class="topnav">{render_nav(page)}</nav>
                <div class="lang-switcher" role="group" aria-label="Language switcher">
                  <span class="lang-caption" data-i18n="lang.label">语言</span>
                  <div class="lang-buttons">
                    <button type="button" class="lang-button" data-lang="zh" aria-label="中文">
                      <span class="flag" aria-hidden="true">🇨🇳</span>
                      <span>中文</span>
                    </button>
                    <button type="button" class="lang-button" data-lang="en" aria-label="English">
                      <span class="flag" aria-hidden="true">🇺🇸</span>
                      <span>English</span>
                    </button>
                    <button type="button" class="lang-button" data-lang="de" aria-label="Deutsch">
                      <span class="flag" aria-hidden="true">🇩🇪</span>
                      <span>Deutsch</span>
                    </button>
                  </div>
                </div>
              </div>
            </header>
        """
    )


def render_footer() -> str:
    return dedent(
        """\
            <footer class="footer">
              <p data-i18n="site.footer">数据源包含日线行情、公司公告、券商研报、外部媒体和季度财务摘要。页面为静态前端站点，非实时盘中系统。</p>
            </footer>
          </div>
        </body>
        </html>
        """
    )


def build_snapshot_payload(project_root: Path) -> dict[str, object]:
    metrics = json.loads((project_root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(project_root / "data" / "raw" / "stock_prices.csv", parse_dates=["date"])
    benchmark = pd.read_csv(
        project_root / "data" / "raw" / "benchmark_prices.csv",
        parse_dates=["date"],
    )
    news = pd.read_csv(
        project_root / "data" / "processed" / "company_news.csv",
        parse_dates=["timestamp", "effective_date"],
    )
    quarterly = pd.read_csv(
        project_root / "data" / "processed" / "quarterly_financials.csv",
        parse_dates=["报告日", "公告日期", "effective_timestamp", "effective_date"],
    )

    prices = prices.sort_values("date").reset_index(drop=True)
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    news = news.sort_values("timestamp", ascending=False).reset_index(drop=True)
    quarterly = quarterly.sort_values("报告日").reset_index(drop=True)

    news["body"] = news["body"].fillna("")
    news["source_label"] = news["source_type"].map(SOURCE_LABELS).fillna(news["source_type"])
    news["impact"] = news["sentiment_score"].apply(classify_impact)
    news["impact_key"] = news["impact"].apply(impact_key_from_label)
    news["month"] = news["effective_date"].dt.strftime("%Y-%m")
    news["timestamp_label"] = news["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    news["effective_label"] = news["effective_date"].dt.strftime("%Y-%m-%d")
    news["body_excerpt"] = news["body"].str.replace(r"\s+", " ", regex=True).str.slice(0, 220)

    perf = prices[["date", "close"]].merge(
        benchmark[["date", "benchmark_close"]],
        on="date",
        how="left",
    )
    perf["stock_index"] = perf["close"] / perf["close"].iloc[0] * 100
    perf["benchmark_index"] = perf["benchmark_close"] / perf["benchmark_close"].iloc[0] * 100
    rolling_high = prices["close"].cummax()
    drawdown = prices["close"] / rolling_high - 1
    latest_close = float(prices["close"].iloc[-1])
    latest_date = prices["date"].iloc[-1]
    last_20_start = max(len(prices) - 21, 0)
    change_20d = prices["close"].iloc[-1] / prices["close"].iloc[last_20_start] - 1
    relative_return = perf["stock_index"].iloc[-1] / perf["benchmark_index"].iloc[-1] - 1
    news_counts = news["impact"].value_counts().to_dict()
    positive_item = news.sort_values("sentiment_score", ascending=False).head(1)
    negative_item = news.sort_values("sentiment_score", ascending=True).head(1)
    classification = metrics.get("classification", {})
    regression = metrics.get("regression", {})
    forecast = metrics.get("forecast", {})
    selected_metrics = classification.get("selected_metrics", {})
    regression_metrics = regression.get("selected_metrics", {})
    updated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    latest_quarterly = quarterly[quarterly["effective_date"] <= latest_date].tail(8).copy()
    if latest_quarterly.empty:
        latest_quarterly = quarterly.tail(8).copy()

    return {
        "generated_at": updated_at,
        "generated_at_label": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "metrics": metrics,
        "summary": {
            "latest_close": latest_close,
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "change_20d": float(change_20d),
            "relative_return": float(relative_return),
            "max_drawdown": float(drawdown.min()),
            "news_total": int(len(news)),
            "news_days": int(metrics.get("data_summary", {}).get("news_days", news["effective_date"].nunique())),
            "classification_accuracy": selected_metrics.get("accuracy"),
            "classification_balanced_accuracy": selected_metrics.get("balanced_accuracy"),
            "classification_auc": selected_metrics.get("roc_auc"),
            "classification_baseline": classification.get("baseline_accuracy"),
            "regression_mae": regression_metrics.get("mae"),
            "regression_rmse": regression_metrics.get("rmse"),
            "regression_r2": regression_metrics.get("r2"),
            "positive_news_count": int(news_counts.get("偏利多", 0)),
            "neutral_news_count": int(news_counts.get("中性", 0)),
            "negative_news_count": int(news_counts.get("偏利空", 0)),
        },
        "signal": {
            "signal_date": forecast.get("signal_date"),
            "next_session_date": forecast.get("next_session_date"),
            "latest_close": forecast.get("latest_close"),
            "up_probability": forecast.get("next_up_probability"),
            "predicted_return": forecast.get("predicted_return"),
            "label": forecast.get("label"),
            "label_key": signal_key_from_label(forecast.get("label")),
        },
        "highlights": {
            "positive_event": positive_item[
                ["timestamp_label", "source_label", "source_type", "impact", "impact_key", "title", "source_link", "sentiment_score"]
            ]
            .to_dict(orient="records"),
            "negative_event": negative_item[
                ["timestamp_label", "source_label", "source_type", "impact", "impact_key", "title", "source_link", "sentiment_score"]
            ]
            .to_dict(orient="records"),
        },
        "models": {
            "classification": {
                "selected_model": classification.get("selected_model"),
                "baseline_accuracy": classification.get("baseline_accuracy"),
                "candidates": classification.get("candidates", []),
                "selected_metrics": classification.get("selected_metrics", {}),
                "feature_importance": classification.get("feature_importance", [])[:15],
            },
            "regression": {
                "selected_model": regression.get("selected_model"),
                "candidates": regression.get("candidates", []),
                "selected_metrics": regression.get("selected_metrics", {}),
                "ridge_equation": regression.get("ridge_equation"),
            },
        },
        "prices": [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "close": float(row.close),
                "volume": float(row.volume),
                "pct_change": float(row.pct_change),
                "stock_index": float(stock_index),
                "benchmark_index": float(benchmark_index),
            }
            for row, stock_index, benchmark_index in zip(
                prices.itertuples(index=False),
                perf["stock_index"],
                perf["benchmark_index"],
            )
        ],
        "quarterly": [
            {
                "report_date": row.报告日.strftime("%Y-%m-%d"),
                "revenue": None if pd.isna(row.fin_revenue) else float(row.fin_revenue),
                "net_profit": None if pd.isna(row.fin_net_profit) else float(row.fin_net_profit),
                "gross_margin": None if pd.isna(row.fin_gross_margin) else float(row.fin_gross_margin),
                "net_margin": None if pd.isna(row.fin_net_margin) else float(row.fin_net_margin),
                "roe": None if pd.isna(row.fin_roe) else float(row.fin_roe),
                "debt_ratio": None if pd.isna(row.fin_debt_ratio) else float(row.fin_debt_ratio),
                "revenue_yoy": None if pd.isna(row.fin_revenue_yoy) else float(row.fin_revenue_yoy),
                "profit_yoy": None if pd.isna(row.fin_profit_yoy) else float(row.fin_profit_yoy),
                "announce_date": row.公告日期.strftime("%Y-%m-%d") if not pd.isna(row.公告日期) else None,
            }
            for row in latest_quarterly.itertuples(index=False)
        ],
        "news": [
            {
                "timestamp": row.timestamp_label,
                "effective_date": row.effective_label,
                "month": row.month,
                "source_label": row.source_label,
                "source_type": row.source_type,
                "impact": row.impact,
                "impact_key": row.impact_key,
                "title": row.title,
                "source_link": row.source_link,
                "source_name": row.source_name,
                "sentiment_score": None if pd.isna(row.sentiment_score) else float(row.sentiment_score),
                "body_excerpt": row.body_excerpt,
            }
            for row in news.itertuples(index=False)
        ],
    }


def build_stylesheet() -> str:
    return dedent(
        """\
        :root {
          --bg: #f4efe6;
          --panel: #fbf8f3;
          --line: #d8cfc0;
          --text: #161a18;
          --muted: #59635e;
          --up: #c65a2e;
          --down: #1f6f5f;
          --neutral: #58606b;
          --shadow: 0 18px 50px rgba(86, 96, 107, 0.08);
          --shadow-strong: 0 24px 66px rgba(86, 96, 107, 0.14);
          --radius: 22px;
          --ease: cubic-bezier(0.22, 1, 0.36, 1);
        }

        html {
          scroll-behavior: smooth;
        }

        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          font-family: "Avenir Next", "PingFang SC", "Noto Serif SC", serif;
          color: var(--text);
          background:
            radial-gradient(circle at top right, rgba(198, 90, 46, 0.10), transparent 24%),
            radial-gradient(circle at top left, rgba(31, 111, 95, 0.10), transparent 18%),
            linear-gradient(180deg, #f3eee4 0%, #f8f5ef 46%, #f1ebdf 100%);
          overflow-x: hidden;
        }

        a {
          color: inherit;
          text-decoration: none;
        }

        button,
        select {
          font-family: inherit;
        }

        .app-shell {
          max-width: 1380px;
          margin: 0 auto;
          padding: 20px 18px 36px;
        }

        .topbar {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 18px;
          margin-bottom: 18px;
          animation: rise-in 0.72s var(--ease) both;
        }

        .topbar h1 {
          margin: 0.18rem 0 0.3rem;
          font-size: clamp(1.9rem, 4vw, 2.7rem);
          line-height: 1.05;
        }

        .eyebrow {
          margin: 0;
          font-size: 0.8rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          font-weight: 700;
          color: var(--down);
        }

        .subtle {
          margin: 0;
          color: var(--muted);
          font-size: 0.95rem;
        }

        .topnav {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }

        .topbar-meta {
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
          justify-content: flex-end;
        }

        .nav-link {
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid var(--line);
          background: rgba(251, 248, 243, 0.9);
          color: #38413d;
          font-size: 0.92rem;
          font-weight: 700;
          transition:
            transform 180ms var(--ease),
            box-shadow 180ms var(--ease),
            border-color 180ms var(--ease),
            background 180ms var(--ease);
        }

        .nav-link.active {
          background: linear-gradient(135deg, rgba(198, 90, 46, 0.12), rgba(31, 111, 95, 0.08));
          border-color: rgba(198, 90, 46, 0.35);
          box-shadow: inset 0 0 0 1px rgba(198, 90, 46, 0.08);
        }

        .nav-link:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 28px rgba(86, 96, 107, 0.10);
          border-color: rgba(198, 90, 46, 0.28);
        }

        .lang-switcher {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 12px 10px 14px;
          border-radius: 20px;
          border: 1px solid rgba(198, 90, 46, 0.20);
          background: linear-gradient(135deg, rgba(251, 248, 243, 0.95), rgba(247, 241, 233, 0.98));
          box-shadow: 0 20px 40px rgba(86, 96, 107, 0.08);
        }

        .lang-caption {
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--muted);
          font-weight: 700;
        }

        .lang-buttons {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }

        .lang-button {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border: 1px solid rgba(216, 207, 192, 0.95);
          border-radius: 999px;
          padding: 8px 12px;
          background: rgba(255, 255, 255, 0.82);
          color: #38413d;
          font-size: 0.9rem;
          font-weight: 700;
          cursor: pointer;
          transition:
            transform 180ms var(--ease),
            box-shadow 180ms var(--ease),
            border-color 180ms var(--ease),
            background 180ms var(--ease);
        }

        .lang-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 28px rgba(86, 96, 107, 0.12);
          border-color: rgba(198, 90, 46, 0.28);
        }

        .lang-button.active {
          background: linear-gradient(135deg, rgba(198, 90, 46, 0.18), rgba(31, 111, 95, 0.12));
          border-color: rgba(198, 90, 46, 0.35);
          color: #1f2622;
          box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.55), 0 16px 32px rgba(86, 96, 107, 0.11);
        }

        .lang-button .flag {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 1.2rem;
          font-size: 1rem;
        }

        .banner {
          margin-bottom: 16px;
          padding: 14px 16px;
          border-radius: 16px;
          border: 1px solid var(--line);
          background: rgba(251, 248, 243, 0.9);
          box-shadow: var(--shadow);
          display: none;
          animation: rise-in 0.72s var(--ease) both;
        }

        .banner.show {
          display: block;
        }

        .banner.warning {
          border-color: rgba(198, 90, 46, 0.35);
          background: rgba(252, 241, 235, 0.95);
        }

        .hero,
        .panel {
          border: 1px solid var(--line);
          border-radius: var(--radius);
          background: rgba(251, 248, 243, 0.93);
          box-shadow: var(--shadow);
          position: relative;
          overflow: hidden;
          transition:
            transform 220ms var(--ease),
            box-shadow 220ms var(--ease),
            border-color 220ms var(--ease),
            background 220ms var(--ease);
          animation: rise-in 0.76s var(--ease) both;
        }

        .hero::before,
        .panel::before,
        .news-card::before,
        .highlight-card::before {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.16) 45%, transparent 100%);
          transform: translateX(-125%);
          transition: transform 520ms var(--ease);
          pointer-events: none;
        }

        .hero:hover,
        .panel:hover {
          transform: translateY(-3px);
          box-shadow: var(--shadow-strong);
          border-color: rgba(198, 90, 46, 0.22);
        }

        .hero:hover::before,
        .panel:hover::before,
        .news-card:hover::before,
        .highlight-card:hover::before {
          transform: translateX(125%);
        }

        .hero {
          padding: 18px 20px;
          margin-bottom: 16px;
        }

        .hero-grid {
          display: grid;
          grid-template-columns: 1.5fr 1fr;
          gap: 18px;
          align-items: center;
        }

        .hero-title {
          margin: 0;
          font-size: 2rem;
          line-height: 1.1;
        }

        .hero-meta {
          color: var(--muted);
          margin-top: 10px;
        }

        .signal-meter {
          border-radius: 22px;
          padding: 18px;
          background: linear-gradient(160deg, rgba(198, 90, 46, 0.14), rgba(31, 111, 95, 0.08) 82%);
          border: 1px solid rgba(198, 90, 46, 0.22);
          position: relative;
          overflow: hidden;
          transition: transform 220ms var(--ease), box-shadow 220ms var(--ease), border-color 220ms var(--ease);
          animation: pulse-glow 8.5s ease-in-out infinite;
        }

        .signal-meter:hover {
          transform: translateY(-2px);
          box-shadow: 0 22px 48px rgba(198, 90, 46, 0.14);
          border-color: rgba(198, 90, 46, 0.30);
        }

        .signal-meter .label {
          font-size: 0.78rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--muted);
          font-weight: 700;
        }

        .signal-meter .value {
          font-size: clamp(2rem, 4vw, 3rem);
          line-height: 1;
          margin-top: 8px;
          font-weight: 700;
        }

        .signal-meter .detail {
          margin-top: 10px;
          color: #39403d;
        }

        .grid {
          display: grid;
          gap: 16px;
        }

        .grid-4 {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .grid-3 {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .grid-2 {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .metric-card {
          padding: 16px;
          min-height: 126px;
          transition:
            transform 220ms var(--ease),
            box-shadow 220ms var(--ease),
            border-color 220ms var(--ease),
            background 220ms var(--ease);
        }

        .metric-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 24px 50px rgba(86, 96, 107, 0.12);
        }

        .metric-label {
          color: var(--muted);
          font-size: 0.84rem;
          margin-bottom: 10px;
        }

        .metric-value {
          font-size: 1.8rem;
          line-height: 1.05;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .metric-detail {
          font-size: 0.9rem;
          color: #4d5652;
        }

        .panel {
          padding: 18px 18px 14px;
        }

        .panel h2,
        .panel h3 {
          margin-top: 0;
          margin-bottom: 0.35rem;
        }

        .panel-subtitle {
          color: var(--muted);
          margin: 0 0 14px;
          font-size: 0.9rem;
        }

        .controls {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 14px;
        }

        .button-group {
          display: inline-flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .button {
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 8px 12px;
          background: white;
          color: #38413d;
          font: inherit;
          cursor: pointer;
          transition:
            transform 180ms var(--ease),
            box-shadow 180ms var(--ease),
            border-color 180ms var(--ease),
            background 180ms var(--ease);
        }

        .button:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 28px rgba(86, 96, 107, 0.10);
          border-color: rgba(198, 90, 46, 0.26);
        }

        .button.active {
          background: linear-gradient(135deg, rgba(198, 90, 46, 0.14), rgba(31, 111, 95, 0.10));
          border-color: rgba(198, 90, 46, 0.35);
        }

        select {
          border-radius: 14px;
          border: 1px solid var(--line);
          padding: 9px 12px;
          background: #fff;
          color: #37403c;
          font: inherit;
          transition:
            transform 180ms var(--ease),
            box-shadow 180ms var(--ease),
            border-color 180ms var(--ease);
        }

        select:hover,
        select:focus {
          transform: translateY(-1px);
          box-shadow: 0 12px 22px rgba(86, 96, 107, 0.08);
          border-color: rgba(198, 90, 46, 0.26);
          outline: none;
        }

        .chart {
          min-height: 340px;
        }

        .news-list,
        .highlight-list,
        .watch-list {
          display: grid;
          gap: 12px;
        }

        .news-card,
        .highlight-card {
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 14px 15px;
          background: rgba(255, 255, 255, 0.75);
          position: relative;
          overflow: hidden;
          transition:
            transform 220ms var(--ease),
            box-shadow 220ms var(--ease),
            border-color 220ms var(--ease),
            background 220ms var(--ease);
          animation: rise-in 0.8s var(--ease) both;
        }

        .news-card:hover,
        .highlight-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 20px 42px rgba(86, 96, 107, 0.11);
          border-color: rgba(198, 90, 46, 0.22);
        }

        .pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 8px;
        }

        .pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          border-radius: 999px;
          border: 1px solid var(--line);
          font-size: 0.76rem;
          font-weight: 700;
          background: rgba(255, 255, 255, 0.7);
        }

        .pill.up {
          color: var(--up);
          border-color: rgba(198, 90, 46, 0.25);
          background: rgba(198, 90, 46, 0.08);
        }

        .pill.down {
          color: var(--down);
          border-color: rgba(31, 111, 95, 0.25);
          background: rgba(31, 111, 95, 0.08);
        }

        .pill.neutral {
          color: var(--neutral);
          border-color: rgba(88, 96, 107, 0.25);
          background: rgba(88, 96, 107, 0.06);
        }

        .news-card h3,
        .highlight-card h3 {
          margin: 0 0 8px;
          font-size: 1.02rem;
          line-height: 1.4;
        }

        .small {
          color: var(--muted);
          font-size: 0.84rem;
        }

        .news-card h3 a,
        .highlight-card h3 a {
          transition: color 160ms var(--ease);
        }

        .news-card h3 a:hover,
        .highlight-card h3 a:hover {
          color: var(--up);
        }

        .table-wrap {
          overflow-x: auto;
          border-radius: 16px;
          border: 1px solid var(--line);
        }

        table {
          width: 100%;
          border-collapse: collapse;
          background: rgba(255, 255, 255, 0.76);
        }

        th,
        td {
          padding: 10px 12px;
          border-bottom: 1px solid rgba(216, 207, 192, 0.65);
          text-align: left;
          font-size: 0.92rem;
          vertical-align: top;
        }

        th {
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--muted);
          background: rgba(249, 245, 238, 0.95);
        }

        tbody tr:last-child td {
          border-bottom: none;
        }

        .footer {
          margin-top: 18px;
          color: var(--muted);
          font-size: 0.84rem;
          animation: rise-in 0.86s var(--ease) both;
        }

        .empty-state {
          color: var(--muted);
          font-size: 0.9rem;
          padding: 14px 0;
        }

        @keyframes rise-in {
          from {
            opacity: 0;
            transform: translateY(18px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes pulse-glow {
          0%, 100% {
            box-shadow: 0 16px 36px rgba(198, 90, 46, 0.06);
          }
          50% {
            box-shadow: 0 22px 48px rgba(31, 111, 95, 0.10);
          }
        }

        @media (max-width: 1080px) {
          .hero-grid,
          .grid-4,
          .grid-3,
          .grid-2 {
            grid-template-columns: 1fr;
          }

          .topbar {
            flex-direction: column;
          }

          .topnav {
            justify-content: flex-start;
          }

          .topbar-meta {
            justify-content: flex-start;
          }

          .lang-switcher {
            width: 100%;
            justify-content: space-between;
          }

          .lang-buttons {
            justify-content: flex-start;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          html {
            scroll-behavior: auto;
          }

          *,
          *::before,
          *::after {
            animation: none !important;
            transition: none !important;
          }
        }
        """
    )


def build_javascript() -> str:
    return dedent(
        """\
        const IMPACT_CLASS = {
          positive: "up",
          negative: "down",
          neutral: "neutral",
          "偏利多": "up",
          "偏利空": "down",
          "中性": "neutral",
        };

        const WINDOW_OPTIONS = {
          "3M": 63,
          "6M": 126,
          "1Y": 252,
          "2Y": 504,
        };

        const LOCALE_BY_LANG = {
          zh: "zh-CN",
          en: "en-US",
          de: "de-DE",
        };

        const I18N = {
          zh: {
            "site.title": "当升科技跟踪 Dashboard",
            "site.subtitle": "股价、新闻、模型与基本面，每个工作日自动刷新一次。",
            "site.footer": "数据源包含日线行情、公司公告、券商研报、外部媒体和季度财务摘要。页面为静态前端站点，非实时盘中系统。",
            "lang.label": "语言",
            "nav.home": "总览",
            "nav.news": "新闻",
            "nav.model": "模型",
            "nav.fundamentals": "基本面",
            "pageTitle.home": "当升科技 Dashboard | 总览",
            "pageTitle.news": "当升科技 Dashboard | 新闻",
            "pageTitle.model": "当升科技 Dashboard | 模型",
            "pageTitle.fundamentals": "当升科技 Dashboard | 基本面",
            "common.loading": "载入中",
            "common.na": "N/A",
            "common.latestClose": "最新收盘价",
            "common.latestCloseDetail": "{date} 收盘",
            "common.return20d": "20日涨跌",
            "common.return20dDetail": "近一个月交易窗口",
            "common.relativeReturn": "相对创业板",
            "common.relativeReturnDetail": "相对基准累计收益",
            "common.maxDrawdown": "最大回撤",
            "common.maxDrawdownDetail": "两年区间最大回撤",
            "common.modelAccuracy": "方向模型准确率",
            "common.modelAccuracyDetail": "基线 {baseline}",
            "common.newsSamples": "新闻样本量",
            "common.newsSamplesDetail": "{days} 个有效消息日",
            "common.signalLabel": "当日信号",
            "common.nextUpProbability": "下一交易日上涨概率",
            "common.predictedReturn": "预测收益 {value}",
            "common.nextSession": "下一交易日 {date}",
            "common.heroSubline": "{date} 的收盘信号，供跟踪参考，不构成投资建议。",
            "common.generatedAt": "最后生成时间：{generatedAt}",
            "common.dataStale": "数据可能已经过期。最新交易日是 {date}，距今天约 {diffDays} 天。",
            "common.dataOk": "数据状态正常。最新交易日 {date}，页面生成于 {generatedAt}。",
            "common.pageLoadFailed": "页面加载失败：{message}",
            "common.sentimentScore": "情绪分数 {value}",
            "common.publishedAt": "发布 {value}",
            "common.currentResults": "当前筛选结果 {count} 条",
            "common.noResults": "当前筛选条件下没有新闻。你可以把来源或方向切回“全部”。",
            "common.forInterpretation": "用于解释性参考",
            "common.notProvided": "未提供",
            "common.allMonths": "全部月份",
            "common.allSources": "全部来源",
            "common.allImpacts": "全部方向",
            "common.accuracy": "Accuracy",
            "common.rocAuc": "ROC AUC",
            "impact.positive": "偏利多",
            "impact.neutral": "中性",
            "impact.negative": "偏利空",
            "signal.strong": "偏积极",
            "signal.weak": "偏谨慎",
            "signal.neutral": "中性偏震荡",
            "source.disclosure": "公司披露",
            "source.research": "卖方研报",
            "source.news": "市场新闻",
            "source.web_news": "媒体报道",
            "home.hero.kicker": "当日信号",
            "home.price.title": "股价与成交量",
            "home.price.subtitle": "保留本地版的窗口切换，观察价格、量能和相对强弱。",
            "home.performance.title": "相对创业板表现",
            "home.performance.subtitle": "把股价和创业板指归一化后放在一起看，判断是否跑赢基准。",
            "home.highlights.title": "重点事件",
            "home.highlights.subtitle": "保留最强正向和负向线索，给家里人看时信息密度会更高。",
            "home.impact.title": "新闻方向分布",
            "home.impact.subtitle": "把近两年的消息按偏利多、中性、偏利空粗分。",
            "home.news.title": "最新消息流",
            "home.news.subtitle": "这里是更像本地版的新闻区块，支持月份、来源、方向筛选。",
            "home.positiveEmpty": "没有可展示的正向事件。",
            "home.negativeEmpty": "没有可展示的负向事件。",
            "news.hero.kicker": "消息流",
            "news.hero.title": "消息流与方向筛选",
            "news.hero.meta": "把公告、研报、外部媒体合在一页里看，便于快速判断当天市场叙事。",
            "news.hero.usageLabel": "使用方式",
            "news.hero.usageValue": "先看方向",
            "news.hero.usageDetail": "先筛月份，再看来源，最后挑偏利多或偏利空。这样比顺着时间线硬刷更有效。",
            "news.impact.title": "新闻方向分布",
            "news.impact.subtitle": "先看整体基调，再去下面筛消息流。",
            "news.feed.title": "全部消息",
            "news.feed.subtitle": "筛选器直接作用于下面这段消息流，按时间倒序展示，点击标题会跳原始链接。",
            "model.hero.kicker": "模型视图",
            "model.hero.title": "当日信号与模型解释",
            "model.hero.meta": "这里重点展示“今天的结论是什么、模型有没有明显优势、主要驱动是什么”。",
            "model.summary.title": "模型摘要",
            "model.summary.subtitle": "分类模型关注方向，回归模型关注下一日收益率。",
            "model.compare.title": "模型对比",
            "model.compare.subtitle": "看 accuracy 和 AUC 是否真正有优势。",
            "model.feature.title": "特征重要性",
            "model.feature.subtitle": "新闻和财务特征有没有实际进入排序靠前的位置。",
            "model.summary.classification": "分类模型",
            "model.summary.regression": "回归模型",
            "model.summary.equation": "回归方程",
            "fund.hero.kicker": "基本面",
            "fund.hero.title": "季度财务快照",
            "fund.hero.meta": "把公告生效后的财务指标拉进来，看收入、利润、毛利率和 ROE 是否同步改善。",
            "fund.hero.tipLabel": "提示",
            "fund.hero.tipValue": "适合中线跟踪",
            "fund.hero.tipDetail": "短线信号要看模型页，中线判断要把业绩兑现、债务和利润同比一起看。",
            "fund.profit.title": "营收与净利润",
            "fund.profit.subtitle": "最新 8 个季度的财务快照。",
            "fund.ratio.title": "毛利率、净利率与 ROE",
            "fund.ratio.subtitle": "利润质量比单看营收更关键。",
            "fund.table.title": "财务表",
            "fund.table.subtitle": "单位沿用抓取数据口径，适合作为快速对比视图。",
            "fund.table.reportDate": "报告期",
            "fund.table.revenue": "营收",
            "fund.table.netProfit": "净利润",
            "fund.table.grossMargin": "毛利率",
            "fund.table.netMargin": "净利率",
            "fund.table.roe": "ROE",
            "fund.table.debtRatio": "资产负债率",
            "fund.table.revenueYoy": "营收同比",
            "fund.table.profitYoy": "利润同比",
            "chart.stock": "当升科技",
            "chart.volume": "成交量",
            "chart.closeAxis": "收盘价",
            "chart.volumeAxis": "成交量",
            "chart.benchmark": "创业板指",
            "chart.normIndex": "归一化指数",
            "chart.revenue": "营收",
            "chart.netProfit": "净利润",
            "chart.grossMargin": "毛利率",
            "chart.netMargin": "净利率",
            "chart.roe": "ROE",
            "modelName.logistic": "Logistic Regression",
            "modelName.random_forest": "Random Forest",
            "modelName.rf_simple_baseline": "随机森林(简化基线)",
            "modelName.hist_gradient_boosting": "Hist Gradient Boosting",
            "modelName.ridge": "Ridge Regression",
            "modelName.random_forest_regressor": "Random Forest Regressor"
          },
          en: {
            "site.title": "Easpring Tracking Dashboard",
            "site.subtitle": "Price, news, models, and fundamentals refreshed every business day.",
            "site.footer": "Sources include daily market data, company disclosures, sell-side research, external media, and quarterly financial abstracts. This is a static frontend dashboard, not a live intraday system.",
            "lang.label": "Language",
            "nav.home": "Overview",
            "nav.news": "News",
            "nav.model": "Model",
            "nav.fundamentals": "Fundamentals",
            "pageTitle.home": "Easpring Dashboard | Overview",
            "pageTitle.news": "Easpring Dashboard | News",
            "pageTitle.model": "Easpring Dashboard | Model",
            "pageTitle.fundamentals": "Easpring Dashboard | Fundamentals",
            "common.loading": "Loading",
            "common.na": "N/A",
            "common.latestClose": "Latest close",
            "common.latestCloseDetail": "Close on {date}",
            "common.return20d": "20-day return",
            "common.return20dDetail": "Recent one-month trading window",
            "common.relativeReturn": "Relative to ChiNext",
            "common.relativeReturnDetail": "Cumulative return versus benchmark",
            "common.maxDrawdown": "Max drawdown",
            "common.maxDrawdownDetail": "Largest drawdown across the two-year window",
            "common.modelAccuracy": "Direction model accuracy",
            "common.modelAccuracyDetail": "Baseline {baseline}",
            "common.newsSamples": "News samples",
            "common.newsSamplesDetail": "{days} valid news days",
            "common.signalLabel": "Daily signal",
            "common.nextUpProbability": "Probability of an up move next session",
            "common.predictedReturn": "Predicted return {value}",
            "common.nextSession": "Next session {date}",
            "common.heroSubline": "Signal based on the {date} close. For monitoring only, not investment advice.",
            "common.generatedAt": "Last generated: {generatedAt}",
            "common.dataStale": "Data may be stale. The latest trading day is {date}, about {diffDays} days ago.",
            "common.dataOk": "Data is up to date enough. Latest trading day {date}; page generated at {generatedAt}.",
            "common.pageLoadFailed": "Page load failed: {message}",
            "common.sentimentScore": "Sentiment score {value}",
            "common.publishedAt": "Published {value}",
            "common.currentResults": "{count} items match the current filters",
            "common.noResults": "No news items match the current filters. Try switching source or impact back to all.",
            "common.forInterpretation": "For interpretability only",
            "common.notProvided": "Not provided",
            "common.allMonths": "All months",
            "common.allSources": "All sources",
            "common.allImpacts": "All impacts",
            "common.accuracy": "Accuracy",
            "common.rocAuc": "ROC AUC",
            "impact.positive": "Positive",
            "impact.neutral": "Neutral",
            "impact.negative": "Negative",
            "signal.strong": "Constructive",
            "signal.weak": "Cautious",
            "signal.neutral": "Neutral / range-bound",
            "source.disclosure": "Company filings",
            "source.research": "Sell-side research",
            "source.news": "Company news",
            "source.web_news": "Press coverage",
            "home.hero.kicker": "Daily signal",
            "home.price.title": "Price and volume",
            "home.price.subtitle": "Keep the local dashboard window switching and inspect price action, liquidity, and relative strength.",
            "home.performance.title": "Relative performance vs ChiNext",
            "home.performance.subtitle": "Normalize Easpring and the ChiNext index to see whether the stock is outperforming the benchmark.",
            "home.highlights.title": "Key events",
            "home.highlights.subtitle": "Keep the strongest positive and negative clues for a faster read.",
            "home.impact.title": "News impact mix",
            "home.impact.subtitle": "A rough split of the last two years into positive, neutral, and negative items.",
            "home.news.title": "Latest news flow",
            "home.news.subtitle": "This section is closer to the local dashboard and supports month, source, and impact filters.",
            "home.positiveEmpty": "No positive highlight is available.",
            "home.negativeEmpty": "No negative highlight is available.",
            "news.hero.kicker": "News stream",
            "news.hero.title": "News flow and impact filters",
            "news.hero.meta": "Combine disclosures, research, and external media in one view to understand the market narrative faster.",
            "news.hero.usageLabel": "How to use",
            "news.hero.usageValue": "Filter by direction first",
            "news.hero.usageDetail": "Start with month, then narrow by source, and finally pick positive or negative items.",
            "news.impact.title": "News impact mix",
            "news.impact.subtitle": "Check the overall tone first, then filter the feed below.",
            "news.feed.title": "All news",
            "news.feed.subtitle": "These filters act directly on the feed below. The list is shown in reverse chronological order and each title links to the source.",
            "model.hero.kicker": "Model view",
            "model.hero.title": "Daily signal and model interpretation",
            "model.hero.meta": "This page focuses on what the model says today, whether it has real edge, and what drives the signal.",
            "model.summary.title": "Model summary",
            "model.summary.subtitle": "The classifier predicts direction while the regressor predicts next-session return.",
            "model.compare.title": "Model comparison",
            "model.compare.subtitle": "Check whether accuracy and AUC show any real edge.",
            "model.feature.title": "Feature importance",
            "model.feature.subtitle": "See whether news and financial variables actually rank near the top.",
            "model.summary.classification": "Classification model",
            "model.summary.regression": "Regression model",
            "model.summary.equation": "Ridge equation",
            "fund.hero.kicker": "Fundamentals",
            "fund.hero.title": "Quarterly financial snapshot",
            "fund.hero.meta": "Bring in financial metrics from the date they became effective and see whether revenue, profit, margin, and ROE improve together.",
            "fund.hero.tipLabel": "Tip",
            "fund.hero.tipValue": "Useful for medium-term tracking",
            "fund.hero.tipDetail": "Use the model page for short-term signals; use fundamentals for medium-term judgment.",
            "fund.profit.title": "Revenue and net profit",
            "fund.profit.subtitle": "The latest eight quarterly snapshots.",
            "fund.ratio.title": "Gross margin, net margin, and ROE",
            "fund.ratio.subtitle": "Profit quality matters more than revenue alone.",
            "fund.table.title": "Financial table",
            "fund.table.subtitle": "Units follow the fetched data source and work as a quick comparison view.",
            "fund.table.reportDate": "Report date",
            "fund.table.revenue": "Revenue",
            "fund.table.netProfit": "Net profit",
            "fund.table.grossMargin": "Gross margin",
            "fund.table.netMargin": "Net margin",
            "fund.table.roe": "ROE",
            "fund.table.debtRatio": "Debt ratio",
            "fund.table.revenueYoy": "Revenue YoY",
            "fund.table.profitYoy": "Profit YoY",
            "chart.stock": "Easpring",
            "chart.volume": "Volume",
            "chart.closeAxis": "Close",
            "chart.volumeAxis": "Volume",
            "chart.benchmark": "ChiNext Index",
            "chart.normIndex": "Normalized index",
            "chart.revenue": "Revenue",
            "chart.netProfit": "Net profit",
            "chart.grossMargin": "Gross margin",
            "chart.netMargin": "Net margin",
            "chart.roe": "ROE",
            "modelName.logistic": "Logistic Regression",
            "modelName.random_forest": "Random Forest",
            "modelName.rf_simple_baseline": "Random Forest (simple baseline)",
            "modelName.hist_gradient_boosting": "Hist Gradient Boosting",
            "modelName.ridge": "Ridge Regression",
            "modelName.random_forest_regressor": "Random Forest Regressor"
          },
          de: {
            "site.title": "Easpring Dashboard",
            "site.subtitle": "Kurs, Nachrichten, Modelle und Fundamentaldaten werden an jedem Werktag aktualisiert.",
            "site.footer": "Die Quellen umfassen Tageskurse, Unternehmensmeldungen, Broker-Research, externe Medien und quartalsweise Finanzdaten. Dies ist ein statisches Frontend-Dashboard und kein Intraday-Livesystem.",
            "lang.label": "Sprache",
            "nav.home": "Übersicht",
            "nav.news": "Nachrichten",
            "nav.model": "Modell",
            "nav.fundamentals": "Fundamentaldaten",
            "pageTitle.home": "Easpring Dashboard | Übersicht",
            "pageTitle.news": "Easpring Dashboard | Nachrichten",
            "pageTitle.model": "Easpring Dashboard | Modell",
            "pageTitle.fundamentals": "Easpring Dashboard | Fundamentaldaten",
            "common.loading": "Lädt",
            "common.na": "k. A.",
            "common.latestClose": "Letzter Schlusskurs",
            "common.latestCloseDetail": "Schlusskurs am {date}",
            "common.return20d": "20-Tage-Rendite",
            "common.return20dDetail": "Letztes Ein-Monats-Handelsfenster",
            "common.relativeReturn": "Relativ zum ChiNext",
            "common.relativeReturnDetail": "Kumulierte Rendite gegenüber dem Benchmark",
            "common.maxDrawdown": "Maximaler Drawdown",
            "common.maxDrawdownDetail": "Größter Rückgang im Zwei-Jahres-Fenster",
            "common.modelAccuracy": "Trefferquote des Richtungsmodells",
            "common.modelAccuracyDetail": "Baseline {baseline}",
            "common.newsSamples": "Nachrichtenstichprobe",
            "common.newsSamplesDetail": "{days} gültige Nachrichtentage",
            "common.signalLabel": "Tagessignal",
            "common.nextUpProbability": "Wahrscheinlichkeit eines Anstiegs in der nächsten Sitzung",
            "common.predictedReturn": "Prognostizierte Rendite {value}",
            "common.nextSession": "Nächste Sitzung {date}",
            "common.heroSubline": "Signal auf Basis des Schlusskurses vom {date}. Nur zur Beobachtung, keine Anlageempfehlung.",
            "common.generatedAt": "Zuletzt erstellt: {generatedAt}",
            "common.dataStale": "Die Daten könnten veraltet sein. Der letzte Handelstag ist {date}, also etwa {diffDays} Tage her.",
            "common.dataOk": "Datenstand ist in Ordnung. Letzter Handelstag {date}; Seite erzeugt um {generatedAt}.",
            "common.pageLoadFailed": "Seite konnte nicht geladen werden: {message}",
            "common.sentimentScore": "Sentiment-Score {value}",
            "common.publishedAt": "Veröffentlicht {value}",
            "common.currentResults": "{count} Einträge passen zu den aktuellen Filtern",
            "common.noResults": "Keine Nachrichten passen zu den aktuellen Filtern. Stelle Quelle oder Richtung wieder auf alle zurück.",
            "common.forInterpretation": "Nur zur Interpretation",
            "common.notProvided": "Nicht verfügbar",
            "common.allMonths": "Alle Monate",
            "common.allSources": "Alle Quellen",
            "common.allImpacts": "Alle Richtungen",
            "common.accuracy": "Accuracy",
            "common.rocAuc": "ROC AUC",
            "impact.positive": "Positiv",
            "impact.neutral": "Neutral",
            "impact.negative": "Negativ",
            "signal.strong": "Konstruktiv",
            "signal.weak": "Vorsichtig",
            "signal.neutral": "Neutral / Seitwärtsphase",
            "source.disclosure": "Unternehmensmeldungen",
            "source.research": "Analysten-Research",
            "source.news": "Unternehmensnachrichten",
            "source.web_news": "Presseberichte",
            "home.hero.kicker": "Tagessignal",
            "home.price.title": "Kurs und Volumen",
            "home.price.subtitle": "Behalte die Zeitfenster-Umschaltung des lokalen Dashboards bei und beobachte Kursverlauf, Liquidität und relative Stärke.",
            "home.performance.title": "Relative Entwicklung vs. ChiNext",
            "home.performance.subtitle": "Normalisiere Easpring und den ChiNext-Index, um die Outperformance besser zu sehen.",
            "home.highlights.title": "Wichtige Ereignisse",
            "home.highlights.subtitle": "Die stärksten positiven und negativen Hinweise für einen schnelleren Überblick.",
            "home.impact.title": "Verteilung der Nachrichtenwirkung",
            "home.impact.subtitle": "Grobe Aufteilung der letzten zwei Jahre in positive, neutrale und negative Meldungen.",
            "home.news.title": "Aktueller Nachrichtenstrom",
            "home.news.subtitle": "Dieser Bereich liegt näher am lokalen Dashboard und unterstützt Filter nach Monat, Quelle und Wirkung.",
            "home.positiveEmpty": "Kein positives Highlight verfügbar.",
            "home.negativeEmpty": "Kein negatives Highlight verfügbar.",
            "news.hero.kicker": "Nachrichtenstrom",
            "news.hero.title": "Nachrichtenfluss und Wirkungsfilter",
            "news.hero.meta": "Fasse Meldungen, Research und externe Medien in einer Ansicht zusammen, um die Markterzählung schneller zu erfassen.",
            "news.hero.usageLabel": "Anwendung",
            "news.hero.usageValue": "Zuerst nach Richtung filtern",
            "news.hero.usageDetail": "Beginne mit dem Monat, verfeinere dann nach Quelle und wähle am Ende positive oder negative Meldungen.",
            "news.impact.title": "Verteilung der Nachrichtenwirkung",
            "news.impact.subtitle": "Zuerst den Gesamttenor prüfen und danach den Feed unten filtern.",
            "news.feed.title": "Alle Nachrichten",
            "news.feed.subtitle": "Diese Filter wirken direkt auf den Nachrichtenfeed unten. Die Liste ist absteigend nach Zeit sortiert; jeder Titel verlinkt zur Quelle.",
            "model.hero.kicker": "Modellansicht",
            "model.hero.title": "Tagessignal und Modellinterpretation",
            "model.hero.meta": "Hier geht es darum, was das Modell heute sagt, ob es echten Mehrwert hat und was das Signal treibt.",
            "model.summary.title": "Modellübersicht",
            "model.summary.subtitle": "Das Klassifikationsmodell schätzt die Richtung, das Regressionsmodell die Rendite der nächsten Sitzung.",
            "model.compare.title": "Modellvergleich",
            "model.compare.subtitle": "Prüfe, ob Accuracy und AUC wirklich einen Vorteil zeigen.",
            "model.feature.title": "Merkmalswichtigkeit",
            "model.feature.subtitle": "Zeigt, ob Nachrichten- und Finanzvariablen tatsächlich weit oben ranken.",
            "model.summary.classification": "Klassifikationsmodell",
            "model.summary.regression": "Regressionsmodell",
            "model.summary.equation": "Ridge-Gleichung",
            "fund.hero.kicker": "Fundamentaldaten",
            "fund.hero.title": "Quartalsweiser Finanzüberblick",
            "fund.hero.meta": "Finanzkennzahlen ab ihrem Wirksamkeitsdatum einbeziehen und prüfen, ob Umsatz, Gewinn, Marge und ROE gemeinsam besser werden.",
            "fund.hero.tipLabel": "Hinweis",
            "fund.hero.tipValue": "Geeignet für mittelfristiges Tracking",
            "fund.hero.tipDetail": "Für kurzfristige Signale nutze die Modellseite; für mittelfristige Einschätzungen die Fundamentaldaten.",
            "fund.profit.title": "Umsatz und Nettogewinn",
            "fund.profit.subtitle": "Die letzten acht Quartals-Snapshots.",
            "fund.ratio.title": "Bruttomarge, Nettomarge und ROE",
            "fund.ratio.subtitle": "Die Gewinnqualität ist wichtiger als Umsatz allein.",
            "fund.table.title": "Finanztabelle",
            "fund.table.subtitle": "Die Einheiten folgen der Datenquelle und dienen als schnelle Vergleichsansicht.",
            "fund.table.reportDate": "Berichtsdatum",
            "fund.table.revenue": "Umsatz",
            "fund.table.netProfit": "Nettogewinn",
            "fund.table.grossMargin": "Bruttomarge",
            "fund.table.netMargin": "Nettomarge",
            "fund.table.roe": "ROE",
            "fund.table.debtRatio": "Verschuldungsgrad",
            "fund.table.revenueYoy": "Umsatz YoY",
            "fund.table.profitYoy": "Gewinn YoY",
            "chart.stock": "Easpring",
            "chart.volume": "Volumen",
            "chart.closeAxis": "Schlusskurs",
            "chart.volumeAxis": "Volumen",
            "chart.benchmark": "ChiNext-Index",
            "chart.normIndex": "Normalisierter Index",
            "chart.revenue": "Umsatz",
            "chart.netProfit": "Nettogewinn",
            "chart.grossMargin": "Bruttomarge",
            "chart.netMargin": "Nettomarge",
            "chart.roe": "ROE",
            "modelName.logistic": "Logistische Regression",
            "modelName.random_forest": "Random Forest",
            "modelName.rf_simple_baseline": "Random Forest (einfache Basislinie)",
            "modelName.hist_gradient_boosting": "Histogram Gradient Boosting",
            "modelName.ridge": "Ridge-Regression",
            "modelName.random_forest_regressor": "Random-Forest-Regressor"
          }
        };

        const IMPACT_KEY_BY_LABEL = {
          "偏利多": "positive",
          "中性": "neutral",
          "偏利空": "negative",
        };

        const SIGNAL_KEY_BY_LABEL = {
          "偏强": "strong",
          "偏弱": "weak",
          "震荡偏中性": "neutral",
        };

        let currentLang = "zh";

        function locale() {
          return LOCALE_BY_LANG[currentLang] || LOCALE_BY_LANG.zh;
        }

        function t(key, vars = {}) {
          const dict = I18N[currentLang] || I18N.zh;
          const base = dict[key] ?? I18N.zh[key] ?? key;
          return String(base).replace(/\\{(\\w+)\\}/g, (_, token) => {
            const value = vars[token];
            return value === undefined || value === null ? "" : String(value);
          });
        }

        function detectLanguage() {
          const params = new URLSearchParams(window.location.search);
          const paramLang = params.get("lang");
          if (paramLang && I18N[paramLang]) {
            localStorage.setItem("dashboard_lang", paramLang);
            return paramLang;
          }
          const stored = localStorage.getItem("dashboard_lang");
          if (stored && I18N[stored]) {
            return stored;
          }
          return "zh";
        }

        function applyStaticTranslations() {
          document.documentElement.lang = currentLang === "zh" ? "zh-CN" : currentLang;
          document.querySelectorAll("[data-i18n]").forEach((node) => {
            node.textContent = t(node.dataset.i18n);
          });
          document.title = t(`pageTitle.${document.body.dataset.page}`);
        }

        function initLanguageSelector() {
          const buttons = Array.from(document.querySelectorAll(".lang-button[data-lang]"));
          if (!buttons.length) return;
          buttons.forEach((button) => {
            const isActive = button.dataset.lang === currentLang;
            button.classList.toggle("active", isActive);
            button.setAttribute("aria-pressed", String(isActive));
            button.addEventListener("click", () => {
              const next = button.dataset.lang;
              if (!next || !I18N[next]) return;
              localStorage.setItem("dashboard_lang", next);
              window.location.reload();
            });
          });
        }

        async function loadDashboardData() {
          const version = encodeURIComponent(window.__DASHBOARD_VERSION__ || "");
          const response = await fetch(`data/dashboard.json?v=${version}`, { cache: "no-store" });
          if (!response.ok) {
            throw new Error("Failed to load dashboard data");
          }
          return response.json();
        }

        function formatNumber(value, digits = 2) {
          if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return t("common.na");
          }
          return new Intl.NumberFormat(locale(), {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
          }).format(Number(value));
        }

        function formatPercent(value, digits = 2) {
          if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return t("common.na");
          }
          return `${formatNumber(Number(value) * 100, digits)}%`;
        }

        function toneClass(value, reverse = false) {
          if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "neutral";
          }
          const num = Number(value);
          if (!reverse && num > 0) return "up";
          if (!reverse && num < 0) return "down";
          if (reverse && num < 0) return "up";
          if (reverse && num > 0) return "down";
          return "neutral";
        }

        function escapeHtml(value) {
          return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
        }

        function sourceLabel(value) {
          const key = typeof value === "string" ? value : value?.source_type;
          return t(`source.${key || "web_news"}`);
        }

        function impactKey(value) {
          if (typeof value === "string") {
            return IMPACT_KEY_BY_LABEL[value] || value;
          }
          return value?.impact_key || IMPACT_KEY_BY_LABEL[value?.impact] || "neutral";
        }

        function impactLabel(value) {
          return t(`impact.${impactKey(value)}`);
        }

        function signalKey(signal) {
          if (!signal) return "neutral";
          return signal.label_key || SIGNAL_KEY_BY_LABEL[signal.label] || "neutral";
        }

        function signalLabel(signal) {
          return t(`signal.${signalKey(signal)}`);
        }

        function modelLabel(name) {
          if (!name) return t("common.na");
          return t(`modelName.${name}`);
        }

        function tracePointCount(trace) {
          if (Array.isArray(trace?.x) && trace.x.length) return trace.x.length;
          if (Array.isArray(trace?.y) && trace.y.length) return trace.y.length;
          return 0;
        }

        function setBarHoverState(chart, hoveredPoint = null) {
          (chart.data || []).forEach((trace, traceIndex) => {
            if (trace?.type !== "bar") return;
            const count = tracePointCount(trace);
            if (!count) return;
            const isHoveredTrace = hoveredPoint && hoveredPoint.curveNumber === traceIndex;
            const opacity = Array.from({ length: count }, (_, pointIndex) => {
              if (!hoveredPoint) return 0.88;
              if (isHoveredTrace && pointIndex === hoveredPoint.pointNumber) return 1;
              if (isHoveredTrace) return 0.58;
              return 0.42;
            });
            const lineWidth = Array.from({ length: count }, (_, pointIndex) => {
              if (!hoveredPoint) return 0;
              if (isHoveredTrace && pointIndex === hoveredPoint.pointNumber) return 2.4;
              return 0.8;
            });
            const lineColor = Array.from({ length: count }, (_, pointIndex) => {
              if (!hoveredPoint) return "rgba(255,255,255,0)";
              if (isHoveredTrace && pointIndex === hoveredPoint.pointNumber) return "rgba(22, 26, 24, 0.42)";
              return "rgba(22, 26, 24, 0.12)";
            });
            Plotly.restyle(
              chart,
              {
                "marker.opacity": [opacity],
                "marker.line.width": [lineWidth],
                "marker.line.color": [lineColor],
              },
              [traceIndex]
            );
          });
        }

        function attachBarHoverEffects(chart) {
          if (!chart) return;
          if (typeof chart.removeAllListeners === "function") {
            chart.removeAllListeners("plotly_hover");
            chart.removeAllListeners("plotly_unhover");
          }
          setBarHoverState(chart, null);
          chart.on("plotly_hover", (eventData) => {
            const hoveredBar = eventData?.points?.find((point) => point?.data?.type === "bar");
            if (!hoveredBar) return;
            setBarHoverState(chart, hoveredBar);
          });
          chart.on("plotly_unhover", () => {
            setBarHoverState(chart, null);
          });
        }

        function formatEquationSummary(equation) {
          if (!equation || typeof equation !== "object") {
            return t("common.notProvided");
          }
          const intercept = formatNumber(equation.intercept ?? 0, 3);
          const coeffs = Object.entries(equation.coefficients || {})
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 3)
            .map(([feature, value]) => `${Number(value) >= 0 ? "+" : ""}${formatNumber(value, 3)}·${feature}`);
          return [`r = ${intercept}`, ...coeffs].join(" ");
        }

        function renderMetricCard(targetId, label, value, detail, tone = "neutral") {
          const target = document.getElementById(targetId);
          if (!target) return;
          target.className = `panel metric-card ${tone}`;
          target.innerHTML = `
            <div class="metric-label">${escapeHtml(label)}</div>
            <div class="metric-value">${escapeHtml(value)}</div>
            <div class="metric-detail">${escapeHtml(detail)}</div>
          `;
        }

        function populateSummary(data) {
          const summary = data.summary;
          const signal = data.signal;
          const signalProbability = signal.up_probability ?? summary.classification_accuracy;
          renderMetricCard(
            "metric-close",
            t("common.latestClose"),
            signal.latest_close !== null && signal.latest_close !== undefined ? `${formatNumber(signal.latest_close, 2)}` : t("common.na"),
            t("common.latestCloseDetail", { date: signal.signal_date || summary.latest_date }),
            "neutral"
          );
          renderMetricCard(
            "metric-20d",
            t("common.return20d"),
            formatPercent(summary.change_20d),
            t("common.return20dDetail"),
            toneClass(summary.change_20d)
          );
          renderMetricCard(
            "metric-relative",
            t("common.relativeReturn"),
            formatPercent(summary.relative_return),
            t("common.relativeReturnDetail"),
            toneClass(summary.relative_return)
          );
          renderMetricCard(
            "metric-drawdown",
            t("common.maxDrawdown"),
            formatPercent(summary.max_drawdown),
            t("common.maxDrawdownDetail"),
            toneClass(summary.max_drawdown, true)
          );
          renderMetricCard(
            "metric-model",
            t("common.modelAccuracy"),
            formatPercent(summary.classification_accuracy),
            t("common.modelAccuracyDetail", { baseline: formatPercent(summary.classification_baseline) }),
            toneClass((summary.classification_accuracy ?? 0) - (summary.classification_baseline ?? 0))
          );
          renderMetricCard(
            "metric-news",
            t("common.newsSamples"),
            `${summary.news_total ?? 0}`,
            t("common.newsSamplesDetail", { days: summary.news_days ?? 0 }),
            "neutral"
          );

          const signalPanel = document.getElementById("signal-panel");
          if (signalPanel) {
            const tone = toneClass(signal.predicted_return ?? 0);
            signalPanel.innerHTML = `
              <div class="label">${escapeHtml(t("common.signalLabel"))}</div>
              <div class="value">${formatPercent(signalProbability, 1)}</div>
              <div class="detail">${escapeHtml(t("common.nextUpProbability"))}</div>
              <div class="pill-row" style="margin-top: 12px;">
                <span class="pill ${tone}">${escapeHtml(signalLabel(signal))}</span>
                <span class="pill neutral">${escapeHtml(t("common.predictedReturn", { value: formatPercent(signal.predicted_return, 2) }))}</span>
                <span class="pill neutral">${escapeHtml(t("common.nextSession", { date: signal.next_session_date || t("common.na") }))}</span>
              </div>
            `;
          }

          const headline = document.getElementById("hero-headline");
          const subline = document.getElementById("hero-subline");
          if (headline) {
            headline.textContent = signalLabel(signal);
          }
          if (subline) {
            subline.textContent = t("common.heroSubline", { date: signal.signal_date || summary.latest_date });
          }
        }

        function renderStatusBanner(data) {
          const banner = document.getElementById("status-banner");
          if (!banner) return;
          const latestDate = new Date(`${data.summary.latest_date}T00:00:00`);
          const now = new Date();
          const diffDays = Math.floor((now - latestDate) / (1000 * 60 * 60 * 24));
          const label = document.getElementById("generated-at");
          if (label) {
            label.textContent = t("common.generatedAt", { generatedAt: data.generated_at_label });
          }
          if (diffDays > 2) {
            banner.className = "banner warning show";
            banner.innerHTML = escapeHtml(t("common.dataStale", { date: data.summary.latest_date, diffDays }));
          } else {
            banner.className = "banner show";
            banner.innerHTML = escapeHtml(t("common.dataOk", { date: data.summary.latest_date, generatedAt: data.generated_at_label }));
          }
        }

        function buildPriceChart(data, windowKey = "1Y") {
          const chart = document.getElementById("price-chart");
          if (!chart) return;
          const count = WINDOW_OPTIONS[windowKey] || WINDOW_OPTIONS["1Y"];
          const rows = data.prices.slice(-count);
          Plotly.newPlot(
            chart,
            [
              {
                x: rows.map((row) => row.date),
                y: rows.map((row) => row.close),
                type: "scatter",
                mode: "lines",
                name: t("chart.stock"),
                line: { color: "#c65a2e", width: 3 },
                yaxis: "y",
              },
              {
                x: rows.map((row) => row.date),
                y: rows.map((row) => row.volume),
                type: "bar",
                name: t("chart.volume"),
                marker: { color: "rgba(31, 111, 95, 0.22)" },
                yaxis: "y2",
              },
            ],
            {
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 42, r: 42, t: 16, b: 40 },
              hovermode: "x unified",
              hoverlabel: {
                bgcolor: "rgba(251, 248, 243, 0.96)",
                bordercolor: "rgba(198, 90, 46, 0.18)",
                font: { color: "#161a18" },
              },
              xaxis: { showgrid: false },
              yaxis: { title: t("chart.closeAxis"), gridcolor: "rgba(216, 207, 192, 0.35)" },
              yaxis2: { title: t("chart.volumeAxis"), overlaying: "y", side: "right", showgrid: false },
              legend: { orientation: "h", y: 1.1 },
            },
            { responsive: true, displayModeBar: false }
          ).then(() => attachBarHoverEffects(chart));

          const performanceChart = document.getElementById("performance-chart");
          if (performanceChart) {
            Plotly.newPlot(
              performanceChart,
              [
                {
                  x: rows.map((row) => row.date),
                  y: rows.map((row) => row.stock_index),
                  type: "scatter",
                  mode: "lines",
                  name: t("chart.stock"),
                  line: { color: "#c65a2e", width: 3 },
                },
                {
                  x: rows.map((row) => row.date),
                  y: rows.map((row) => row.benchmark_index),
                  type: "scatter",
                  mode: "lines",
                  name: t("chart.benchmark"),
                  line: { color: "#1f6f5f", width: 2.6 },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 40 },
                hovermode: "x unified",
                hoverlabel: {
                  bgcolor: "rgba(251, 248, 243, 0.96)",
                  bordercolor: "rgba(31, 111, 95, 0.18)",
                  font: { color: "#161a18" },
                },
                xaxis: { showgrid: false },
                yaxis: { title: t("chart.normIndex"), gridcolor: "rgba(216, 207, 192, 0.35)" },
                legend: { orientation: "h", y: 1.12 },
              },
              { responsive: true, displayModeBar: false }
            );
          }
        }

        function setupWindowButtons(data) {
          const buttons = Array.from(document.querySelectorAll("[data-window]"));
          if (!buttons.length) return;
          let current = "1Y";
          const rerender = () => {
            buttons.forEach((button) => {
              button.classList.toggle("active", button.dataset.window === current);
            });
            buildPriceChart(data, current);
          };
          buttons.forEach((button) => {
            button.addEventListener("click", () => {
              current = button.dataset.window;
              rerender();
            });
          });
          rerender();
        }

        function renderHighlights(data) {
          const positive = document.getElementById("positive-highlight");
          const negative = document.getElementById("negative-highlight");
          const renderItem = (container, item, fallbackKey) => {
            if (!container) return;
            if (!item) {
              container.innerHTML = `<div class="empty-state">${escapeHtml(t(fallbackKey))}</div>`;
              return;
            }
            const tone = IMPACT_CLASS[impactKey(item)] || "neutral";
            container.innerHTML = `
              <div class="highlight-card">
                <div class="pill-row">
                  <span class="pill ${tone}">${escapeHtml(impactLabel(item))}</span>
                  <span class="pill neutral">${escapeHtml(sourceLabel(item))}</span>
                  <span class="pill neutral">${escapeHtml(item.timestamp_label)}</span>
                </div>
                <h3><a href="${escapeHtml(item.source_link)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a></h3>
                <p class="small">${escapeHtml(t("common.sentimentScore", { value: formatNumber(item.sentiment_score, 3) }))}</p>
              </div>
            `;
          };
          renderItem(positive, data.highlights.positive_event?.[0], "home.positiveEmpty");
          renderItem(negative, data.highlights.negative_event?.[0], "home.negativeEmpty");
        }

        function buildImpactBars(data) {
          const chart = document.getElementById("impact-chart");
          if (!chart) return;
          Plotly.newPlot(
            chart,
            [
              {
                x: [t("impact.positive"), t("impact.neutral"), t("impact.negative")],
                y: [
                  data.summary.positive_news_count || 0,
                  data.summary.neutral_news_count || 0,
                  data.summary.negative_news_count || 0,
                ],
                type: "bar",
                marker: {
                  color: ["#c65a2e", "#58606b", "#1f6f5f"],
                },
              },
            ],
            {
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 42, r: 18, t: 12, b: 40 },
              hoverlabel: {
                bgcolor: "rgba(251, 248, 243, 0.96)",
                bordercolor: "rgba(198, 90, 46, 0.18)",
                font: { color: "#161a18" },
              },
              xaxis: { showgrid: false },
              yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
            },
            { responsive: true, displayModeBar: false }
          ).then(() => attachBarHoverEffects(chart));
        }

        function renderNewsFeed(data, options = {}) {
          const target = document.getElementById(options.targetId || "news-feed");
          if (!target) return;

          const monthSelect = document.getElementById(options.monthId || "news-month");
          const sourceSelect = document.getElementById(options.sourceId || "news-source");
          const impactSelect = document.getElementById(options.impactId || "news-impact");
          const countTarget = document.getElementById(options.countId || "");

          const allNews = data.news;
          const allMonths = [...new Set(allNews.map((item) => item.month).filter(Boolean))];

          function setOptions(selectNode, defaultLabel, values, currentValue, labelFn = (value) => value) {
            if (!selectNode) return "all";
            selectNode.innerHTML = `<option value="all">${escapeHtml(defaultLabel)}</option>${values
              .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(labelFn(value))}</option>`)
              .join("")}`;
            const nextValue = values.includes(currentValue) ? currentValue : "all";
            selectNode.value = nextValue;
            return nextValue;
          }

          if (monthSelect && !monthSelect.dataset.ready) {
            monthSelect.innerHTML = `<option value="all">${escapeHtml(t("common.allMonths"))}</option>${allMonths
              .map((month) => `<option value="${escapeHtml(month)}">${escapeHtml(month)}</option>`)
              .join("")}`;
            monthSelect.dataset.ready = "true";
          }

          const rerender = () => {
            const month = monthSelect ? monthSelect.value : "all";
            const monthScopedRows = allNews.filter((item) => month === "all" || item.month === month);

            const sourceOptions = [...new Set(monthScopedRows.map((item) => item.source_type).filter(Boolean))];
            const source = setOptions(
              sourceSelect,
              t("common.allSources"),
              sourceOptions,
              sourceSelect ? sourceSelect.value : "all",
              (value) => sourceLabel(value)
            );

            const sourceScopedRows = monthScopedRows.filter((item) => source === "all" || item.source_type === source);
            const impactOptions = [...new Set(sourceScopedRows.map((item) => impactKey(item)).filter(Boolean))];
            const impact = setOptions(
              impactSelect,
              t("common.allImpacts"),
              impactOptions,
              impactSelect ? impactSelect.value : "all",
              (value) => impactLabel(value)
            );

            const limit = options.limit || allNews.length;
            const rows = monthScopedRows
              .filter((item) => source === "all" || item.source_type === source)
              .filter((item) => impact === "all" || impactKey(item) === impact)
              .slice(0, limit);

            if (countTarget) {
              countTarget.textContent = t("common.currentResults", { count: rows.length });
            }

            if (!rows.length) {
              target.innerHTML = `<div class="empty-state">${escapeHtml(t("common.noResults"))}</div>`;
              return;
            }

            target.innerHTML = rows
              .map((item) => {
                const tone = IMPACT_CLASS[impactKey(item)] || "neutral";
                const body = item.body_excerpt ? `<p class="small">${escapeHtml(item.body_excerpt)}</p>` : "";
                return `
                  <article class="news-card">
                    <div class="pill-row">
                      <span class="pill ${tone}">${escapeHtml(impactLabel(item))}</span>
                      <span class="pill neutral">${escapeHtml(sourceLabel(item))}</span>
                      <span class="pill neutral">${escapeHtml(item.effective_date)}</span>
                    </div>
                    <h3><a href="${escapeHtml(item.source_link)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a></h3>
                    ${body}
                    <p class="small">${escapeHtml(t("common.sentimentScore", { value: formatNumber(item.sentiment_score, 3) }))} | ${escapeHtml(t("common.publishedAt", { value: item.timestamp }))}</p>
                  </article>
                `;
              })
              .join("");
          };

          [monthSelect, sourceSelect, impactSelect]
            .filter(Boolean)
            .forEach((node) => node.addEventListener("change", rerender));
          rerender();
        }

        function buildModelCharts(data) {
          const compareChart = document.getElementById("model-compare-chart");
          if (compareChart) {
            const candidates = data.models.classification.candidates || [];
            Plotly.newPlot(
              compareChart,
              [
                {
                  x: candidates.map((item) => modelLabel(item.model_name || item.model || item.name || "model")),
                  y: candidates.map((item) => item.accuracy ?? 0),
                  type: "bar",
                  name: t("common.accuracy"),
                  marker: { color: "#c65a2e" },
                },
                {
                  x: candidates.map((item) => modelLabel(item.model_name || item.model || item.name || "model")),
                  y: candidates.map((item) => item.roc_auc ?? 0),
                  type: "bar",
                  name: t("common.rocAuc"),
                  marker: { color: "#1f6f5f" },
                },
              ],
              {
                barmode: "group",
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 60 },
                hoverlabel: {
                  bgcolor: "rgba(251, 248, 243, 0.96)",
                  bordercolor: "rgba(198, 90, 46, 0.18)",
                  font: { color: "#161a18" },
                },
                xaxis: { tickangle: -18, showgrid: false },
                yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                legend: { orientation: "h", y: 1.15 },
              },
              { responsive: true, displayModeBar: false }
            ).then(() => attachBarHoverEffects(compareChart));
          }

          const featureChart = document.getElementById("feature-chart");
          if (featureChart) {
            const features = data.models.classification.feature_importance || [];
            Plotly.newPlot(
              featureChart,
              [
                {
                  x: features.map((item) => item.importance ?? item.weight ?? 0),
                  y: features.map((item) => item.feature || item.name || ""),
                  type: "bar",
                  orientation: "h",
                  marker: { color: "#58606b" },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 150, r: 18, t: 16, b: 40 },
                hoverlabel: {
                  bgcolor: "rgba(251, 248, 243, 0.96)",
                  bordercolor: "rgba(88, 96, 107, 0.18)",
                  font: { color: "#161a18" },
                },
                xaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                yaxis: { automargin: true },
              },
              { responsive: true, displayModeBar: false }
            ).then(() => attachBarHoverEffects(featureChart));
          }

          const summary = document.getElementById("model-summary");
          if (summary) {
            const cls = data.models.classification;
            const reg = data.models.regression;
            summary.innerHTML = `
              <div class="grid grid-3">
                <div class="panel metric-card ${toneClass((cls.selected_metrics.accuracy ?? 0) - (cls.baseline_accuracy ?? 0))}">
                  <div class="metric-label">${escapeHtml(t("model.summary.classification"))}</div>
                  <div class="metric-value">${escapeHtml(modelLabel(cls.selected_model))}</div>
                  <div class="metric-detail">${escapeHtml(t("common.accuracy"))} ${formatPercent(cls.selected_metrics.accuracy)} | ${escapeHtml(t("common.rocAuc"))} ${formatPercent(cls.selected_metrics.roc_auc)}</div>
                </div>
                <div class="panel metric-card ${toneClass(reg.selected_metrics.r2 ?? 0)}">
                  <div class="metric-label">${escapeHtml(t("model.summary.regression"))}</div>
                  <div class="metric-value">${escapeHtml(modelLabel(reg.selected_model))}</div>
                  <div class="metric-detail">MAE ${formatNumber(reg.selected_metrics.mae, 4)} | R² ${formatNumber(reg.selected_metrics.r2, 4)}</div>
                </div>
                <div class="panel metric-card neutral">
                  <div class="metric-label">${escapeHtml(t("model.summary.equation"))}</div>
                  <div class="metric-value" style="font-size:1rem; line-height:1.45;">${escapeHtml(formatEquationSummary(reg.ridge_equation))}</div>
                  <div class="metric-detail">${escapeHtml(t("common.forInterpretation"))}</div>
                </div>
              </div>
            `;
          }
        }

        function buildFundamentalCharts(data) {
          const rows = data.quarterly || [];
          const profitChart = document.getElementById("fundamentals-profit-chart");
          if (profitChart) {
            Plotly.newPlot(
              profitChart,
              [
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.revenue),
                  type: "bar",
                  name: t("chart.revenue"),
                  marker: { color: "rgba(198, 90, 46, 0.75)" },
                },
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.net_profit),
                  type: "scatter",
                  mode: "lines+markers",
                  name: t("chart.netProfit"),
                  line: { color: "#1f6f5f", width: 3 },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 40 },
                hovermode: "x unified",
                hoverlabel: {
                  bgcolor: "rgba(251, 248, 243, 0.96)",
                  bordercolor: "rgba(198, 90, 46, 0.18)",
                  font: { color: "#161a18" },
                },
                xaxis: { showgrid: false },
                yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                legend: { orientation: "h", y: 1.14 },
              },
              { responsive: true, displayModeBar: false }
            ).then(() => attachBarHoverEffects(profitChart));
          }

          const ratioChart = document.getElementById("fundamentals-ratio-chart");
          if (ratioChart) {
            Plotly.newPlot(
              ratioChart,
              [
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.gross_margin),
                  type: "scatter",
                  mode: "lines+markers",
                  name: t("chart.grossMargin"),
                  line: { color: "#c65a2e", width: 3 },
                },
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.net_margin),
                  type: "scatter",
                  mode: "lines+markers",
                  name: t("chart.netMargin"),
                  line: { color: "#58606b", width: 3 },
                },
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.roe),
                  type: "scatter",
                  mode: "lines+markers",
                  name: t("chart.roe"),
                  line: { color: "#1f6f5f", width: 3 },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 40 },
                hovermode: "x unified",
                hoverlabel: {
                  bgcolor: "rgba(251, 248, 243, 0.96)",
                  bordercolor: "rgba(31, 111, 95, 0.18)",
                  font: { color: "#161a18" },
                },
                xaxis: { showgrid: false },
                yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                legend: { orientation: "h", y: 1.15 },
              },
              { responsive: true, displayModeBar: false }
            );
          }

          const table = document.getElementById("fundamentals-table");
          if (table) {
            const body = rows
              .map(
                (row) => `
                  <tr>
                    <td>${escapeHtml(row.report_date)}</td>
                    <td>${formatNumber(row.revenue, 3)}</td>
                    <td>${formatNumber(row.net_profit, 3)}</td>
                    <td>${formatNumber(row.gross_margin, 2)}</td>
                    <td>${formatNumber(row.net_margin, 2)}</td>
                    <td>${formatNumber(row.roe, 2)}</td>
                    <td>${formatNumber(row.debt_ratio, 2)}</td>
                    <td>${formatNumber(row.revenue_yoy, 2)}</td>
                    <td>${formatNumber(row.profit_yoy, 2)}</td>
                  </tr>
                `
              )
              .join("");
            table.innerHTML = body;
          }
        }

        function initIndexPage(data) {
          populateSummary(data);
          renderStatusBanner(data);
          setupWindowButtons(data);
          renderHighlights(data);
          buildImpactBars(data);
          renderNewsFeed(data, {
            targetId: "news-feed",
            monthId: "news-month",
            sourceId: "news-source",
            impactId: "news-impact",
            countId: "news-count",
            limit: 8,
          });
        }

        function initNewsPage(data) {
          renderStatusBanner(data);
          renderNewsFeed(data, {
            targetId: "news-feed-full",
            monthId: "news-month-full",
            sourceId: "news-source-full",
            impactId: "news-impact-full",
            countId: "news-count-full",
            limit: data.news.length,
          });
          buildImpactBars(data);
        }

        function initModelPage(data) {
          renderStatusBanner(data);
          populateSummary(data);
          buildModelCharts(data);
        }

        function initFundamentalsPage(data) {
          renderStatusBanner(data);
          buildFundamentalCharts(data);
        }

        document.addEventListener("DOMContentLoaded", async () => {
          currentLang = detectLanguage();
          applyStaticTranslations();
          initLanguageSelector();
          try {
            const data = await loadDashboardData();
            const page = document.body.dataset.page;
            if (page === "home") initIndexPage(data);
            if (page === "news") initNewsPage(data);
            if (page === "model") initModelPage(data);
            if (page === "fundamentals") initFundamentalsPage(data);
          } catch (error) {
            const banner = document.getElementById("status-banner");
            if (banner) {
              banner.className = "banner warning show";
              banner.textContent = t("common.pageLoadFailed", { message: error.message });
            }
          }
        });
        """
    )


def build_index_page() -> str:
    return (
        render_head("当升科技 Dashboard | 总览", "home", asset_version="__ASSET_VERSION__")
        + dedent(
            """\
            <div id="status-banner" class="banner"></div>
            <section class="hero">
              <div class="hero-grid">
                <div>
                  <p class="eyebrow" data-i18n="home.hero.kicker">当日信号</p>
                  <h2 id="hero-headline" class="hero-title">载入中</h2>
                  <p id="hero-subline" class="hero-meta">正在读取最新交易日信号。</p>
                  <p id="generated-at" class="hero-meta"></p>
                </div>
                <div id="signal-panel" class="signal-meter"></div>
              </div>
            </section>

            <section class="grid grid-3">
              <div id="metric-close"></div>
              <div id="metric-20d"></div>
              <div id="metric-relative"></div>
              <div id="metric-drawdown"></div>
              <div id="metric-model"></div>
              <div id="metric-news"></div>
            </section>

            <section class="grid grid-2" style="margin-top: 16px;">
              <div class="panel">
                <h2 data-i18n="home.price.title">股价与成交量</h2>
                <p class="panel-subtitle" data-i18n="home.price.subtitle">保留本地版的窗口切换，观察价格、量能和相对强弱。</p>
                <div class="controls">
                  <div class="button-group">
                    <button class="button" data-window="3M">3M</button>
                    <button class="button" data-window="6M">6M</button>
                    <button class="button active" data-window="1Y">1Y</button>
                    <button class="button" data-window="2Y">2Y</button>
                  </div>
                </div>
                <div id="price-chart" class="chart"></div>
              </div>
              <div class="panel">
                <h2 data-i18n="home.performance.title">相对创业板表现</h2>
                <p class="panel-subtitle" data-i18n="home.performance.subtitle">把股价和创业板指归一化后放在一起看，判断是否跑赢基准。</p>
                <div id="performance-chart" class="chart"></div>
              </div>
            </section>

            <section class="grid grid-2" style="margin-top: 16px;">
              <div class="panel">
                <h2 data-i18n="home.highlights.title">重点事件</h2>
                <p class="panel-subtitle" data-i18n="home.highlights.subtitle">保留最强正向和负向线索，给家里人看时信息密度会更高。</p>
                <div class="highlight-list">
                  <div id="positive-highlight"></div>
                  <div id="negative-highlight"></div>
                </div>
              </div>
              <div class="panel">
                <h2 data-i18n="home.impact.title">新闻方向分布</h2>
                <p class="panel-subtitle" data-i18n="home.impact.subtitle">把近两年的消息按偏利多、中性、偏利空粗分。</p>
                <div id="impact-chart" class="chart"></div>
              </div>
            </section>

<section class="panel" style="margin-top: 16px;">
  <h2 data-i18n="home.news.title">最新消息流</h2>
  <p class="panel-subtitle" data-i18n="home.news.subtitle">这里是更像本地版的新闻区块，支持月份、来源、方向筛选。</p>
  <div class="controls">
    <select id="news-month"></select>
    <select id="news-source"></select>
    <select id="news-impact"></select>
  </div>
  <p id="news-count" class="panel-subtitle"></p>
  <div id="news-feed" class="news-list"></div>
</section>
            """
        )
        + render_footer()
    )


def build_news_page() -> str:
    return (
        render_head("当升科技 Dashboard | 新闻", "news", asset_version="__ASSET_VERSION__")
        + dedent(
            """\
            <div id="status-banner" class="banner"></div>
            <section class="hero">
              <div class="hero-grid">
                <div>
                  <p class="eyebrow" data-i18n="news.hero.kicker">消息流</p>
                  <h2 class="hero-title" data-i18n="news.hero.title">消息流与方向筛选</h2>
                  <p class="hero-meta" data-i18n="news.hero.meta">把公告、研报、外部媒体合在一页里看，便于快速判断当天市场叙事。</p>
                </div>
                <div class="signal-meter">
                  <div class="label" data-i18n="news.hero.usageLabel">使用方式</div>
                  <div class="value" style="font-size: 1.45rem;" data-i18n="news.hero.usageValue">先看方向</div>
                  <div class="detail" data-i18n="news.hero.usageDetail">先筛月份，再看来源，最后挑偏利多或偏利空。这样比顺着时间线硬刷更有效。</div>
                </div>
              </div>
            </section>

            <section class="panel">
              <h2 data-i18n="news.impact.title">新闻方向分布</h2>
              <p class="panel-subtitle" data-i18n="news.impact.subtitle">先看整体基调，再去下面筛消息流。</p>
              <div id="impact-chart" class="chart"></div>
            </section>

            <section class="panel" style="margin-top: 16px;">
              <h2 data-i18n="news.feed.title">全部消息</h2>
              <p class="panel-subtitle" data-i18n="news.feed.subtitle">筛选器直接作用于下面这段消息流，按时间倒序展示，点击标题会跳原始链接。</p>
              <div class="controls">
                <select id="news-month-full"></select>
                <select id="news-source-full"></select>
                <select id="news-impact-full"></select>
              </div>
              <p id="news-count-full" class="panel-subtitle"></p>
              <div id="news-feed-full" class="news-list"></div>
            </section>
            """
        )
        + render_footer()
    )


def build_model_page() -> str:
    return (
        render_head("当升科技 Dashboard | 模型", "model", asset_version="__ASSET_VERSION__")
        + dedent(
            """\
            <div id="status-banner" class="banner"></div>
            <section class="hero">
              <div class="hero-grid">
                <div>
                  <p class="eyebrow" data-i18n="model.hero.kicker">模型视图</p>
                  <h2 class="hero-title" data-i18n="model.hero.title">当日信号与模型解释</h2>
                  <p class="hero-meta" data-i18n="model.hero.meta">这里重点展示“今天的结论是什么、模型有没有明显优势、主要驱动是什么”。</p>
                </div>
                <div id="signal-panel" class="signal-meter"></div>
              </div>
            </section>

            <section class="grid grid-3">
              <div id="metric-close"></div>
              <div id="metric-model"></div>
              <div id="metric-news"></div>
            </section>

            <section class="panel" style="margin-top: 16px;">
              <h2 data-i18n="model.summary.title">模型摘要</h2>
              <p class="panel-subtitle" data-i18n="model.summary.subtitle">分类模型关注方向，回归模型关注下一日收益率。</p>
              <div id="model-summary"></div>
            </section>

            <section class="grid grid-2" style="margin-top: 16px;">
              <div class="panel">
                <h2 data-i18n="model.compare.title">模型对比</h2>
                <p class="panel-subtitle" data-i18n="model.compare.subtitle">看 accuracy 和 AUC 是否真正有优势。</p>
                <div id="model-compare-chart" class="chart"></div>
              </div>
              <div class="panel">
                <h2 data-i18n="model.feature.title">特征重要性</h2>
                <p class="panel-subtitle" data-i18n="model.feature.subtitle">新闻和财务特征有没有实际进入排序靠前的位置。</p>
                <div id="feature-chart" class="chart"></div>
              </div>
            </section>
            """
        )
        + render_footer()
    )


def build_fundamentals_page() -> str:
    return (
        render_head("当升科技 Dashboard | 基本面", "fundamentals", asset_version="__ASSET_VERSION__")
        + dedent(
            """\
            <div id="status-banner" class="banner"></div>
            <section class="hero">
              <div class="hero-grid">
                <div>
                  <p class="eyebrow" data-i18n="fund.hero.kicker">基本面</p>
                  <h2 class="hero-title" data-i18n="fund.hero.title">季度财务快照</h2>
                  <p class="hero-meta" data-i18n="fund.hero.meta">把公告生效后的财务指标拉进来，看收入、利润、毛利率和 ROE 是否同步改善。</p>
                </div>
                <div class="signal-meter">
                  <div class="label" data-i18n="fund.hero.tipLabel">提示</div>
                  <div class="value" style="font-size: 1.35rem;" data-i18n="fund.hero.tipValue">适合中线跟踪</div>
                  <div class="detail" data-i18n="fund.hero.tipDetail">短线信号要看模型页，中线判断要把业绩兑现、债务和利润同比一起看。</div>
                </div>
              </div>
            </section>

            <section class="grid grid-2">
              <div class="panel">
                <h2 data-i18n="fund.profit.title">营收与净利润</h2>
                <p class="panel-subtitle" data-i18n="fund.profit.subtitle">最新 8 个季度的财务快照。</p>
                <div id="fundamentals-profit-chart" class="chart"></div>
              </div>
              <div class="panel">
                <h2 data-i18n="fund.ratio.title">毛利率、净利率与 ROE</h2>
                <p class="panel-subtitle" data-i18n="fund.ratio.subtitle">利润质量比单看营收更关键。</p>
                <div id="fundamentals-ratio-chart" class="chart"></div>
              </div>
            </section>

            <section class="panel" style="margin-top: 16px;">
              <h2 data-i18n="fund.table.title">财务表</h2>
              <p class="panel-subtitle" data-i18n="fund.table.subtitle">单位沿用抓取数据口径，适合作为快速对比视图。</p>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th data-i18n="fund.table.reportDate">报告期</th>
                      <th data-i18n="fund.table.revenue">营收</th>
                      <th data-i18n="fund.table.netProfit">净利润</th>
                      <th data-i18n="fund.table.grossMargin">毛利率</th>
                      <th data-i18n="fund.table.netMargin">净利率</th>
                      <th data-i18n="fund.table.roe">ROE</th>
                      <th data-i18n="fund.table.debtRatio">资产负债率</th>
                      <th data-i18n="fund.table.revenueYoy">营收同比</th>
                      <th data-i18n="fund.table.profitYoy">利润同比</th>
                    </tr>
                  </thead>
                  <tbody id="fundamentals-table"></tbody>
                </table>
              </div>
            </section>
            """
        )
        + render_footer()
    )


def write_snapshot_site(project_root: Path, site_dir: Path) -> None:
    payload = build_snapshot_payload(project_root)
    asset_version = payload["generated_at"].replace("-", "").replace(":", "").replace("+", "").replace("T", "")
    if site_dir.exists():
        shutil.rmtree(site_dir)
    (site_dir / "assets").mkdir(parents=True, exist_ok=True)
    (site_dir / "data").mkdir(parents=True, exist_ok=True)

    (site_dir / "assets" / "styles.css").write_text(build_stylesheet(), encoding="utf-8")
    (site_dir / "assets" / "app.js").write_text(build_javascript(), encoding="utf-8")
    (site_dir / "assets" / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    (site_dir / "data" / "dashboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (site_dir / "index.html").write_text(
        build_index_page().replace("__ASSET_VERSION__", asset_version),
        encoding="utf-8",
    )
    (site_dir / "news.html").write_text(
        build_news_page().replace("__ASSET_VERSION__", asset_version),
        encoding="utf-8",
    )
    (site_dir / "model.html").write_text(
        build_model_page().replace("__ASSET_VERSION__", asset_version),
        encoding="utf-8",
    )
    (site_dir / "fundamentals.html").write_text(
        build_fundamentals_page().replace("__ASSET_VERSION__", asset_version),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static multi-page dashboard site.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/ and reports/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/index.html"),
        help="Output entry HTML path. The full site is written to its parent directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    site_dir = output_path.parent
    write_snapshot_site(project_root, site_dir)


if __name__ == "__main__":
    main()
