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
    "disclosure": "公司公告",
    "research": "券商研报",
    "news": "个股新闻",
    "web_news": "外部媒体",
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


def render_nav(current_page: str) -> str:
    items = [
        ("index.html", "总览", "home"),
        ("news.html", "新闻", "news"),
        ("model.html", "模型", "model"),
        ("fundamentals.html", "基本面", "fundamentals"),
    ]
    links: list[str] = []
    for href, label, page in items:
        active = " active" if page == current_page else ""
        links.append(f'<a class="nav-link{active}" href="{href}">{label}</a>')
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
                <h1>当升科技跟踪 Dashboard</h1>
                <p class="subtle">股价、新闻、模型与基本面，每个工作日自动刷新一次。</p>
              </div>
              <nav class="topnav">{render_nav(page)}</nav>
            </header>
        """
    )


def render_footer() -> str:
    return dedent(
        """\
            <footer class="footer">
              <p>数据源包含日线行情、公司公告、券商研报、外部媒体和季度财务摘要。页面为静态前端站点，非实时盘中系统。</p>
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
        },
        "highlights": {
            "positive_event": positive_item[
                ["timestamp_label", "source_label", "impact", "title", "source_link", "sentiment_score"]
            ]
            .to_dict(orient="records"),
            "negative_event": negative_item[
                ["timestamp_label", "source_label", "impact", "title", "source_link", "sentiment_score"]
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
          --radius: 22px;
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
        }

        a {
          color: inherit;
          text-decoration: none;
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

        .nav-link {
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid var(--line);
          background: rgba(251, 248, 243, 0.9);
          color: #38413d;
          font-size: 0.92rem;
          font-weight: 700;
        }

        .nav-link.active {
          background: linear-gradient(135deg, rgba(198, 90, 46, 0.12), rgba(31, 111, 95, 0.08));
          border-color: rgba(198, 90, 46, 0.35);
        }

        .banner {
          margin-bottom: 16px;
          padding: 14px 16px;
          border-radius: 16px;
          border: 1px solid var(--line);
          background: rgba(251, 248, 243, 0.9);
          box-shadow: var(--shadow);
          display: none;
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
        }

        .empty-state {
          color: var(--muted);
          font-size: 0.9rem;
          padding: 14px 0;
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
        }
        """
    )


def build_javascript() -> str:
    return dedent(
        """\
        const IMPACT_CLASS = {
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

        async function loadDashboardData() {
          const version = encodeURIComponent(window.__DASHBOARD_VERSION__ || "");
          const response = await fetch(`data/dashboard.json?v=${version}`, { cache: "no-store" });
          if (!response.ok) {
            throw new Error("Failed to load dashboard data");
          }
          return response.json();
        }

        function formatPercent(value, digits = 2) {
          if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "N/A";
          }
          return `${(Number(value) * 100).toFixed(digits)}%`;
        }

        function formatNumber(value, digits = 2) {
          if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "N/A";
          }
          return Number(value).toFixed(digits);
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
            "最新收盘价",
            signal.latest_close ? `${formatNumber(signal.latest_close, 2)}` : "N/A",
            `${signal.signal_date || summary.latest_date} 收盘`,
            "neutral"
          );
          renderMetricCard(
            "metric-20d",
            "20日涨跌",
            formatPercent(summary.change_20d),
            "近一个月交易窗口",
            toneClass(summary.change_20d)
          );
          renderMetricCard(
            "metric-relative",
            "相对创业板",
            formatPercent(summary.relative_return),
            "相对基准累计收益",
            toneClass(summary.relative_return)
          );
          renderMetricCard(
            "metric-drawdown",
            "最大回撤",
            formatPercent(summary.max_drawdown),
            "两年区间最大回撤",
            toneClass(summary.max_drawdown, true)
          );
          renderMetricCard(
            "metric-model",
            "方向模型准确率",
            formatPercent(summary.classification_accuracy),
            `基线 ${formatPercent(summary.classification_baseline)}`,
            toneClass((summary.classification_accuracy ?? 0) - (summary.classification_baseline ?? 0))
          );
          renderMetricCard(
            "metric-news",
            "新闻样本量",
            `${summary.news_total ?? 0}`,
            `${summary.news_days ?? 0} 个有效消息日`,
            "neutral"
          );

          const signalPanel = document.getElementById("signal-panel");
          if (signalPanel) {
            const tone = toneClass(signal.predicted_return ?? 0);
            signalPanel.innerHTML = `
              <div class="label">当日信号</div>
              <div class="value">${formatPercent(signalProbability, 1)}</div>
              <div class="detail">下一交易日上涨概率</div>
              <div class="pill-row" style="margin-top: 12px;">
                <span class="pill ${tone}">${escapeHtml(signal.label || "中性")}</span>
                <span class="pill neutral">预测收益 ${formatPercent(signal.predicted_return, 2)}</span>
                <span class="pill neutral">下一交易日 ${escapeHtml(signal.next_session_date || "N/A")}</span>
              </div>
            `;
          }

          const headline = document.getElementById("hero-headline");
          const subline = document.getElementById("hero-subline");
          if (headline) {
            headline.textContent = signal.label || "震荡偏中性";
          }
          if (subline) {
            subline.textContent = `${signal.signal_date || summary.latest_date} 的收盘信号，供跟踪参考，不构成投资建议。`;
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
            label.textContent = `最后生成时间：${data.generated_at_label}`;
          }
          if (diffDays > 2) {
            banner.className = "banner warning show";
            banner.innerHTML = `数据可能已经过期。最新交易日是 ${escapeHtml(data.summary.latest_date)}，距今天约 ${diffDays} 天。`;
          } else {
            banner.className = "banner show";
            banner.innerHTML = `数据状态正常。最新交易日 ${escapeHtml(data.summary.latest_date)}，页面生成于 ${escapeHtml(data.generated_at_label)}。`;
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
                name: "当升科技",
                line: { color: "#c65a2e", width: 3 },
                yaxis: "y",
              },
              {
                x: rows.map((row) => row.date),
                y: rows.map((row) => row.volume),
                type: "bar",
                name: "成交量",
                marker: { color: "rgba(31, 111, 95, 0.22)" },
                yaxis: "y2",
              },
            ],
            {
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 42, r: 42, t: 16, b: 40 },
              xaxis: { showgrid: false },
              yaxis: { title: "收盘价", gridcolor: "rgba(216, 207, 192, 0.35)" },
              yaxis2: { title: "成交量", overlaying: "y", side: "right", showgrid: false },
              legend: { orientation: "h", y: 1.1 },
            },
            { responsive: true, displayModeBar: false }
          );

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
                  name: "当升科技",
                  line: { color: "#c65a2e", width: 3 },
                },
                {
                  x: rows.map((row) => row.date),
                  y: rows.map((row) => row.benchmark_index),
                  type: "scatter",
                  mode: "lines",
                  name: "创业板指",
                  line: { color: "#1f6f5f", width: 2.6 },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 40 },
                xaxis: { showgrid: false },
                yaxis: { title: "归一化指数", gridcolor: "rgba(216, 207, 192, 0.35)" },
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
          const renderItem = (container, item, fallback) => {
            if (!container) return;
            if (!item) {
              container.innerHTML = `<div class="empty-state">${fallback}</div>`;
              return;
            }
            const tone = IMPACT_CLASS[item.impact] || "neutral";
            container.innerHTML = `
              <div class="highlight-card">
                <div class="pill-row">
                  <span class="pill ${tone}">${escapeHtml(item.impact)}</span>
                  <span class="pill neutral">${escapeHtml(item.source_label)}</span>
                  <span class="pill neutral">${escapeHtml(item.timestamp_label)}</span>
                </div>
                <h3><a href="${escapeHtml(item.source_link)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a></h3>
                <p class="small">情绪分数 ${formatNumber(item.sentiment_score, 3)}</p>
              </div>
            `;
          };
          renderItem(positive, data.highlights.positive_event?.[0], "没有可展示的正向事件。");
          renderItem(negative, data.highlights.negative_event?.[0], "没有可展示的负向事件。");
        }

        function buildImpactBars(data) {
          const chart = document.getElementById("impact-chart");
          if (!chart) return;
          Plotly.newPlot(
            chart,
            [
              {
                x: ["偏利多", "中性", "偏利空"],
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
              xaxis: { showgrid: false },
              yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
            },
            { responsive: true, displayModeBar: false }
          );
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

          function setOptions(selectNode, defaultLabel, values, currentValue) {
            if (!selectNode) return "all";
            selectNode.innerHTML = `<option value="all">${defaultLabel}</option>${values
              .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
              .join("")}`;
            const nextValue = values.includes(currentValue) ? currentValue : "all";
            selectNode.value = nextValue;
            return nextValue;
          }

          if (monthSelect && !monthSelect.dataset.ready) {
            monthSelect.innerHTML = `<option value="all">全部月份</option>${allMonths
              .map((month) => `<option value="${escapeHtml(month)}">${escapeHtml(month)}</option>`)
              .join("")}`;
            monthSelect.dataset.ready = "true";
          }

          const rerender = () => {
            const month = monthSelect ? monthSelect.value : "all";
            const monthScopedRows = allNews.filter((item) => month === "all" || item.month === month);

            const sourceOptions = [...new Set(monthScopedRows.map((item) => item.source_label).filter(Boolean))];
            const source = setOptions(
              sourceSelect,
              "全部来源",
              sourceOptions,
              sourceSelect ? sourceSelect.value : "all"
            );

            const sourceScopedRows = monthScopedRows.filter((item) => source === "all" || item.source_label === source);
            const impactOptions = [...new Set(sourceScopedRows.map((item) => item.impact).filter(Boolean))];
            const impact = setOptions(
              impactSelect,
              "全部方向",
              impactOptions,
              impactSelect ? impactSelect.value : "all"
            );

            const limit = options.limit || allNews.length;
            const rows = monthScopedRows
              .filter((item) => source === "all" || item.source_label === source)
              .filter((item) => impact === "all" || item.impact === impact)
              .slice(0, limit);

            if (countTarget) {
              countTarget.textContent = `当前筛选结果 ${rows.length} 条`;
            }

            if (!rows.length) {
              target.innerHTML = `<div class="empty-state">当前筛选条件下没有新闻。你可以把来源或方向切回“全部”。</div>`;
              return;
            }

            target.innerHTML = rows
              .map((item) => {
                const tone = IMPACT_CLASS[item.impact] || "neutral";
                const body = item.body_excerpt ? `<p class="small">${escapeHtml(item.body_excerpt)}</p>` : "";
                return `
                  <article class="news-card">
                    <div class="pill-row">
                      <span class="pill ${tone}">${escapeHtml(item.impact)}</span>
                      <span class="pill neutral">${escapeHtml(item.source_label)}</span>
                      <span class="pill neutral">${escapeHtml(item.effective_date)}</span>
                    </div>
                    <h3><a href="${escapeHtml(item.source_link)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a></h3>
                    ${body}
                    <p class="small">情绪分数 ${formatNumber(item.sentiment_score, 3)} | 发布 ${escapeHtml(item.timestamp)}</p>
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
                  x: candidates.map((item) => item.model_name || item.model || "model"),
                  y: candidates.map((item) => item.accuracy ?? 0),
                  type: "bar",
                  name: "Accuracy",
                  marker: { color: "#c65a2e" },
                },
                {
                  x: candidates.map((item) => item.model_name || item.model || "model"),
                  y: candidates.map((item) => item.roc_auc ?? 0),
                  type: "bar",
                  name: "ROC AUC",
                  marker: { color: "#1f6f5f" },
                },
              ],
              {
                barmode: "group",
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 60 },
                xaxis: { tickangle: -18, showgrid: false },
                yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                legend: { orientation: "h", y: 1.15 },
              },
              { responsive: true, displayModeBar: false }
            );
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
                xaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                yaxis: { automargin: true },
              },
              { responsive: true, displayModeBar: false }
            );
          }

          const summary = document.getElementById("model-summary");
          if (summary) {
            const cls = data.models.classification;
            const reg = data.models.regression;
            summary.innerHTML = `
              <div class="grid grid-3">
                <div class="panel metric-card ${toneClass((cls.selected_metrics.accuracy ?? 0) - (cls.baseline_accuracy ?? 0))}">
                  <div class="metric-label">分类模型</div>
                  <div class="metric-value">${escapeHtml(cls.selected_model || "N/A")}</div>
                  <div class="metric-detail">Accuracy ${formatPercent(cls.selected_metrics.accuracy)} | AUC ${formatPercent(cls.selected_metrics.roc_auc)}</div>
                </div>
                <div class="panel metric-card ${toneClass(reg.selected_metrics.r2 ?? 0)}">
                  <div class="metric-label">回归模型</div>
                  <div class="metric-value">${escapeHtml(reg.selected_model || "N/A")}</div>
                  <div class="metric-detail">MAE ${formatNumber(reg.selected_metrics.mae, 4)} | R² ${formatNumber(reg.selected_metrics.r2, 4)}</div>
                </div>
                <div class="panel metric-card neutral">
                  <div class="metric-label">回归方程</div>
                  <div class="metric-value" style="font-size:1rem; line-height:1.45;">${escapeHtml(reg.ridge_equation || "未提供")}</div>
                  <div class="metric-detail">用于解释性参考</div>
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
                  name: "营收",
                  marker: { color: "rgba(198, 90, 46, 0.75)" },
                },
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.net_profit),
                  type: "scatter",
                  mode: "lines+markers",
                  name: "净利润",
                  line: { color: "#1f6f5f", width: 3 },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 40 },
                xaxis: { showgrid: false },
                yaxis: { gridcolor: "rgba(216, 207, 192, 0.35)" },
                legend: { orientation: "h", y: 1.14 },
              },
              { responsive: true, displayModeBar: false }
            );
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
                  name: "毛利率",
                  line: { color: "#c65a2e", width: 3 },
                },
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.net_margin),
                  type: "scatter",
                  mode: "lines+markers",
                  name: "净利率",
                  line: { color: "#58606b", width: 3 },
                },
                {
                  x: rows.map((row) => row.report_date),
                  y: rows.map((row) => row.roe),
                  type: "scatter",
                  mode: "lines+markers",
                  name: "ROE",
                  line: { color: "#1f6f5f", width: 3 },
                },
              ],
              {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 42, r: 18, t: 16, b: 40 },
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
          renderNewsFeed(data, { targetId: "news-feed", monthId: "news-month", sourceId: "news-source", impactId: "news-impact", countId: "news-count", limit: 8 });
        }

        function initNewsPage(data) {
          renderStatusBanner(data);
          renderNewsFeed(data, { targetId: "news-feed-full", monthId: "news-month-full", sourceId: "news-source-full", impactId: "news-impact-full", countId: "news-count-full", limit: data.news.length });
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
              banner.textContent = `页面加载失败：${error.message}`;
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
                  <p class="eyebrow">Daily Signal</p>
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
                <h2>股价与成交量</h2>
                <p class="panel-subtitle">保留本地版的窗口切换，观察价格、量能和相对强弱。</p>
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
                <h2>相对创业板表现</h2>
                <p class="panel-subtitle">把股价和创业板指归一化后放在一起看，判断是否跑赢基准。</p>
                <div id="performance-chart" class="chart"></div>
              </div>
            </section>

            <section class="grid grid-2" style="margin-top: 16px;">
              <div class="panel">
                <h2>重点事件</h2>
                <p class="panel-subtitle">保留最强正向和负向线索，给家里人看时信息密度会更高。</p>
                <div class="highlight-list">
                  <div id="positive-highlight"></div>
                  <div id="negative-highlight"></div>
                </div>
              </div>
              <div class="panel">
                <h2>新闻方向分布</h2>
                <p class="panel-subtitle">把近两年的消息按偏利多、中性、偏利空粗分。</p>
                <div id="impact-chart" class="chart"></div>
              </div>
            </section>

<section class="panel" style="margin-top: 16px;">
  <h2>最新消息流</h2>
  <p class="panel-subtitle">这里是更像本地版的新闻区块，支持月份、来源、方向筛选。</p>
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
                  <p class="eyebrow">News Stream</p>
                  <h2 class="hero-title">消息流与方向筛选</h2>
                  <p class="hero-meta">把公告、研报、外部媒体合在一页里看，便于快速判断当天市场叙事。</p>
                </div>
                <div class="signal-meter">
                  <div class="label">使用方式</div>
                  <div class="value" style="font-size: 1.45rem;">先看方向</div>
                  <div class="detail">先筛月份，再看来源，最后挑偏利多或偏利空。这样比顺着时间线硬刷更有效。</div>
                </div>
              </div>
            </section>

            <section class="panel">
              <h2>新闻方向分布</h2>
              <p class="panel-subtitle">先看整体基调，再去下面筛消息流。</p>
              <div id="impact-chart" class="chart"></div>
            </section>

            <section class="panel" style="margin-top: 16px;">
              <h2>全部消息</h2>
              <p class="panel-subtitle">筛选器直接作用于下面这段消息流，按时间倒序展示，点击标题会跳原始链接。</p>
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
                  <p class="eyebrow">Model View</p>
                  <h2 class="hero-title">当日信号与模型解释</h2>
                  <p class="hero-meta">这里重点展示“今天的结论是什么、模型有没有明显优势、主要驱动是什么”。</p>
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
              <h2>模型摘要</h2>
              <p class="panel-subtitle">分类模型关注方向，回归模型关注下一日收益率。</p>
              <div id="model-summary"></div>
            </section>

            <section class="grid grid-2" style="margin-top: 16px;">
              <div class="panel">
                <h2>模型对比</h2>
                <p class="panel-subtitle">看 accuracy 和 AUC 是否真正有优势。</p>
                <div id="model-compare-chart" class="chart"></div>
              </div>
              <div class="panel">
                <h2>特征重要性</h2>
                <p class="panel-subtitle">新闻和财务特征有没有实际进入排序靠前的位置。</p>
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
                  <p class="eyebrow">Fundamentals</p>
                  <h2 class="hero-title">季度财务快照</h2>
                  <p class="hero-meta">把公告生效后的财务指标拉进来，看收入、利润、毛利率和 ROE 是否同步改善。</p>
                </div>
                <div class="signal-meter">
                  <div class="label">提示</div>
                  <div class="value" style="font-size: 1.35rem;">适合中线跟踪</div>
                  <div class="detail">短线信号要看模型页，中线判断要把业绩兑现、债务和利润同比一起看。</div>
                </div>
              </div>
            </section>

            <section class="grid grid-2">
              <div class="panel">
                <h2>营收与净利润</h2>
                <p class="panel-subtitle">最新 8 个季度的财务快照。</p>
                <div id="fundamentals-profit-chart" class="chart"></div>
              </div>
              <div class="panel">
                <h2>毛利率、净利率与 ROE</h2>
                <p class="panel-subtitle">利润质量比单看营收更关键。</p>
                <div id="fundamentals-ratio-chart" class="chart"></div>
              </div>
            </section>

            <section class="panel" style="margin-top: 16px;">
              <h2>财务表</h2>
              <p class="panel-subtitle">单位沿用抓取数据口径，适合作为快速对比视图。</p>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>报告期</th>
                      <th>营收</th>
                      <th>净利润</th>
                      <th>毛利率</th>
                      <th>净利率</th>
                      <th>ROE</th>
                      <th>资产负债率</th>
                      <th>营收同比</th>
                      <th>利润同比</th>
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
