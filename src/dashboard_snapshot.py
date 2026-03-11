from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


COLOR_UP = "#c65a2e"
COLOR_DOWN = "#1f6f5f"


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


def load_payload(project_root: Path) -> dict[str, object]:
    metrics = json.loads((project_root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(project_root / "data" / "raw" / "stock_prices.csv", parse_dates=["date"])
    benchmark = pd.read_csv(project_root / "data" / "raw" / "benchmark_prices.csv", parse_dates=["date"])
    news = pd.read_csv(
        project_root / "data" / "processed" / "company_news.csv",
        parse_dates=["timestamp", "effective_date"],
    )
    quarterly = pd.read_csv(
        project_root / "data" / "processed" / "quarterly_financials.csv",
        parse_dates=["报告日", "公告日期", "effective_timestamp", "effective_date"],
    )

    source_labels = {
        "disclosure": "公司公告",
        "research": "券商研报",
        "news": "个股新闻",
        "web_news": "外部媒体",
    }
    news["source_label"] = news["source_type"].map(source_labels).fillna(news["source_type"])
    news["impact"] = news["sentiment_score"].apply(classify_impact)
    news["body"] = news["body"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 160)

    return {
        "metrics": metrics,
        "prices": prices.sort_values("date").reset_index(drop=True),
        "benchmark": benchmark.sort_values("date").reset_index(drop=True),
        "news": news.sort_values("timestamp", ascending=False).reset_index(drop=True),
        "quarterly": quarterly.sort_values("报告日").reset_index(drop=True),
    }


def metric_card(label: str, value: str, detail: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-detail">{detail}</div>
    </div>
    """


def build_snapshot_payload(payload: dict[str, object]) -> dict[str, object]:
    prices = payload["prices"].copy()
    benchmark = payload["benchmark"].copy()
    news = payload["news"].copy()
    quarterly = payload["quarterly"].copy()
    metrics = payload["metrics"]

    return {
        "metrics": {
            "forecast": metrics["forecast"],
            "data_summary": metrics["data_summary"],
            "descriptive_summary": metrics["descriptive_summary"],
            "classification": {
                "selected_model": metrics["classification"]["selected_model"],
                "selected_metrics": metrics["classification"]["selected_metrics"],
                "baseline_accuracy": metrics["classification"]["baseline_accuracy"],
                "candidates": metrics["classification"]["candidates"],
                "feature_importance": metrics["classification"]["feature_importance"][:12],
            },
            "regression": {
                "selected_model": metrics["regression"]["selected_model"],
                "selected_metrics": metrics["regression"]["selected_metrics"],
            },
            "config": metrics["config"],
        },
        "prices": prices.assign(date=prices["date"].dt.strftime("%Y-%m-%d"))[
            ["date", "close", "volume", "pct_change"]
        ].to_dict("records"),
        "benchmark": benchmark.assign(date=benchmark["date"].dt.strftime("%Y-%m-%d"))[
            ["date", "benchmark_close"]
        ].to_dict("records"),
        "news": news.assign(
            timestamp=news["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
            effective_date=news["effective_date"].dt.strftime("%Y-%m-%d"),
        )[
            ["timestamp", "effective_date", "source_label", "source_type", "impact", "title", "source_link", "sentiment_score", "body"]
        ].to_dict("records"),
        "quarterly": quarterly.assign(报告日=quarterly["报告日"].dt.strftime("%Y-%m-%d"))[
            ["报告日", "fin_revenue", "fin_net_profit", "fin_gross_margin", "fin_net_margin", "fin_roe", "fin_debt_ratio", "fin_revenue_yoy", "fin_profit_yoy"]
        ].to_dict("records"),
    }


def render_snapshot(project_root: Path, output_path: Path | None = None) -> Path:
    payload = load_payload(project_root)
    metrics = payload["metrics"]
    prices = payload["prices"]
    news = payload["news"]
    latest_date = prices["date"].max()
    generated_at = pd.to_datetime(metrics["config"]["generated_at"])
    recent_news_count = int((news["timestamp"] >= latest_date - pd.Timedelta(days=7)).sum())

    cards = [
        metric_card("最新收盘价", f"{metrics['forecast']['latest_close']:.2f}", f"截至 {latest_date:%Y-%m-%d}"),
        metric_card("两年累计涨幅", format_pct(metrics["descriptive_summary"]["stock_total_return"]), "当升科技前复权区间表现"),
        metric_card("相对创业板", format_pct(metrics["descriptive_summary"]["excess_return_vs_benchmark"]), f"创业板区间 {format_pct(metrics['descriptive_summary']['benchmark_total_return'])}"),
        metric_card("次日上涨概率", f"{metrics['forecast']['next_up_probability'] * 100:.1f}%", f"模型标签: {metrics['forecast']['label']}"),
        metric_card("次日预测收益", format_pct(metrics["forecast"]["predicted_return"]), "仅作辅助，不是交易指令"),
        metric_card("最近 7 天消息", str(recent_news_count), f"PDF 正文 {metrics['data_summary']['pdf_extract_ok']} 份"),
    ]

    payload_json = json.dumps(build_snapshot_payload(payload), ensure_ascii=False).replace("</", "<\\/")
    plotly_js = get_plotlyjs()

    html = """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>当升科技每日看板</title>
      <style>
        :root {
          --up: #c65a2e;
          --down: #1f6f5f;
          --ink: #1d2320;
          --muted: #58606b;
          --line: #d8cfc0;
          --paper: #fbf8f3;
          --bg: #f4efe6;
        }
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
          color: var(--ink);
          background:
            radial-gradient(circle at top right, rgba(198,90,46,0.10), transparent 20%),
            radial-gradient(circle at top left, rgba(31,111,95,0.10), transparent 18%),
            linear-gradient(180deg, var(--bg) 0%, #f7f4ed 50%, #f0e9de 100%);
        }
        .page {
          max-width: 1280px;
          margin: 0 auto;
          padding: 24px 18px 48px;
        }
        .hero, .panel {
          background: rgba(251,248,243,0.92);
          border: 1px solid var(--line);
          border-radius: 22px;
          box-shadow: 0 14px 40px rgba(86,96,107,0.07);
        }
        .hero {
          padding: 22px 24px;
          margin-bottom: 18px;
        }
        .kicker {
          color: var(--down);
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .hero h1 {
          margin: 8px 0 10px;
          font-size: 34px;
          line-height: 1.08;
        }
        .subtitle {
          color: #4d5652;
          font-size: 15px;
          line-height: 1.6;
          max-width: 900px;
        }
        .pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 14px;
        }
        .pill {
          padding: 8px 12px;
          border: 1px solid var(--line);
          border-radius: 999px;
          background: rgba(255,255,255,0.70);
          font-size: 13px;
          color: #49524e;
        }
        .status-banner {
          display: none;
          margin: 0 0 18px;
          padding: 14px 16px;
          border-radius: 18px;
          border: 1px solid var(--line);
          font-size: 14px;
          line-height: 1.6;
          box-shadow: 0 10px 24px rgba(86,96,107,0.06);
        }
        .status-banner.ok {
          display: block;
          background: rgba(31,111,95,0.10);
          border-color: rgba(31,111,95,0.30);
          color: #234a40;
        }
        .status-banner.warn {
          display: block;
          background: rgba(198,90,46,0.10);
          border-color: rgba(198,90,46,0.35);
          color: #6d3f25;
        }
        .cards {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
          margin-bottom: 18px;
        }
        .metric-card {
          border-radius: 18px;
          border: 1px solid var(--line);
          background: rgba(255,255,255,0.86);
          padding: 14px 16px;
          min-height: 112px;
        }
        .metric-label {
          color: #5a625d;
          font-size: 13px;
          margin-bottom: 6px;
        }
        .metric-value {
          font-size: 29px;
          font-weight: 700;
          margin-bottom: 4px;
        }
        .metric-detail {
          color: #56615d;
          font-size: 13px;
        }
        .toolbar {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: center;
          margin: 0 0 12px;
        }
        .segment {
          display: inline-flex;
          border: 1px solid var(--line);
          border-radius: 999px;
          overflow: hidden;
          background: rgba(255,255,255,0.76);
        }
        .segment button {
          border: 0;
          background: transparent;
          color: #49524e;
          padding: 9px 14px;
          cursor: pointer;
          font-size: 13px;
        }
        .segment button.active {
          background: var(--down);
          color: white;
        }
        .select {
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 9px 12px;
          background: rgba(255,255,255,0.82);
          color: var(--ink);
          font-size: 13px;
        }
        .grid-2 {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
          margin-bottom: 18px;
        }
        .panel {
          padding: 14px 16px 12px;
        }
        .panel h2 {
          margin: 2px 0 8px;
          font-size: 19px;
        }
        .signal-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
          gap: 16px;
          margin-bottom: 18px;
        }
        .signal-hero {
          display: grid;
          grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
          gap: 16px;
          align-items: stretch;
        }
        .signal-box {
          border: 1px solid var(--line);
          border-radius: 18px;
          background: rgba(255,255,255,0.72);
          padding: 14px 16px;
        }
        .signal-box h3 {
          margin: 0 0 10px;
          font-size: 16px;
        }
        .signal-number {
          font-size: 34px;
          font-weight: 700;
          margin: 4px 0;
        }
        .signal-copy {
          color: #54605c;
          font-size: 14px;
          line-height: 1.7;
        }
        .mini-metrics {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
          margin-top: 12px;
        }
        .mini-metric {
          border: 1px solid #e6ddd0;
          border-radius: 14px;
          padding: 10px 12px;
          background: rgba(255,255,255,0.74);
        }
        .mini-label {
          font-size: 12px;
          color: #65706c;
          margin-bottom: 4px;
        }
        .mini-value {
          font-size: 19px;
          font-weight: 700;
        }
        .panel p {
          margin: 0 0 10px;
          color: #5e6763;
          font-size: 13px;
          line-height: 1.5;
        }
        .plot {
          min-height: 310px;
        }
        .news-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }
        .news-table th, .news-table td {
          padding: 10px 8px;
          border-bottom: 1px solid #ece6db;
          text-align: left;
          vertical-align: top;
        }
        .news-table th {
          color: #5a625d;
          font-weight: 600;
        }
        .news-table a {
          color: var(--down);
          text-decoration: none;
        }
        .empty {
          color: #6a7470;
          font-size: 14px;
          padding: 18px 4px;
        }
        .event-list {
          list-style: none;
          padding: 0;
          margin: 0;
        }
        .event-list li {
          padding: 10px 0;
          border-bottom: 1px solid #ece6db;
        }
        .event-list a {
          color: var(--down);
          text-decoration: none;
        }
        .score {
          color: #5a625d;
          font-size: 13px;
          margin-left: 8px;
        }
        .footnote {
          color: #616a66;
          font-size: 13px;
          line-height: 1.6;
          margin-top: 12px;
        }
        @media (max-width: 960px) {
          .cards, .grid-2, .signal-grid, .signal-hero, .mini-metrics {
            grid-template-columns: 1fr;
          }
          .hero h1 {
            font-size: 28px;
          }
        }
      </style>
      <script>__PLOTLY_JS__</script>
    </head>
    <body>
      <script id="payload-data" type="application/json">__PAYLOAD_JSON__</script>
      <div class="page">
        <section class="hero">
          <div class="kicker">EASPRING 300073 PUBLIC DASHBOARD</div>
          <h1>当升科技每日跟踪看板</h1>
          <div class="subtitle">
            这是免费公网版，不需要你本地开着。当前页面由 GitHub Actions 自动生成并发布，
            工作日每天 09:15 中国时间更新一次。和本地版不同，这里是纯前端静态页，但我保留了时间窗口切换和新闻筛选。
          </div>
          <div class="pill-row">
            <div class="pill">最新生成时间：__GENERATED_AT__</div>
            <div class="pill">最近交易日：__LATEST_DATE__</div>
            <div class="pill">模型标签：__FORECAST_LABEL__</div>
            <div class="pill">当前站点地址：github.io</div>
          </div>
        </section>

        <section id="status-banner" class="status-banner"></section>

        <section class="cards">__CARDS__</section>

        <section class="signal-grid">
          <div class="panel">
            <h2>当日信号</h2>
            <p>这是最重要的一块。它告诉你模型对下一交易日的方向判断、收益倾向，以及这套判断到底靠不靠谱。</p>
            <div class="signal-hero">
              <div id="signal-gauge" class="plot" style="min-height:280px;"></div>
              <div class="signal-box">
                <h3 id="signal-headline">信号摘要</h3>
                <div id="signal-main-number" class="signal-number">--</div>
                <div id="signal-copy" class="signal-copy"></div>
                <div class="mini-metrics">
                  <div class="mini-metric">
                    <div class="mini-label">预测收益率</div>
                    <div id="signal-return" class="mini-value">--</div>
                  </div>
                  <div class="mini-metric">
                    <div class="mini-label">20日涨跌</div>
                    <div id="signal-20d" class="mini-value">--</div>
                  </div>
                  <div class="mini-metric">
                    <div class="mini-label">分类准确率</div>
                    <div id="signal-accuracy" class="mini-value">--</div>
                  </div>
                  <div class="mini-metric">
                    <div class="mini-label">多数基线</div>
                    <div id="signal-baseline" class="mini-value">--</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="panel">
            <h2>更新监控</h2>
            <p>用来判断这个站点是不是还在正常更新，避免你几周之后才发现它静默停了。</p>
            <div class="signal-box">
              <h3>站点健康状态</h3>
              <div id="monitor-headline" class="signal-number">--</div>
              <div id="monitor-copy" class="signal-copy"></div>
              <div class="mini-metrics">
                <div class="mini-metric">
                  <div class="mini-label">最新生成时间</div>
                  <div id="monitor-generated" class="mini-value" style="font-size:16px;">--</div>
                </div>
                <div class="mini-metric">
                  <div class="mini-label">最近交易日</div>
                  <div id="monitor-trade-date" class="mini-value" style="font-size:16px;">--</div>
                </div>
                <div class="mini-metric">
                  <div class="mini-label">最近 7 天消息</div>
                  <div id="monitor-news" class="mini-value">--</div>
                </div>
                <div class="mini-metric">
                  <div class="mini-label">PDF 覆盖</div>
                  <div id="monitor-pdf" class="mini-value">--</div>
                </div>
              </div>
              <div class="footnote" style="margin-top:14px;">
                如果这里显示“更新滞后”，优先检查仓库的 Actions 页面是否有失败记录。
              </div>
            </div>
          </div>
        </section>

        <section class="panel" style="margin-bottom:18px;">
          <h2>时间窗口</h2>
          <p>这里的切换不需要后端，浏览器本地就能重新画图。</p>
          <div class="toolbar">
            <div class="segment" id="range-buttons">
              <button data-days="90">3个月</button>
              <button data-days="180">6个月</button>
              <button data-days="365" class="active">1年</button>
              <button data-days="730">2年</button>
            </div>
          </div>
        </section>

        <section class="grid-2">
          <div class="panel">
            <h2>股价走势</h2>
            <p>收盘价与 20 日均线。</p>
            <div id="price-chart" class="plot"></div>
          </div>
          <div class="panel">
            <h2>相对创业板</h2>
            <p>把两者归一到 100，更容易看相对强弱。</p>
            <div id="relative-chart" class="plot"></div>
          </div>
        </section>

        <section class="grid-2">
          <div class="panel">
            <h2>成交量</h2>
            <p>红色表示当日涨跌幅非负，绿色表示当日下跌。</p>
            <div id="volume-chart" class="plot"></div>
          </div>
          <div class="panel">
            <h2>月度信息流</h2>
            <p>消息数和平均情绪一起看。</p>
            <div id="news-chart" class="plot"></div>
          </div>
        </section>

        <section class="grid-2">
          <div class="panel">
            <h2>季度收入与净利润</h2>
            <p>用来判断基本面是否在修复。</p>
            <div id="fundamental-chart" class="plot"></div>
          </div>
          <div class="panel">
            <h2>毛利率 / 净利率 / ROE</h2>
            <p>更偏经营质量视角。</p>
            <div id="margin-chart" class="plot"></div>
          </div>
        </section>

        <section class="grid-2">
          <div class="panel">
            <h2>模型对比</h2>
            <p>看当前选中的分类模型是不是明显优于其它候选模型。</p>
            <div id="model-compare-chart" class="plot"></div>
          </div>
          <div class="panel">
            <h2>模型最看重的特征</h2>
            <p>这不是当日逐笔归因，但能告诉你模型大体更重视什么。</p>
            <div id="feature-chart" class="plot"></div>
          </div>
        </section>

        <section class="panel" style="margin-bottom:18px;">
          <h2>新闻筛选</h2>
          <p>这里可以像本地版那样切月份和筛来源，只是数据更新频率是每天一次，不是实时。</p>
          <div class="toolbar">
            <select id="news-month" class="select"></select>
            <select id="news-source" class="select"></select>
            <select id="news-impact" class="select">
              <option value="全部">方向：全部</option>
              <option value="偏利多">方向：偏利多</option>
              <option value="中性">方向：中性</option>
              <option value="偏利空">方向：偏利空</option>
            </select>
          </div>
          <div id="news-summary" class="footnote"></div>
          <table class="news-table">
            <thead>
              <tr>
                <th>时间</th>
                <th>来源</th>
                <th>方向</th>
                <th>标题</th>
                <th>情绪</th>
              </tr>
            </thead>
            <tbody id="news-body"></tbody>
          </table>
        </section>

        <section class="grid-2">
          <div class="panel">
            <h2>最强利多事件</h2>
            <p>按情绪分数从高到低排序。</p>
            <ul id="positive-list" class="event-list"></ul>
          </div>
          <div class="panel">
            <h2>最强利空事件</h2>
            <p>按情绪分数从低到高排序。</p>
            <ul id="negative-list" class="event-list"></ul>
          </div>
        </section>
      </div>

      <script>
        const payload = JSON.parse(document.getElementById("payload-data").textContent);
        const state = {
          lookbackDays: 365,
          newsMonth: "全部",
          newsSource: "全部",
          newsImpact: "全部",
        };

        const colorUp = "__COLOR_UP__";
        const colorDown = "__COLOR_DOWN__";

        function latestDate(records, field = "date") {
          return records[records.length - 1][field];
        }

        function filterByDays(records, field, days) {
          if (days >= 730) {
            return records.slice();
          }
          const maxDate = new Date(latestDate(records, field));
          const start = new Date(maxDate);
          start.setDate(start.getDate() - (days - 1));
          return records.filter((row) => new Date(row[field]) >= start);
        }

        function activeRangeButtons() {
          document.querySelectorAll("#range-buttons button").forEach((button) => {
            button.classList.toggle("active", Number(button.dataset.days) === state.lookbackDays);
          });
        }

        function renderPriceCharts() {
          const prices = filterByDays(payload.prices, "date", state.lookbackDays);
          const dates = prices.map((row) => row.date);
          const close = prices.map((row) => row.close);
          const ma20 = prices.map((_, index) => {
            const subset = close.slice(Math.max(0, index - 19), index + 1);
            return subset.reduce((sum, value) => sum + value, 0) / subset.length;
          });
          const volume = prices.map((row) => row.volume);
          const pct = prices.map((row) => row.pct_change ?? 0);
          const volumeColors = pct.map((value) => (value >= 0 ? colorUp : colorDown));

          Plotly.react(
            "price-chart",
            [
              {x: dates, y: close, type: "scatter", mode: "lines", name: "收盘价", line: {color: colorUp, width: 2.7}},
              {x: dates, y: ma20, type: "scatter", mode: "lines", name: "20日均线", line: {color: colorDown, width: 2, dash: "dot"}},
            ],
            {
              height: 320,
              margin: {l: 36, r: 18, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              legend: {orientation: "h", y: 1.1, x: 0},
              xaxis: {showgrid: false},
              yaxis: {gridcolor: "rgba(120,120,120,0.15)"},
            },
            {responsive: true, displayModeBar: false}
          );

          Plotly.react(
            "volume-chart",
            [
              {x: dates, y: volume, type: "bar", name: "成交量", marker: {color: volumeColors, opacity: 0.82}},
            ],
            {
              height: 320,
              margin: {l: 36, r: 18, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              xaxis: {showgrid: false},
              yaxis: {gridcolor: "rgba(120,120,120,0.15)"},
            },
            {responsive: true, displayModeBar: false}
          );
        }

        function renderRelativeChart() {
          const priceWindow = filterByDays(payload.prices, "date", state.lookbackDays);
          const benchmarkMap = new Map(payload.benchmark.map((row) => [row.date, row.benchmark_close]));
          const merged = priceWindow
            .map((row) => ({
              date: row.date,
              close: row.close,
              benchmark_close: benchmarkMap.get(row.date),
            }))
            .filter((row) => row.benchmark_close != null);
          const firstClose = merged[0].close;
          const firstBenchmark = merged[0].benchmark_close;

          Plotly.react(
            "relative-chart",
            [
              {
                x: merged.map((row) => row.date),
                y: merged.map((row) => row.close / firstClose * 100),
                type: "scatter",
                mode: "lines",
                name: "当升科技",
                line: {color: colorUp, width: 2.7},
              },
              {
                x: merged.map((row) => row.date),
                y: merged.map((row) => row.benchmark_close / firstBenchmark * 100),
                type: "scatter",
                mode: "lines",
                name: "创业板指",
                line: {color: colorDown, width: 2.4},
              },
            ],
            {
              height: 320,
              margin: {l: 42, r: 18, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              legend: {orientation: "h", y: 1.1, x: 0},
              xaxis: {showgrid: false},
              yaxis: {title: "起点=100", gridcolor: "rgba(120,120,120,0.15)"},
            },
            {responsive: true, displayModeBar: false}
          );
        }

        function monthKey(dateString) {
          return dateString.slice(0, 7);
        }

        function renderNewsMonthlyChart() {
          const monthly = {};
          payload.news.forEach((row) => {
            const key = monthKey(row.effective_date);
            if (!monthly[key]) {
              monthly[key] = {count: 0, sentimentSum: 0};
            }
            monthly[key].count += 1;
            monthly[key].sentimentSum += Number(row.sentiment_score || 0);
          });
          const months = Object.keys(monthly).sort();
          const counts = months.map((key) => monthly[key].count);
          const sentiment = months.map((key) => monthly[key].sentimentSum / monthly[key].count);

          Plotly.react(
            "news-chart",
            [
              {x: months, y: counts, type: "bar", name: "消息数", marker: {color: "rgba(31,111,95,0.82)"}, yaxis: "y"},
              {x: months, y: sentiment, type: "scatter", mode: "lines+markers", name: "平均情绪", line: {color: colorUp, width: 2.4}, yaxis: "y2"},
            ],
            {
              height: 320,
              margin: {l: 36, r: 36, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              legend: {orientation: "h", y: 1.1, x: 0},
              xaxis: {showgrid: false},
              yaxis: {title: "消息数", gridcolor: "rgba(120,120,120,0.15)"},
              yaxis2: {title: "平均情绪", overlaying: "y", side: "right", gridcolor: "rgba(120,120,120,0.15)"},
            },
            {responsive: true, displayModeBar: false}
          );
        }

        function renderFundamentalCharts() {
          const quarterly = payload.quarterly;
          const dates = quarterly.map((row) => row["报告日"]);

          Plotly.react(
            "fundamental-chart",
            [
              {x: dates, y: quarterly.map((row) => row.fin_revenue), type: "bar", name: "收入(十亿元)", marker: {color: "rgba(31,111,95,0.82)"}, yaxis: "y"},
              {x: dates, y: quarterly.map((row) => row.fin_net_profit), type: "scatter", mode: "lines+markers", name: "净利润(十亿元)", line: {color: colorUp, width: 2.4}, yaxis: "y2"},
            ],
            {
              height: 320,
              margin: {l: 36, r: 36, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              legend: {orientation: "h", y: 1.1, x: 0},
              xaxis: {showgrid: false},
              yaxis: {title: "收入", gridcolor: "rgba(120,120,120,0.15)"},
              yaxis2: {title: "净利润", overlaying: "y", side: "right", gridcolor: "rgba(120,120,120,0.15)"},
            },
            {responsive: true, displayModeBar: false}
          );

          Plotly.react(
            "margin-chart",
            [
              {x: dates, y: quarterly.map((row) => row.fin_gross_margin), type: "scatter", mode: "lines+markers", name: "毛利率", line: {color: colorUp, width: 2.4}},
              {x: dates, y: quarterly.map((row) => row.fin_net_margin), type: "scatter", mode: "lines+markers", name: "净利率", line: {color: colorDown, width: 2.4}},
              {x: dates, y: quarterly.map((row) => row.fin_roe), type: "scatter", mode: "lines+markers", name: "ROE", line: {color: "#8b6f47", width: 2.2, dash: "dot"}},
            ],
            {
              height: 320,
              margin: {l: 36, r: 18, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              legend: {orientation: "h", y: 1.1, x: 0},
              xaxis: {showgrid: false},
              yaxis: {title: "百分比", gridcolor: "rgba(120,120,120,0.15)"},
            },
            {responsive: true, displayModeBar: false}
          );
        }

        function renderSignalPanel() {
          const forecast = payload.metrics.forecast;
          const descriptive = payload.metrics.descriptive_summary;
          const cls = payload.metrics.classification;
          const upProbability = Number(forecast.next_up_probability || 0) * 100;
          const predictedReturn = Number(forecast.predicted_return || 0);
          const accuracy = Number(cls.selected_metrics.accuracy || 0);
          const baseline = Number(cls.baseline_accuracy || 0);

          Plotly.react(
            "signal-gauge",
            [
              {
                type: "indicator",
                mode: "gauge+number",
                value: upProbability,
                number: {suffix: "%", font: {size: 34}},
                title: {text: "下一交易日上涨概率"},
                gauge: {
                  axis: {range: [0, 100]},
                  bar: {color: colorUp},
                  steps: [
                    {range: [0, 40], color: "rgba(31,111,95,0.18)"},
                    {range: [40, 60], color: "rgba(88,96,107,0.15)"},
                    {range: [60, 100], color: "rgba(198,90,46,0.18)"},
                  ],
                  threshold: {line: {color: "#1d2320", width: 4}, value: 50},
                },
              },
            ],
            {
              height: 280,
              margin: {l: 8, r: 8, t: 14, b: 8},
              paper_bgcolor: "rgba(255,255,255,0.72)",
            },
            {responsive: true, displayModeBar: false}
          );

          document.getElementById("signal-headline").textContent = `模型标签：${forecast.label}`;
          document.getElementById("signal-main-number").textContent = `${upProbability.toFixed(1)}%`;
          document.getElementById("signal-copy").textContent =
            `对 ${forecast.next_session_date} 的判断是“${forecast.label}”。如果你只看一个数字，就看上涨概率和预测收益率；如果这两个都接近中轴，就说明它不是高确信度信号。`;
          document.getElementById("signal-return").textContent = `${(predictedReturn * 100).toFixed(2)}%`;
          document.getElementById("signal-20d").textContent = `${(Number(descriptive.latest_20d_return || 0) * 100).toFixed(2)}%`;
          document.getElementById("signal-accuracy").textContent = `${(accuracy * 100).toFixed(1)}%`;
          document.getElementById("signal-baseline").textContent = `${(baseline * 100).toFixed(1)}%`;
        }

        function renderStatusBanner() {
          const generatedAt = new Date(payload.metrics.config.generated_at);
          const now = new Date();
          const ageHours = (now - generatedAt) / 36e5;
          const banner = document.getElementById("status-banner");
          const actionsUrl = "https://github.com/Shuyu-G/easpring-300073-dashboard/actions";
          const stale = ageHours > 54;
          banner.className = `status-banner ${stale ? "warn" : "ok"}`;
          banner.innerHTML = stale
            ? `更新提醒：这份页面距离上次成功生成已经超过 <strong>${Math.round(ageHours)}</strong> 小时，可能自动更新停了。请检查 <a href="${actionsUrl}" target="_blank" rel="noreferrer">GitHub Actions</a>。`
            : `更新状态正常：页面最近一次成功生成于 <strong>${generatedAt.toLocaleString("zh-CN")}</strong>，目前看起来还在按计划工作。`;

          document.getElementById("monitor-headline").textContent = stale ? "更新滞后" : "更新正常";
          document.getElementById("monitor-copy").textContent = stale
            ? "这通常意味着最近的定时任务失败了，或者没有按预期执行。"
            : "最近的自动更新仍在工作，当前页面不是本地缓存旧页面。";
          document.getElementById("monitor-generated").textContent = generatedAt.toLocaleString("zh-CN");
          document.getElementById("monitor-trade-date").textContent = payload.prices[payload.prices.length - 1].date;
          document.getElementById("monitor-news").textContent = String(
            payload.news.filter((row) => {
              const diff = (new Date(payload.prices[payload.prices.length - 1].date) - new Date(row.effective_date)) / 86400000;
              return diff <= 7;
            }).length
          );
          document.getElementById("monitor-pdf").textContent = String(payload.metrics.data_summary.pdf_extract_ok || 0);
        }

        function renderModelCharts() {
          const candidates = payload.metrics.classification.candidates || [];
          Plotly.react(
            "model-compare-chart",
            [
              {
                x: candidates.map((row) => row.name),
                y: candidates.map((row) => Number(row.accuracy || 0)),
                type: "bar",
                name: "准确率",
                marker: {color: "rgba(31,111,95,0.82)"},
              },
              {
                x: candidates.map((row) => row.name),
                y: candidates.map((row) => Number(row.balanced_accuracy || 0)),
                type: "bar",
                name: "平衡准确率",
                marker: {color: "rgba(198,90,46,0.82)"},
              },
              {
                x: candidates.map((row) => row.name),
                y: candidates.map((row) => Number(row.roc_auc || 0)),
                type: "bar",
                name: "ROC AUC",
                marker: {color: "rgba(139,111,71,0.82)"},
              },
            ],
            {
              barmode: "group",
              height: 320,
              margin: {l: 36, r: 18, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              legend: {orientation: "h", y: 1.1, x: 0},
              yaxis: {gridcolor: "rgba(120,120,120,0.15)"},
              xaxis: {showgrid: false},
            },
            {responsive: true, displayModeBar: false}
          );

          const importance = (payload.metrics.classification.feature_importance || [])
            .slice()
            .sort((a, b) => Number(a.importance || 0) - Number(b.importance || 0));
          Plotly.react(
            "feature-chart",
            [
              {
                x: importance.map((row) => Number(row.importance || 0)),
                y: importance.map((row) => row.feature),
                type: "bar",
                orientation: "h",
                marker: {
                  color: importance.map((row) => Number(row.importance || 0) >= 0 ? colorUp : colorDown),
                },
              },
            ],
            {
              height: 320,
              margin: {l: 96, r: 18, t: 10, b: 30},
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.72)",
              xaxis: {gridcolor: "rgba(120,120,120,0.15)"},
              yaxis: {showgrid: false},
            },
            {responsive: true, displayModeBar: false}
          );
        }

        function populateNewsFilters() {
          const monthSelect = document.getElementById("news-month");
          const sourceSelect = document.getElementById("news-source");

          const months = ["全部", ...Array.from(new Set(payload.news.map((row) => monthKey(row.effective_date)))).sort().reverse()];
          monthSelect.innerHTML = months.map((value) => `<option value="${value}">月份：${value}</option>`).join("");
          monthSelect.value = "全部";

          const sources = ["全部", ...Array.from(new Set(payload.news.map((row) => row.source_label)))];
          sourceSelect.innerHTML = sources.map((value) => `<option value="${value}">来源：${value}</option>`).join("");
          sourceSelect.value = "全部";
        }

        function filteredNews() {
          return payload.news.filter((row) => {
            const matchMonth = state.newsMonth === "全部" || monthKey(row.effective_date) === state.newsMonth;
            const matchSource = state.newsSource === "全部" || row.source_label === state.newsSource;
            const matchImpact = state.newsImpact === "全部" || row.impact === state.newsImpact;
            return matchMonth && matchSource && matchImpact;
          });
        }

        function renderNewsTable() {
          const rows = filteredNews();
          const body = document.getElementById("news-body");
          const summary = document.getElementById("news-summary");

          if (!rows.length) {
            body.innerHTML = `<tr><td colspan="5" class="empty">当前筛选条件下没有匹配的消息。</td></tr>`;
            summary.textContent = "共 0 条。";
            return;
          }

          summary.textContent = `当前筛选结果 ${rows.length} 条，平均情绪 ${(
            rows.reduce((sum, row) => sum + Number(row.sentiment_score || 0), 0) / rows.length
          ).toFixed(3)}。`;

          body.innerHTML = rows.slice(0, 60).map((row) => `
            <tr>
              <td>${row.timestamp}</td>
              <td>${row.source_label}</td>
              <td>${row.impact}</td>
              <td><a href="${row.source_link}" target="_blank" rel="noreferrer">${row.title}</a><div style="color:#66706c;font-size:12px;margin-top:4px;">${row.body || ""}</div></td>
              <td>${Number(row.sentiment_score || 0).toFixed(3)}</td>
            </tr>
          `).join("");
        }

        function renderPositiveNegative() {
          const positive = payload.news.slice().sort((a, b) => Number(b.sentiment_score || 0) - Number(a.sentiment_score || 0)).slice(0, 8);
          const negative = payload.news.slice().sort((a, b) => Number(a.sentiment_score || 0) - Number(b.sentiment_score || 0)).slice(0, 8);
          document.getElementById("positive-list").innerHTML = positive.map((row) => `
            <li><a href="${row.source_link}" target="_blank" rel="noreferrer">${row.timestamp.slice(0, 10)} ${row.title}</a><span class="score">${Number(row.sentiment_score || 0).toFixed(3)}</span></li>
          `).join("");
          document.getElementById("negative-list").innerHTML = negative.map((row) => `
            <li><a href="${row.source_link}" target="_blank" rel="noreferrer">${row.timestamp.slice(0, 10)} ${row.title}</a><span class="score">${Number(row.sentiment_score || 0).toFixed(3)}</span></li>
          `).join("");
        }

        function bindControls() {
          document.querySelectorAll("#range-buttons button").forEach((button) => {
            button.addEventListener("click", () => {
              state.lookbackDays = Number(button.dataset.days);
              activeRangeButtons();
              renderPriceCharts();
              renderRelativeChart();
            });
          });

          document.getElementById("news-month").addEventListener("change", (event) => {
            state.newsMonth = event.target.value;
            renderNewsTable();
          });
          document.getElementById("news-source").addEventListener("change", (event) => {
            state.newsSource = event.target.value;
            renderNewsTable();
          });
          document.getElementById("news-impact").addEventListener("change", (event) => {
            state.newsImpact = event.target.value;
            renderNewsTable();
          });
        }

        function init() {
          renderStatusBanner();
          activeRangeButtons();
          populateNewsFilters();
          bindControls();
          renderSignalPanel();
          renderPriceCharts();
          renderRelativeChart();
          renderNewsMonthlyChart();
          renderFundamentalCharts();
          renderModelCharts();
          renderNewsTable();
          renderPositiveNegative();
        }

        init();
      </script>
    </body>
    </html>
    """

    html = (
        html.replace("__PLOTLY_JS__", plotly_js)
        .replace("__PAYLOAD_JSON__", payload_json)
        .replace("__GENERATED_AT__", generated_at.strftime("%Y-%m-%d %H:%M"))
        .replace("__LATEST_DATE__", latest_date.strftime("%Y-%m-%d"))
        .replace("__FORECAST_LABEL__", metrics["forecast"]["label"])
        .replace("__CARDS__", "".join(cards))
        .replace("__COLOR_UP__", COLOR_UP)
        .replace("__COLOR_DOWN__", COLOR_DOWN)
    )

    output_path = output_path or (project_root / "reports" / "dashboard_snapshot.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a static dashboard snapshot.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/, reports/, and models/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated HTML snapshot.",
    )
    args = parser.parse_args()

    output = render_snapshot(args.project_root.resolve(), output_path=args.output.resolve() if args.output else None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
