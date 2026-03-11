from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots


COLOR_UP = "#c65a2e"
COLOR_DOWN = "#1f6f5f"
COLOR_NEUTRAL = "#58606b"


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
    news["body"] = news["body"].fillna("")
    news["impact"] = news["sentiment_score"].apply(classify_impact)
    source_labels = {
        "disclosure": "公司公告",
        "research": "券商研报",
        "news": "个股新闻",
        "web_news": "外部媒体",
    }
    news["source_label"] = news["source_type"].map(source_labels).fillna(news["source_type"])
    return {
        "metrics": metrics,
        "prices": prices.sort_values("date").reset_index(drop=True),
        "benchmark": benchmark.sort_values("date").reset_index(drop=True),
        "news": news.sort_values("timestamp", ascending=False).reset_index(drop=True),
        "quarterly": quarterly.sort_values("报告日").reset_index(drop=True),
    }


def build_price_figure(prices: pd.DataFrame) -> go.Figure:
    frame = prices.tail(260).copy()
    frame["ma20"] = frame["close"].rolling(20).mean()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame["close"], name="收盘价", mode="lines", line=dict(color=COLOR_UP, width=2.6)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame["ma20"], name="20日均线", mode="lines", line=dict(color=COLOR_DOWN, width=2, dash="dot")),
        row=1,
        col=1,
    )
    colors = [COLOR_UP if value >= 0 else COLOR_DOWN for value in frame["pct_change"].fillna(0)]
    fig.add_trace(
        go.Bar(x=frame["date"], y=frame["volume"], name="成交量", marker=dict(color=colors, opacity=0.78)),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=16, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#fbfbf8",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    fig.update_xaxes(showgrid=False)
    return fig


def build_relative_figure(prices: pd.DataFrame, benchmark: pd.DataFrame) -> go.Figure:
    frame = prices[["date", "close"]].merge(benchmark[["date", "benchmark_close"]], on="date", how="left").tail(260).copy()
    frame["当升科技"] = frame["close"] / frame["close"].iloc[0] * 100
    frame["创业板指"] = frame["benchmark_close"] / frame["benchmark_close"].iloc[0] * 100
    melted = frame.melt(id_vars=["date"], value_vars=["当升科技", "创业板指"], var_name="series", value_name="index_value")
    fig = px.line(
        melted,
        x="date",
        y="index_value",
        color="series",
        color_discrete_map={"当升科技": COLOR_UP, "创业板指": COLOR_DOWN},
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=16, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#fbfbf8",
        legend_title_text="",
        yaxis_title="起点=100",
        xaxis_title="",
    )
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


def build_news_figure(news: pd.DataFrame) -> go.Figure:
    monthly = (
        news.assign(month=news["effective_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(news_count=("title", "count"), avg_sentiment=("sentiment_score", "mean"))
        .reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly["month"], y=monthly["news_count"], name="消息数", marker=dict(color="rgba(31,111,95,0.82)")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["avg_sentiment"], name="平均情绪", mode="lines+markers", line=dict(color=COLOR_UP, width=2.4)),
        secondary_y=True,
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=16, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#fbfbf8",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(title_text="消息数", secondary_y=False, gridcolor="rgba(120,120,120,0.15)")
    fig.update_yaxes(title_text="平均情绪", secondary_y=True, gridcolor="rgba(120,120,120,0.15)")
    return fig


def build_fundamental_figure(quarterly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=quarterly["报告日"], y=quarterly["fin_revenue"], name="收入(十亿元)", marker=dict(color="rgba(31,111,95,0.82)")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=quarterly["报告日"],
            y=quarterly["fin_net_profit"],
            name="净利润(十亿元)",
            mode="lines+markers",
            line=dict(color=COLOR_UP, width=2.4),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=16, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#fbfbf8",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(title_text="收入", secondary_y=False, gridcolor="rgba(120,120,120,0.15)")
    fig.update_yaxes(title_text="净利润", secondary_y=True, gridcolor="rgba(120,120,120,0.15)")
    return fig


def metric_card(label: str, value: str, detail: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-detail">{detail}</div>
    </div>
    """


def render_snapshot(project_root: Path, output_path: Path | None = None) -> Path:
    payload = load_payload(project_root)
    metrics = payload["metrics"]
    prices = payload["prices"]
    benchmark = payload["benchmark"]
    news = payload["news"]
    quarterly = payload["quarterly"]

    latest_date = prices["date"].max()
    generated_at = pd.to_datetime(metrics["config"]["generated_at"])
    recent_news = news.head(12).copy()
    positive = news.sort_values("sentiment_score", ascending=False).head(6).copy()
    negative = news.sort_values("sentiment_score", ascending=True).head(6).copy()

    price_chart = to_html(build_price_figure(prices), include_plotlyjs=True, full_html=False)
    relative_chart = to_html(build_relative_figure(prices, benchmark), include_plotlyjs=False, full_html=False)
    news_chart = to_html(build_news_figure(news), include_plotlyjs=False, full_html=False)
    fundamental_chart = to_html(build_fundamental_figure(quarterly), include_plotlyjs=False, full_html=False)

    cards = [
        metric_card("最新收盘价", f"{metrics['forecast']['latest_close']:.2f}", f"截至 {latest_date:%Y-%m-%d}"),
        metric_card("两年累计涨幅", format_pct(metrics["descriptive_summary"]["stock_total_return"]), "当升科技前复权区间表现"),
        metric_card("相对创业板", format_pct(metrics["descriptive_summary"]["excess_return_vs_benchmark"]), f"创业板区间 {format_pct(metrics['descriptive_summary']['benchmark_total_return'])}"),
        metric_card("次日上涨概率", f"{metrics['forecast']['next_up_probability'] * 100:.1f}%", f"模型标签: {metrics['forecast']['label']}"),
        metric_card("次日预测收益", format_pct(metrics["forecast"]["predicted_return"]), "仅作辅助，不是交易指令"),
        metric_card("PDF 正文覆盖", str(metrics["data_summary"]["pdf_extract_ok"]), f"公告 {metrics['data_summary']['pdf_source_counts'].get('disclosure', 0)} / 研报 {metrics['data_summary']['pdf_source_counts'].get('research', 0)}"),
    ]

    recent_news_rows = "".join(
        f"""
        <tr>
          <td>{row.timestamp:%Y-%m-%d %H:%M}</td>
          <td>{row.source_label}</td>
          <td>{row.impact}</td>
          <td><a href="{row.source_link}" target="_blank" rel="noreferrer">{row.title}</a></td>
        </tr>
        """
        for row in recent_news.itertuples()
    )
    positive_rows = "".join(
        f"<li><a href='{row.source_link}' target='_blank' rel='noreferrer'>{row.timestamp:%Y-%m-%d} {row.title}</a> <span>{row.sentiment_score:.3f}</span></li>"
        for row in positive.itertuples()
    )
    negative_rows = "".join(
        f"<li><a href='{row.source_link}' target='_blank' rel='noreferrer'>{row.timestamp:%Y-%m-%d} {row.title}</a> <span>{row.sentiment_score:.3f}</span></li>"
        for row in negative.itertuples()
    )

    html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>当升科技每日看板</title>
      <style>
        body {{
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
          background: linear-gradient(180deg, #f4efe6 0%, #f9f7f1 45%, #f2ece2 100%);
          color: #1d2320;
        }}
        .page {{
          max-width: 1240px;
          margin: 0 auto;
          padding: 24px 18px 48px;
        }}
        .hero {{
          background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(247,240,231,0.92));
          border: 1px solid #d8cfc0;
          border-radius: 24px;
          padding: 22px 24px;
          box-shadow: 0 20px 60px rgba(86,96,107,0.08);
        }}
        .hero h1 {{
          margin: 8px 0 10px;
          font-size: 34px;
          line-height: 1.1;
        }}
        .kicker {{
          color: #1f6f5f;
          font-weight: 700;
          font-size: 13px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }}
        .subtitle {{
          color: #4d5652;
          font-size: 15px;
        }}
        .cards {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
          margin-top: 18px;
        }}
        .metric-card {{
          background: rgba(255,255,255,0.92);
          border: 1px solid #d8cfc0;
          border-radius: 18px;
          padding: 14px 16px;
        }}
        .metric-label {{
          color: #5a625d;
          font-size: 13px;
          margin-bottom: 6px;
        }}
        .metric-value {{
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 4px;
        }}
        .metric-detail {{
          color: #53605d;
          font-size: 13px;
        }}
        .grid-2 {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
          margin-top: 18px;
        }}
        .panel {{
          background: rgba(255,255,255,0.9);
          border: 1px solid #d8cfc0;
          border-radius: 20px;
          padding: 14px 16px 10px;
          box-shadow: 0 10px 30px rgba(86,96,107,0.06);
        }}
        .panel h2 {{
          font-size: 19px;
          margin: 2px 0 10px;
        }}
        .panel p {{
          color: #5e6763;
          font-size: 13px;
          margin: 0 0 10px;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }}
        th, td {{
          padding: 10px 8px;
          border-bottom: 1px solid #ece6db;
          vertical-align: top;
          text-align: left;
        }}
        th {{
          color: #5a625d;
          font-weight: 600;
        }}
        ul {{
          margin: 0;
          padding-left: 18px;
        }}
        li {{
          margin: 0 0 10px;
        }}
        li span {{
          display: inline-block;
          margin-left: 8px;
          color: #5a625d;
        }}
        a {{
          color: #1f6f5f;
          text-decoration: none;
        }}
        @media (max-width: 900px) {{
          .cards, .grid-2 {{
            grid-template-columns: 1fr;
          }}
          .hero h1 {{
            font-size: 28px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="page">
        <section class="hero">
          <div class="kicker">EASPRING 300073 SNAPSHOT</div>
          <h1>当升科技每日看板</h1>
          <div class="subtitle">
            生成时间 {generated_at:%Y-%m-%d %H:%M}。这是单文件静态版，适合直接发给家人查看。
          </div>
        </section>
        <section class="cards">
          {''.join(cards)}
        </section>
        <section class="grid-2">
          <div class="panel">
            <h2>股价与成交</h2>
            <p>最近约一年的收盘价、20 日均线和成交量。</p>
            {price_chart}
          </div>
          <div class="panel">
            <h2>相对走势</h2>
            <p>把当升科技和创业板指都归一到 100，更容易看相对强弱。</p>
            {relative_chart}
          </div>
        </section>
        <section class="grid-2">
          <div class="panel">
            <h2>月度消息热度</h2>
            <p>消息数量和情绪均值放在一起看。</p>
            {news_chart}
          </div>
          <div class="panel">
            <h2>季度基本面</h2>
            <p>收入与净利润的季度变化。</p>
            {fundamental_chart}
          </div>
        </section>
        <section class="grid-2">
          <div class="panel">
            <h2>最近消息</h2>
            <p>最新 12 条，可直接点原文。</p>
            <table>
              <thead>
                <tr><th>时间</th><th>来源</th><th>方向</th><th>标题</th></tr>
              </thead>
              <tbody>{recent_news_rows}</tbody>
            </table>
          </div>
          <div class="panel">
            <h2>情绪最强事件</h2>
            <p>左边偏利多，右边偏利空。</p>
            <div class="grid-2" style="margin-top: 0;">
              <div>
                <h3 style="margin: 0 0 10px; font-size: 16px;">偏利多</h3>
                <ul>{positive_rows}</ul>
              </div>
              <div>
                <h3 style="margin: 0 0 10px; font-size: 16px;">偏利空</h3>
                <ul>{negative_rows}</ul>
              </div>
            </div>
          </div>
        </section>
      </div>
    </body>
    </html>
    """

    output_path = output_path or (project_root / "reports" / "dashboard_snapshot.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
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
