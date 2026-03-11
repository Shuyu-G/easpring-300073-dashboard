from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.easpring_pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
COLOR_UP = "#c65a2e"
COLOR_DOWN = "#1f6f5f"
COLOR_NEUTRAL = "#58606b"
COLOR_BG = "#f4efe6"
COLOR_PANEL = "#fbf8f3"
COLOR_LINE = "#d8cfc0"
SOURCE_LABELS = {
    "disclosure": "公司公告",
    "research": "券商研报",
    "news": "个股新闻",
    "web_news": "外部媒体",
}


st.set_page_config(
    page_title="当升科技 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(198, 90, 46, 0.10), transparent 22%),
            radial-gradient(circle at top left, rgba(31, 111, 95, 0.10), transparent 18%),
            linear-gradient(180deg, {COLOR_BG} 0%, #f8f5ef 48%, #f1ebdf 100%);
        color: #1d2320;
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3 {{
        font-family: "Avenir Next", "PingFang SC", "Noto Serif SC", serif;
        letter-spacing: 0.01em;
    }}
    [data-testid="stSidebar"] {{
        background: rgba(251, 248, 243, 0.95);
        border-right: 1px solid {COLOR_LINE};
    }}
    .hero {{
        padding: 1.2rem 1.4rem;
        border: 1px solid {COLOR_LINE};
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(251, 248, 243, 0.96), rgba(247, 240, 231, 0.90)),
            {COLOR_PANEL};
        box-shadow: 0 20px 60px rgba(86, 96, 107, 0.09);
        margin-bottom: 1rem;
    }}
    .hero-kicker {{
        color: {COLOR_DOWN};
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .hero-title {{
        font-size: 2.1rem;
        line-height: 1.15;
        margin: 0.15rem 0 0.35rem 0;
        color: #151815;
        font-weight: 700;
    }}
    .hero-subtitle {{
        color: #4d5652;
        font-size: 0.98rem;
        margin: 0;
    }}
    .metric-card {{
        border-radius: 18px;
        border: 1px solid {COLOR_LINE};
        padding: 0.95rem 1rem;
        min-height: 128px;
        background: rgba(251, 248, 243, 0.92);
        box-shadow: 0 14px 32px rgba(86, 96, 107, 0.07);
    }}
    .metric-card.up {{
        background: linear-gradient(180deg, rgba(198, 90, 46, 0.10), rgba(251, 248, 243, 0.96));
    }}
    .metric-card.down {{
        background: linear-gradient(180deg, rgba(31, 111, 95, 0.10), rgba(251, 248, 243, 0.96));
    }}
    .metric-card.neutral {{
        background: linear-gradient(180deg, rgba(88, 96, 107, 0.08), rgba(251, 248, 243, 0.96));
    }}
    .metric-label {{
        font-size: 0.84rem;
        color: #5a625d;
        margin-bottom: 0.4rem;
    }}
    .metric-value {{
        font-size: 1.75rem;
        line-height: 1.05;
        font-weight: 700;
        color: #161a18;
        margin-bottom: 0.3rem;
    }}
    .metric-detail {{
        font-size: 0.88rem;
        color: #53605d;
    }}
    .panel-caption {{
        color: #5e6763;
        font-size: 0.85rem;
    }}
    .news-pill {{
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid {COLOR_LINE};
        background: rgba(255, 255, 255, 0.7);
        margin-right: 0.35rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_payload() -> dict[str, object]:
    metrics = json.loads((PROJECT_ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "stock_prices.csv", parse_dates=["date"])
    benchmark = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "benchmark_prices.csv", parse_dates=["date"])
    news = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "company_news.csv",
        parse_dates=["timestamp", "effective_date"],
    )
    daily_news = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "daily_news_features.csv",
        parse_dates=["date"],
    )
    quarterly = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "quarterly_financials.csv",
        parse_dates=["报告日", "公告日期", "effective_timestamp", "effective_date"],
    )
    training_data = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "training_data.csv",
        parse_dates=["date"],
    )
    news["body"] = news["body"].fillna("")
    news["source_label"] = news["source_type"].map(SOURCE_LABELS).fillna(news["source_type"])
    news["impact"] = news["sentiment_score"].apply(classify_impact)
    return {
        "metrics": metrics,
        "prices": prices.sort_values("date").reset_index(drop=True),
        "benchmark": benchmark.sort_values("date").reset_index(drop=True),
        "news": news.sort_values("timestamp", ascending=False).reset_index(drop=True),
        "daily_news": daily_news.sort_values("date").reset_index(drop=True),
        "quarterly": quarterly.sort_values("报告日").reset_index(drop=True),
        "training_data": training_data.sort_values("date").reset_index(drop=True),
    }


def classify_impact(score: float) -> str:
    if pd.isna(score):
        return "中性"
    if score >= 0.35:
        return "偏利多"
    if score <= -0.35:
        return "偏利空"
    return "中性"


def metric_tone(value: float | None, reverse: bool = False) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    positive = value < 0 if reverse else value > 0
    negative = value > 0 if reverse else value < 0
    if positive:
        return "up"
    if negative:
        return "down"
    return "neutral"


def render_metric_card(container: st.delta_generator.DeltaGenerator, label: str, value: str, detail: str, tone: str) -> None:
    container.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def make_performance_frame(prices: pd.DataFrame, benchmark: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    frame = prices[["date", "close"]].merge(
        benchmark[["date", "benchmark_close"]],
        on="date",
        how="left",
    )
    start_date = frame["date"].max() - pd.Timedelta(days=lookback_days - 1)
    frame = frame[frame["date"] >= start_date].copy()
    if frame.empty:
        return frame
    frame["当升科技"] = frame["close"] / frame["close"].iloc[0] * 100
    frame["创业板指"] = frame["benchmark_close"] / frame["benchmark_close"].iloc[0] * 100
    return frame


def price_figure(prices: pd.DataFrame, lookback_days: int) -> go.Figure:
    frame = prices.copy()
    start_date = frame["date"].max() - pd.Timedelta(days=lookback_days - 1)
    frame = frame[frame["date"] >= start_date].copy()
    frame["ma20"] = frame["close"].rolling(20).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["close"],
            name="收盘价",
            mode="lines",
            line=dict(color=COLOR_UP, width=2.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["ma20"],
            name="20日均线",
            mode="lines",
            line=dict(color=COLOR_DOWN, width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )
    bar_colors = [COLOR_UP if value >= 0 else COLOR_DOWN for value in frame["pct_change"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["volume"],
            name="成交量",
            marker=dict(color=bar_colors, opacity=0.78),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


def performance_figure(prices: pd.DataFrame, benchmark: pd.DataFrame, lookback_days: int) -> go.Figure:
    frame = make_performance_frame(prices, benchmark, lookback_days)
    melted = frame.melt(
        id_vars=["date"],
        value_vars=["当升科技", "创业板指"],
        var_name="series",
        value_name="index_value",
    )
    fig = px.line(
        melted,
        x="date",
        y="index_value",
        color="series",
        color_discrete_map={"当升科技": COLOR_UP, "创业板指": COLOR_DOWN},
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend_title_text="",
        yaxis_title="起点=100",
        xaxis_title="",
    )
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


def monthly_news_figure(news: pd.DataFrame) -> go.Figure:
    monthly = (
        news.assign(month=news["effective_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            news_count=("title", "count"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["news_count"],
            name="消息数",
            marker=dict(color="rgba(31,111,95,0.82)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_sentiment"],
            name="平均情绪",
            mode="lines+markers",
            line=dict(color=COLOR_UP, width=2.6),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(title_text="消息数", secondary_y=False, gridcolor="rgba(120,120,120,0.15)")
    fig.update_yaxes(title_text="平均情绪", secondary_y=True, gridcolor="rgba(120,120,120,0.15)")
    return fig


def source_mix_figure(news: pd.DataFrame) -> go.Figure:
    monthly = (
        news.assign(month=news["effective_date"].dt.to_period("M").astype(str))
        .groupby(["month", "source_label"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        monthly,
        x="month",
        y="count",
        color="source_label",
        barmode="stack",
        color_discrete_sequence=["#1f6f5f", "#c65a2e", "#8b6f47", "#55606b"],
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend_title_text="来源",
        xaxis_title="",
        yaxis_title="消息数",
    )
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


def fundamentals_figure(quarterly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=quarterly["报告日"],
            y=quarterly["fin_revenue"],
            name="收入(十亿元)",
            marker=dict(color="rgba(31,111,95,0.82)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=quarterly["报告日"],
            y=quarterly["fin_net_profit"],
            name="归母净利润(十亿元)",
            mode="lines+markers",
            line=dict(color=COLOR_UP, width=2.4),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(title_text="收入(十亿元)", secondary_y=False, gridcolor="rgba(120,120,120,0.15)")
    fig.update_yaxes(title_text="净利润(十亿元)", secondary_y=True, gridcolor="rgba(120,120,120,0.15)")
    return fig


def margin_figure(quarterly: pd.DataFrame) -> go.Figure:
    latest = quarterly.tail(12).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=latest["报告日"],
            y=latest["fin_gross_margin"],
            name="毛利率",
            mode="lines+markers",
            line=dict(color=COLOR_UP, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=latest["报告日"],
            y=latest["fin_net_margin"],
            name="净利率",
            mode="lines+markers",
            line=dict(color=COLOR_DOWN, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=latest["报告日"],
            y=latest["fin_roe"],
            name="ROE",
            mode="lines+markers",
            line=dict(color="#8b6f47", width=2.2, dash="dot"),
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="",
        yaxis_title="百分比",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


def feature_figure(metrics: dict[str, object]) -> go.Figure:
    importance = pd.DataFrame(metrics["classification"]["feature_importance"][:12])
    importance = importance.sort_values("importance", ascending=True)
    importance["direction"] = importance["importance"].apply(lambda value: "正向" if value >= 0 else "负向")
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"正向": COLOR_UP, "负向": COLOR_DOWN},
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend_title_text="",
        xaxis_title="系数/重要性",
        yaxis_title="",
    )
    fig.update_xaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


def candidate_figure(metrics: dict[str, object]) -> go.Figure:
    frame = pd.DataFrame(metrics["classification"]["candidates"])[
        ["name", "accuracy", "balanced_accuracy", "roc_auc"]
    ].rename(
        columns={
            "name": "模型",
            "accuracy": "准确率",
            "balanced_accuracy": "平衡准确率",
            "roc_auc": "ROC AUC",
        }
    )
    melted = frame.melt(id_vars=["模型"], var_name="指标", value_name="数值")
    fig = px.bar(
        melted,
        x="模型",
        y="数值",
        color="指标",
        barmode="group",
        color_discrete_sequence=["#1f6f5f", "#c65a2e", "#8b6f47"],
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        yaxis_title="评分",
        xaxis_title="",
    )
    fig.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    return fig


payload = load_payload()
metrics = payload["metrics"]
prices = payload["prices"]
benchmark = payload["benchmark"]
news = payload["news"]
daily_news = payload["daily_news"]
quarterly = payload["quarterly"]
training_data = payload["training_data"]

generated_at = pd.to_datetime(metrics["config"]["generated_at"])
latest_close = float(metrics["forecast"]["latest_close"])
next_up_probability = float(metrics["forecast"]["next_up_probability"])
predicted_return = float(metrics["forecast"]["predicted_return"])
stock_total_return = float(metrics["descriptive_summary"]["stock_total_return"])
benchmark_total_return = float(metrics["descriptive_summary"]["benchmark_total_return"])
excess_return = float(metrics["descriptive_summary"]["excess_return_vs_benchmark"])
max_drawdown = float(metrics["descriptive_summary"]["max_drawdown"])
recent_20d_return = float(metrics["descriptive_summary"]["latest_20d_return"])
latest_date = prices["date"].max()
news_last_7d = news[news["timestamp"] >= latest_date - pd.Timedelta(days=7)]

st.sidebar.markdown("## 视图控制")
lookback_label = st.sidebar.selectbox("价格窗口", ["3个月", "6个月", "1年", "2年"], index=2)
lookback_days = {"3个月": 90, "6个月": 180, "1年": 365, "2年": 730}[lookback_label]
source_options = [SOURCE_LABELS[key] for key in SOURCE_LABELS]
selected_sources = st.sidebar.multiselect("消息来源", source_options, default=source_options)
selected_impacts = st.sidebar.multiselect("影响方向", ["偏利多", "中性", "偏利空"], default=["偏利多", "中性", "偏利空"])
news_window = st.sidebar.selectbox("新闻窗口", ["最近7天", "最近30天", "最近90天", "全部"], index=1)
news_window_days = {"最近7天": 7, "最近30天": 30, "最近90天": 90, "全部": None}[news_window]
st.sidebar.markdown("---")
if st.sidebar.button("刷新磁盘数据", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
if st.sidebar.button("运行一次全量更新", use_container_width=True):
    with st.spinner("正在重抓价格、新闻、财务并重训模型，约 20 到 40 秒。"):
        run_pipeline(PROJECT_ROOT)
    st.cache_data.clear()
    st.success("更新完成，正在载入最新数据。")
    st.rerun()
st.sidebar.caption("“全量更新”会直接重跑 ETL + ML 管道。部署到云上后，也可以用定时任务每天自动跑。")

filtered_news = news[news["source_label"].isin(selected_sources) & news["impact"].isin(selected_impacts)].copy()
if news_window_days is not None:
    filtered_news = filtered_news[filtered_news["timestamp"] >= latest_date - pd.Timedelta(days=news_window_days)]

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">EASPRING 300073 MONITOR</div>
        <div class="hero-title">当升科技每日跟踪看板</div>
        <p class="hero-subtitle">
            最新数据生成于 <strong>{generated_at.strftime("%Y-%m-%d %H:%M")}</strong>。
            这里把股价、新闻、季度财务和模型信号放到同一页，适合每天直接看。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_row = st.columns(4)
render_metric_card(top_row[0], "最新收盘价", f"{latest_close:.2f}", f"截至 {latest_date:%Y-%m-%d}", metric_tone(recent_20d_return))
render_metric_card(top_row[1], "20日涨跌", format_pct(recent_20d_return), f"两年累计 {format_pct(stock_total_return)}", metric_tone(recent_20d_return))
render_metric_card(top_row[2], "相对创业板", format_pct(excess_return), f"创业板两年 {format_pct(benchmark_total_return)}", metric_tone(excess_return))
render_metric_card(top_row[3], "最大回撤", format_pct(max_drawdown), "风险越低越好", metric_tone(max_drawdown, reverse=True))

second_row = st.columns(4)
render_metric_card(second_row[0], "次日上涨概率", f"{next_up_probability * 100:.1f}%", f"模型标签: {metrics['forecast']['label']}", metric_tone(next_up_probability - 0.5))
render_metric_card(second_row[1], "次日预测收益", format_pct(predicted_return, digits=2), "仅作辅助，不是交易指令", metric_tone(predicted_return))
render_metric_card(second_row[2], "最近7天新消息", str(len(news_last_7d)), f"其中利多 {int((news_last_7d['impact'] == '偏利多').sum())} 条", metric_tone((news_last_7d["impact"] == "偏利多").sum() - (news_last_7d["impact"] == "偏利空").sum()))
render_metric_card(second_row[3], "PDF 正文覆盖", str(metrics["data_summary"]["pdf_extract_ok"]), f"公告 {metrics['data_summary']['pdf_source_counts'].get('disclosure', 0)} / 研报 {metrics['data_summary']['pdf_source_counts'].get('research', 0)}", "neutral")

overview_tab, news_tab, model_tab, fundamentals_tab = st.tabs(["总览", "新闻", "模型", "基本面"])

with overview_tab:
    col_a, col_b = st.columns([1.55, 1.0])
    with col_a:
        st.markdown("### 股价与成交")
        st.caption("上方是收盘价和 20 日均线，下方是成交量。")
        st.plotly_chart(price_figure(prices, lookback_days), use_container_width=True)
    with col_b:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=next_up_probability * 100,
                number={"suffix": "%", "font": {"size": 34}},
                title={"text": "下一交易日上涨概率"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLOR_UP},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(31,111,95,0.22)"},
                        {"range": [40, 60], "color": "rgba(88,96,107,0.18)"},
                        {"range": [60, 100], "color": "rgba(198,90,46,0.18)"},
                    ],
                    "threshold": {"line": {"color": "#1d2320", "width": 4}, "value": 50},
                },
            )
        )
        gauge.update_layout(
            margin=dict(l=8, r=8, t=26, b=10),
            height=250,
            paper_bgcolor="rgba(255,255,255,0.65)",
        )
        st.markdown("### 当日信号")
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f"""
            <div class="panel-caption">
                模型对 <strong>{metrics["forecast"]["next_session_date"]}</strong> 的信号是
                <span class="news-pill">{metrics["forecast"]["label"]}</span>
                预测收益率约 <strong>{format_pct(predicted_return)}</strong>。
            </div>
            """,
            unsafe_allow_html=True,
        )

    row_2a, row_2b = st.columns(2)
    with row_2a:
        st.markdown("### 相对走势")
        st.caption("把当升科技和创业板指都归一到 100，更容易看相对强弱。")
        st.plotly_chart(performance_figure(prices, benchmark, lookback_days), use_container_width=True)
    with row_2b:
        st.markdown("### 月度消息热度")
        st.caption("消息数量不等于确定性，配合情绪均值一起看。")
        st.plotly_chart(monthly_news_figure(news), use_container_width=True)

    st.markdown("### 最近更新")
    st.caption("适合直接看每天新增的消息、来源和影响方向。")
    recent_table = filtered_news[["timestamp", "source_label", "impact", "title", "source_link", "sentiment_score"]].head(20).copy()
    recent_table = recent_table.rename(
        columns={
            "timestamp": "时间",
            "source_label": "来源",
            "impact": "方向",
            "title": "标题",
            "source_link": "链接",
            "sentiment_score": "情绪分数",
        }
    )
    st.dataframe(
        recent_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "链接": st.column_config.LinkColumn("原文", display_text="打开"),
            "情绪分数": st.column_config.NumberColumn(format="%.3f"),
        },
    )

with news_tab:
    col_a, col_b = st.columns([1.05, 1.15])
    with col_a:
        st.markdown("### 来源结构")
        st.caption("不同来源的比例，决定了这段时间信息更偏公告、研报还是媒体。")
        st.plotly_chart(source_mix_figure(news), use_container_width=True)
    with col_b:
        st.markdown("### 过滤后的消息列表")
        st.caption("可以按来源和方向筛选，看适合给家里人直接浏览的版本。")
        news_table = filtered_news[
            ["timestamp", "source_label", "impact", "title", "source_link", "sentiment_score", "body"]
        ].copy()
        news_table["摘要"] = news_table["body"].str.slice(0, 120)
        news_table = news_table.drop(columns=["body"]).rename(
            columns={
                "timestamp": "时间",
                "source_label": "来源",
                "impact": "方向",
                "title": "标题",
                "source_link": "链接",
                "sentiment_score": "情绪分数",
            }
        )
        st.dataframe(
            news_table.head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "链接": st.column_config.LinkColumn("原文", display_text="打开"),
                "情绪分数": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    positive_col, negative_col = st.columns(2)
    positive = news.sort_values("sentiment_score", ascending=False).head(8).copy()
    negative = news.sort_values("sentiment_score", ascending=True).head(8).copy()
    with positive_col:
        st.markdown("### 最强利多事件")
        st.dataframe(
            positive[["timestamp", "source_label", "title", "source_link", "sentiment_score"]].rename(
                columns={
                    "timestamp": "时间",
                    "source_label": "来源",
                    "title": "标题",
                    "source_link": "链接",
                    "sentiment_score": "情绪分数",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "链接": st.column_config.LinkColumn("原文", display_text="打开"),
                "情绪分数": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with negative_col:
        st.markdown("### 最强利空事件")
        st.dataframe(
            negative[["timestamp", "source_label", "title", "source_link", "sentiment_score"]].rename(
                columns={
                    "timestamp": "时间",
                    "source_label": "来源",
                    "title": "标题",
                    "source_link": "链接",
                    "sentiment_score": "情绪分数",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "链接": st.column_config.LinkColumn("原文", display_text="打开"),
                "情绪分数": st.column_config.NumberColumn(format="%.3f"),
            },
        )

with model_tab:
    summary_a, summary_b = st.columns([1.15, 1.0])
    with summary_a:
        st.markdown("### 分类模型表现")
        st.plotly_chart(candidate_figure(metrics), use_container_width=True)
    with summary_b:
        st.markdown("### 当前模型判断")
        st.markdown(
            f"""
            - 选中分类模型: `{metrics["classification"]["selected_model"]}`
            - 测试集准确率: `{metrics["classification"]["selected_metrics"]["accuracy"]:.4f}`
            - 平衡准确率: `{metrics["classification"]["selected_metrics"]["balanced_accuracy"]:.4f}`
            - ROC AUC: `{metrics["classification"]["selected_metrics"]["roc_auc"]:.4f}`
            - 多数方向基线: `{metrics["classification"]["baseline_accuracy"]:.4f}`
            - 回归 R²: `{metrics["regression"]["selected_metrics"]["r2"]:.4f}`
            """
        )
        st.caption("这部分用来看模型是否真的有增益。当前结论仍然是弱信号。")

    st.markdown("### 模型最看重的特征")
    st.plotly_chart(feature_figure(metrics), use_container_width=True)

    relation = daily_news.merge(training_data[["date", "next_return"]], on="date", how="left").dropna(subset=["next_return"]).copy()
    relation["next_return_pct"] = relation["next_return"] * 100
    scatter = px.scatter(
        relation,
        x="news_sentiment_sum",
        y="next_return_pct",
        size="news_count",
        color="news_count",
        color_continuous_scale=["#1f6f5f", "#c65a2e"],
        labels={"news_sentiment_sum": "当日情绪总分", "next_return_pct": "下一日收益率(%)"},
    )
    scatter.update_layout(
        margin=dict(l=8, r=8, t=18, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
    )
    scatter.update_xaxes(gridcolor="rgba(120,120,120,0.15)")
    scatter.update_yaxes(gridcolor="rgba(120,120,120,0.15)")
    st.markdown("### 情绪与次日收益的散点关系")
    st.plotly_chart(scatter, use_container_width=True)

with fundamentals_tab:
    upper, lower = st.columns(2)
    with upper:
        st.markdown("### 收入与净利润")
        st.plotly_chart(fundamentals_figure(quarterly), use_container_width=True)
    with lower:
        st.markdown("### 毛利率 / 净利率 / ROE")
        st.plotly_chart(margin_figure(quarterly), use_container_width=True)

    st.markdown("### 最近财务快照")
    latest_quarters = quarterly.tail(8).copy()
    latest_quarters = latest_quarters.rename(
        columns={
            "报告日": "报告日",
            "公告日期": "公告日",
            "fin_revenue": "收入(十亿元)",
            "fin_net_profit": "净利润(十亿元)",
            "fin_gross_margin": "毛利率",
            "fin_net_margin": "净利率",
            "fin_roe": "ROE",
            "fin_debt_ratio": "资产负债率",
            "fin_revenue_yoy": "收入同比",
            "fin_profit_yoy": "利润同比",
        }
    )
    st.dataframe(
        latest_quarters[
            ["报告日", "公告日", "收入(十亿元)", "净利润(十亿元)", "毛利率", "净利率", "ROE", "资产负债率", "收入同比", "利润同比"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "收入(十亿元)": st.column_config.NumberColumn(format="%.2f"),
            "净利润(十亿元)": st.column_config.NumberColumn(format="%.2f"),
            "毛利率": st.column_config.NumberColumn(format="%.2f"),
            "净利率": st.column_config.NumberColumn(format="%.2f"),
            "ROE": st.column_config.NumberColumn(format="%.2f"),
            "资产负债率": st.column_config.NumberColumn(format="%.2f"),
            "收入同比": st.column_config.NumberColumn(format="%.2f"),
            "利润同比": st.column_config.NumberColumn(format="%.2f"),
        },
    )
