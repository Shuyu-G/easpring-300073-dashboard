from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from airflow import DAG

try:
    from airflow.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python_operator import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.easpring_pipeline import (  # noqa: E402
    PipelineConfig,
    build_training_dataset,
    fetch_company_news,
    fetch_financial_data,
    fetch_pdf_text_cache,
    fetch_stock_prices,
    make_paths,
    train_models,
    write_report,
)

CONFIG = PipelineConfig(project_root=PROJECT_ROOT)
PATHS = make_paths(PROJECT_ROOT)


def _fetch_prices() -> None:
    fetch_stock_prices(CONFIG, PATHS)


def _fetch_news() -> None:
    fetch_company_news(CONFIG, PATHS)


def _fetch_financials() -> None:
    fetch_financial_data(CONFIG, PATHS)


def _fetch_pdf_texts() -> None:
    fetch_pdf_text_cache(PATHS)


def _build_dataset() -> None:
    build_training_dataset(CONFIG, PATHS)


def _train_models() -> None:
    train_models(CONFIG, PATHS)


def _write_report() -> None:
    write_report(CONFIG, PATHS)


dag = DAG(
    dag_id="easpring_300073_stock_news_ml",
    start_date=datetime(2026, 3, 1),
    schedule="@weekly",
    catchup=False,
    tags=["stocks", "news", "ml", "etl", "300073"],
)

fetch_prices_task = PythonOperator(
    task_id="fetch_stock_prices",
    python_callable=_fetch_prices,
    dag=dag,
)

fetch_news_task = PythonOperator(
    task_id="fetch_company_news",
    python_callable=_fetch_news,
    dag=dag,
)

fetch_financials_task = PythonOperator(
    task_id="fetch_financial_data",
    python_callable=_fetch_financials,
    dag=dag,
)

fetch_pdf_texts_task = PythonOperator(
    task_id="fetch_pdf_text_cache",
    python_callable=_fetch_pdf_texts,
    dag=dag,
)

build_dataset_task = PythonOperator(
    task_id="build_training_dataset",
    python_callable=_build_dataset,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id="train_models",
    python_callable=_train_models,
    dag=dag,
)

write_report_task = PythonOperator(
    task_id="write_report",
    python_callable=_write_report,
    dag=dag,
)

[fetch_prices_task, fetch_news_task] >> fetch_pdf_texts_task
[fetch_pdf_texts_task, fetch_financials_task] >> build_dataset_task >> train_models_task >> write_report_task
