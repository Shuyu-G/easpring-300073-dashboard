# Easpring 300073 Stock/News Pipeline

This project follows the `lecture6` ETL + ML assignment pattern, but applies it to `北京当升材料科技股份有限公司 (300073)`.

## What it does

- Fetches two years of A-share daily prices for `300073`
- Fetches official CNInfo disclosures, Eastmoney research reports, and recent stock news
- Fetches monthly Google News RSS archive results for broader media coverage
- Decodes Google News links and extracts article text when possible
- Extracts PDF body text from material CNInfo disclosures and sell-side research reports
- Adds quarterly financial indicators that become effective on report announcement dates
- Computes simple Chinese sentiment features
- Trains:
  - a direction classifier for next-day up/down
  - a return regressor for next-day return
- Writes a local analysis report and an Airflow DAG

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.easpring_pipeline --project-root .
```

## Dashboard

Run the local dashboard:

```bash
.venv/bin/streamlit run dashboard_app.py
```

Build the free public static site:

```bash
.venv/bin/python -m src.dashboard_snapshot --project-root . --output site/index.html
```

This writes a multi-file static site under:

- `site/index.html`
- `site/news.html`
- `site/model.html`
- `site/fundamentals.html`
- `site/data/dashboard.json`
- `site/assets/`

The `--output` argument still points to the entry HTML path, but the generator writes the full site into its parent directory.

The dashboard reads the latest files from:

- `data/raw/`
- `data/processed/`
- `reports/metrics.json`
- `models/`

Inside the dashboard sidebar you can:

- reload data from disk
- trigger one full ETL + ML refresh
- filter the news feed by source and impact direction

## Daily Refresh

To refresh the data manually:

```bash
.venv/bin/python -m src.easpring_pipeline --project-root .
```

If you want the dashboard to stay fresh every day, schedule the command above with a local scheduler such as `cron`, `launchd`, or an always-on cloud VM.

## Free Public Deployment With GitHub Pages

This repo now includes a GitHub Pages workflow:

- `.github/workflows/deploy-pages.yml`

What it does:

- installs dependencies
- refreshes prices, news, PDF text, and models
- generates a multi-page static dashboard under `site/`
- deploys that file to GitHub Pages

Schedule:

- weekdays at `01:15 UTC`
- which is `09:15` in China Standard Time

To enable it:

1. Push this project to a GitHub repository.
2. Make the repository public if you want to stay on GitHub's free public-runner tier.
3. In GitHub repository settings, open `Pages`.
4. Set `Build and deployment` to `GitHub Actions`.
5. Run the `Deploy Dashboard To GitHub Pages` workflow once from the `Actions` tab.

Important caveat:

- The workflow depends on third-party Chinese market data endpoints via `AKShare`.
- If GitHub-hosted runners cannot reliably fetch those endpoints from their region, scheduled refreshes may fail.
- If that happens, the fallback is to run the pipeline locally and push the refreshed snapshot to GitHub.

## Test saved models

```bash
.venv/bin/python test_model.py --data data/processed/training_data.csv
```

## Outputs

- `data/raw/`
- `data/raw/google_news_archive.csv`
- `data/raw/financial_abstract.csv`
- `data/raw/financial_report.csv`
- `data/raw/pdf_text_cache.csv`
- `data/processed/`
- `data/processed/quarterly_financials.csv`
- `models/`
- `reports/analysis.md`
- `reports/metrics.json`

## Airflow

The DAG file is in `dags/easpring_300073_ml_dag.py`.

Workflow:

```text
fetch_stock_prices + fetch_company_news
                    -> build_training_dataset
                    -> train_models
                    -> write_report
```

## Container

Build and run the dashboard container locally:

```bash
docker build -t easpring-dashboard .
docker run --rm -p 8501:8501 easpring-dashboard
```

Notes:

- The current project stores refreshed data on the local filesystem.
- For an always-on AWS deployment with daily in-place refresh, a VM-style target such as Lightsail or EC2 is a better fit than a stateless container service.
