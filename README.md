# 🛒 Retail Price Forecasting — Walmart vs Kroger

An end-to-end data pipeline that scrapes daily grocery prices from Walmart and Kroger across Washington State, forecasts 14-day price trends using Facebook Prophet, and visualizes competitor price gaps on an interactive Streamlit dashboard — all powered by AWS.

> Built as part of the MSDS Capstone Project at Seattle University  
> Analyzing competitor pricing data to support Costco's pricing strategy

---

## 📊 Live Dashboard

```
streamlit run src/dashboard.py
```

---

## 🏗️ Architecture

```
EventBridge (daily trigger)
        ↓
AWS Lambda
├── Walmart Scraper   → walmart/pricing/YYYY-MM-DD/
└── Kroger Scraper    → raw/retailer=kroger/dt=YYYY-MM-DD/
        ↓
Amazon S3 (raw pricing CSVs + gz files)
        ↓
data_loader.py → cleans + aggregates daily avg prices
        ↓
forecast_model.py → Facebook Prophet (one model per retailer × product)
        ↓
S3 (forecasts/YYYY-MM-DD/price_forecasts.csv)
        ↓
Streamlit Dashboard (actual vs forecast + Walmart vs Kroger price gap)
```

---

## 🔍 How Data is Scraped

### Walmart

Walmart uses the **Walmart Affiliate API** with cryptographic RSA authentication.
Every request generates a fresh digital signature using a private `.pem` key —
the Consumer ID + timestamp is signed with SHA256 and sent as a base64-encoded header.

Scraping runs in two steps:
1. **Search** — queries like `"large eggs 12 count"` collect up to 30 item IDs
2. **Lookup** — each item ID is looked up per location zip code to get the local price

Each product category runs against a fixed set of Washington State locations:

| Category | Locations | Zip Codes |
|---|---|---|
| Eggs | 6 locations | Tukwila, Tacoma, Everett, Puyallup, Aurora Village, Kirkland |
| Olive Oil | 8 locations | Ridgefield, Lacey, Burlington, Lake Stevens, Kirkland, E Vancouver, Lynnwood, Redmond |
| Paper Towels | 10 locations | Seattle, Issaquah, Tukwila, Woodinville, Kirkland, Covington, Lynnwood, Aurora Village, Redmond + 1 |

Results saved to: `s3://walmart-scraper-data/walmart/pricing/YYYY-MM-DD/category_YYYY-MM-DD.csv`

---

### Kroger

Kroger uses an official **OAuth2 Developer API** at `developer.kroger.com`.
A `KrogerAuth` class handles token lifecycle — fetching, tracking expiry,
and auto-refreshing 60 seconds before the token expires so no requests fail mid-run.

Key design decision: store locations are loaded from a **pre-saved S3 file** instead
of calling the Locations API on every run. The file was generated once by paginating
through all Washington State Kroger stores and saved as `.csv.gz` — this eliminated
significant overhead from every daily run.

For each of the **57 WA store locations**, the scraper queries the Products API
with the search term and collects up to 3 pages × 50 products = 150 results per store.
The `effective_price` column is automatically the lower of `regular_price` vs `promo_price`.

Results saved to: `s3://walmart-scraper-data/raw/retailer=kroger/dt=YYYY-MM-DD/term=eggs/kroger_wa_eggs_TIMESTAMP.csv.gz`

---

### Why Kroger Has More Rows

| | Walmart | Kroger |
|---|---|---|
| Locations scraped | 6–10 per category | 57 statewide |
| Products per location | up to 30 | up to 150 |
| Daily rows | ~700 avg | ~50,000 avg |
| Total collected | 53,302 rows | 3,859,952 rows |
| Auth method | RSA signed headers | OAuth2 client credentials |
| Price column | `sale_price` | `effective_price` |
| Date column | `run_date` | `snapshot_utc` |
| Storage format | `.csv` | `.csv.gz` |
| Start date | March 1, 2026 | February 17, 2026 |

---

## 📦 Products Tracked

| Product | Retailers | History |
|---|---|---|
| Eggs | Walmart, Kroger | Mar 2026 → present |
| Olive Oil | Walmart, Kroger | Mar 2026 → present |
| Paper Towels | Walmart, Kroger | Mar 2026 → present |

---

## 📈 Model Performance

All models trained using Facebook Prophet on 74–76 days of daily average prices.

| Retailer | Product | MAE | MAPE |
|---|---|---|---|
| Kroger | Eggs | $0.03 | 0.7% |
| Kroger | Olive Oil | $0.06 | 0.5% |
| Kroger | Paper Towels | $0.04 | 0.3% |
| Walmart | Eggs | $0.08 | 1.7% |
| Walmart | Olive Oil | $0.45 | 3.8% |
| Walmart | Paper Towels | $0.49 | 2.8% |

> Price caps were applied to Walmart data to filter out bulk/commercial SKUs
> (eggs > $15, olive oil > $25, paper towels > $30) before training.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Cloud Infrastructure | AWS Lambda, Amazon S3, EventBridge |
| Walmart Data Collection | Walmart Affiliate API, RSA cryptographic auth |
| Kroger Data Collection | Kroger Developer API, OAuth2 |
| Forecasting Model | Facebook Prophet |
| Dashboard | Streamlit, Plotly |
| Language | Python 3.13 |

---

## 🗂️ Project Structure

```
retail-price-forecasting/
│
├── data/
│   └── sample/                    ← sample CSVs for local testing
│       ├── eggs_2026-03-01.csv
│       ├── olive_oil_2026-03-01.csv
│       ├── paper_towel_2026-03-01.csv
│       └── kroger_wa_*.csv.gz
│
├── src/
│   ├── data_loader.py             ← reads S3, cleans + aggregates daily prices
│   ├── forecast_model.py          ← trains Prophet, saves forecasts to S3
│   └── dashboard.py               ← Streamlit dashboard
│
├── screenshots/                   ← dashboard screenshots
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/retail-price-forecasting.git
cd retail-price-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set AWS credentials**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-2
```

**4. Run the forecast model**
```bash
python src/forecast_model.py
```

**5. Launch the dashboard**
```bash
streamlit run src/dashboard.py
```

---

## 🔍 Key Findings

- Walmart egg prices fluctuate more day-to-day than Kroger (~±$0.30 vs ±$0.08)
- Kroger olive oil prices are consistently ~$3.50 lower than Walmart
- Prophet achieves under 4% MAPE across all 6 retailer × product models
- Price gaps between Walmart and Kroger shift weekly, suggesting promotional pricing activity
- Kroger data is significantly cleaner — structured API vs web scraping means fewer outliers

---

## 👩‍💻 Built By

**Fariha Shah** — MSDS Student, Seattle University  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

> Capstone Project — Division of Data Science, Seattle University, 2026
