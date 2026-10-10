# Seed Demand Forecasting Demo

A public, privacy-safe Streamlit recreation of a graduate capstone forecasting project.
All data is synthetic (2024-2026) — no real client data is used.

## Files
- `app.py` — the Streamlit app
- `xgb_model.pkl`, `product_encoder.pkl`, `product_list.pkl` — trained model artifacts
- `synthetic_demo_data.csv` — synthetic monthly demand data (10 generic seed products)
- `train_monthly_model.py` — script that trains the model from the synthetic data (re-run if you regenerate data)
- `requirements.txt` — dependencies

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repo (e.g. a new repo or a `streamlit-demo/` folder in your existing portfolio repo).
2. Go to https://share.streamlit.io, sign in with GitHub, and click "New app".
3. Point it at your repo, branch, and `app.py` as the main file.
4. Deploy — you'll get a public URL like `https://<your-app>.streamlit.app`.
5. Paste that URL into the portfolio's "Interactive Sales Forecasting Tool" card (replace the `href="#"` in index.html).
