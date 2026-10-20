# SuperKart Sales Forecasting – Model Deployment

An end-to-end machine learning project that predicts **product-level sales revenue** (`Product_Store_Sales_Total`) for SuperKart, a retail chain of supermarkets and food marts. The trained model is served through a **Flask REST API** and a **Streamlit web app**, each running in its own Docker container.

## Business Goal

SuperKart wants accurate sales forecasts for the upcoming quarter to support **inventory management, procurement and regional sales strategy**. The model estimates how much revenue a given product will generate in a given store, so managers can plan stock levels and reduce both overstocking and stock-outs.

## Approach

**Data audit**
- 8,763 records, 12 columns; no duplicate rows and no missing values.
- Outliers detected with the IQR rule (~1% per column) and intentionally kept — they are genuine values (e.g., premium products).
- Inconsistent label `reg` merged into `Regular`.

**Feature engineering**
- `Product_Id_char` – product family from the ID prefix (FD = food, DR = drinks, NC = non-consumables)
- `Store_Age_Years` – store age instead of establishment year
- `Product_Type_Category` – Perishables vs. Non Perishables

**Modeling**
- Preprocessing pipeline: `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- Models: Random Forest and XGBoost, baseline and tuned with `GridSearchCV`
- **Primary metric: RMSE** — in the same units as revenue and penalizes large forecasting misses, which are the most costly for inventory planning

## Results

| Model | Train RMSE | Test RMSE |
|---|---|---|
| Random Forest (baseline) | 107 | 285 |
| **Random Forest (tuned)** | **198** | **279** |
| XGBoost (baseline) | 131 | 310 |
| XGBoost (tuned) | 216 | 284 |

**Final model: Tuned Random Forest** — lowest test RMSE (≈279 vs. average sales of ≈3,450), R² ≈ 0.93, MAPE ≈ 4%. Tuning reduced overfitting (smaller train–test gap) and improved test performance.

## Key Business Insights

- `Product_MRP` and `Product_Weight` are the strongest drivers of sales.
- Departmental Stores and Supermarket Type1 have the highest sales per product; Food Marts the lowest.
- Tier 1 cities have the highest median sales, followed by Tier 2 and Tier 3.
- Fruits & Vegetables and Snack Foods generate the most total revenue — keep them consistently stocked.
- Shelf space (`Product_Allocated_Area`) shows almost no relationship with sales, so space should be allocated by turnover and margin.

## Repository Structure

```
backend/
    app.py                  # Flask API serving the model
    requirements.txt
    Dockerfile
    superkart_model.joblib  # Serialized pipeline (preprocessing + tuned Random Forest)
frontend/
    app.py                  # Streamlit UI – calls the Flask API
    requirements.txt
    Dockerfile
```

## API Endpoints (port 7860)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check – returns a welcome message |
| POST | `/v1/predict` | Single prediction from a JSON payload; returns `{"Sales": <value>}` |
| POST | `/v1/predictbatch` | Batch prediction from an uploaded CSV file; returns `{row_index: prediction}` |

**Example request body for `/v1/predict`:**

```json
{
  "Product_Weight": 12.66,
  "Product_Sugar_Content": "Low Sugar",
  "Product_Allocated_Area": 0.027,
  "Product_MRP": 117.08,
  "Store_Size": "Medium",
  "Store_Location_City_Type": "Tier 2",
  "Store_Type": "Supermarket Type2",
  "Product_Id_char": "FD",
  "Store_Age_Years": 16,
  "Product_Type_Category": "Non Perishables"
}
```

Batch CSV files must contain the same 10 columns.

## Streamlit App (port 8501)

- **Single prediction:** enter product and store details and click *Predict Sales*.
- **Batch prediction:** upload a CSV, click *Predict for Batch*, view the predictions table and download the results as a CSV.

All predictions are sent to the Flask API — the frontend does not load the model itself.

## Running in GitHub Codespaces

```bash
# Build the images
cd backend && docker build -t superkart-backend .
cd ../frontend && docker build -t superkart-frontend .

# Run both containers on the Codespace host network
docker run -d --name backend --network host superkart-backend
docker run -d --name frontend --network host -e BACKEND_URL=http://localhost:7860 superkart-frontend
```

Then, in the **Ports** tab, add ports **7860** (API) and **8501** (Streamlit app) if they are not listed, and set both to **Public**.

To restart after the Codespace has stopped:

```bash
docker start backend frontend
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · joblib · Flask · Streamlit · Docker · GitHub Codespaces
