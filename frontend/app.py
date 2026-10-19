
import os
import requests
import pandas as pd
import streamlit as st

# Base URL of the deployed Flask backend (container name 'backend' on the Docker network)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860").rstrip("/")

st.title("SuperKart Sales Prediction App")
st.write("This app predicts the total sales revenue (`Product_Store_Sales_Total`) of a product in a particular store "
         "by calling the deployed SuperKart prediction API.")

# ---------------- Online (single) prediction ----------------
st.header("Single Prediction")

Product_Weight = st.number_input("Product Weight", min_value=0.0, max_value=30.0, value=12.66, step=0.1)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
Product_MRP = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=117.08, step=1.0)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product Family (Product ID prefix)", ["FD", "DR", "NC"],
                               help="FD = Food, DR = Drinks, NC = Non-consumables")
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16, step=1)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

if st.button("Predict Sales", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Estimated Product Store Sales Total: {predicted_sales:,.2f}")
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API at {BACKEND_URL}: {e}")

# ---------------- Batch prediction ----------------
st.header("Batch Prediction")
st.write("Upload a CSV file with the columns: " + ", ".join(product_data.keys()))

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded {len(input_df)} rows. Preview:")
    st.dataframe(input_df.head())

    if st.button("Predict for Batch", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=120,
            )
            if response.status_code == 200:
                predictions = response.json()
                results_df = input_df.copy()
                results_df["Predicted_Sales"] = [predictions[str(i)] for i in range(len(results_df))]

                st.success(f"Predictions completed for {len(results_df)} rows!")
                st.dataframe(results_df)
                st.download_button(
                    "Download predictions as CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"API error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the prediction API at {BACKEND_URL}: {e}")
