
import streamlit as st
import pandas as pd
import joblib

# Load the trained regression model
@st.cache_resource
def load_model():
    return joblib.load("superkart_model.joblib")

model = load_model()

# Streamlit UI for SuperKart Sales Prediction
st.title("SuperKart Sales Prediction App")
st.write("This app predicts the total sales revenue (`Product_Store_Sales_Total`) of a product in a particular store.")
st.write("Adjust the values below to get a prediction.")

# Collect numerical inputs using sliders
Product_Weight = st.slider("Product Weight", 4.0, 22.0, 12.6, 0.1)
Product_Allocated_Area = st.slider("Product Allocated Area (ratio)", 0.0, 0.3, 0.06, 0.01)
Product_MRP = st.slider("Product MRP", 31.0, 266.0, 147.0, 1.0)
Store_Establishment_Year = st.slider("Store Establishment Year", 1987, 2009, 1999, 1)

# Collect categorical inputs using text input and selectboxes
Product_Id = st.text_input("Product ID", "FD6114")
Product_Sugar_Content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar', 'reg'])
Product_Type = st.selectbox("Product Type", [
    'Fruits and Vegetables', 'Snack Foods', 'Frozen Foods', 'Dairy', 
    'Household', 'Baking Goods', 'Canned', 'Health and Hygiene', 
    'Meat', 'Soft Drinks', 'Breads', 'Hard Drinks', 'Others', 
    'Starchy Foods', 'Breakfast', 'Seafood'
])
Store_Id = st.selectbox("Store ID", ['OUT001', 'OUT002', 'OUT003', 'OUT004', 'OUT027', 'OUT013', 'OUT049', 'OUT046', 'OUT035', 'OUT010'])
Store_Size = st.selectbox("Store Size", ['Small', 'Medium', 'High'])
Store_Location_City_Type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
Store_Type = st.selectbox("Store Type", ['Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'])

# Create input DataFrame
input_data = pd.DataFrame([{
    'Product_Id': Product_Id,
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Id': Store_Id,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Predict button
if st.button("Predict Sales"):
    try:
        predicted_sales = model.predict(input_data)[0]
        st.success(f"💰 Estimated Product Store Sales Total: ${predicted_sales:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
