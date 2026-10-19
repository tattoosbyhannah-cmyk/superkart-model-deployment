
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model pipeline (preprocessing + Random Forest) from the same directory
model = joblib.load('superkart_model.joblib')

# Features expected by the model (must match the training columns)
FEATURES = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type',
    'Product_Id_char', 'Store_Age_Years', 'Product_Type_Category'
]

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to the SuperKart Sales Prediction API'

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        # Keep only the expected features, in the expected order
        df = pd.DataFrame([data])[FEATURES]
        # Generate prediction using the loaded model
        prediction = model.predict(df)
        # Return the prediction as a JSON response
        return jsonify({'Sales': round(float(prediction[0]), 2)})
    except KeyError as e:
        return jsonify({'error': f'Missing feature(s): {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        # Check if a file was selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Drop the target column if it exists in the uploaded data
        if 'Product_Store_Sales_Total' in df.columns:
            df = df.drop(columns=['Product_Store_Sales_Total'])

        # Generate predictions for the batch
        predictions = model.predict(df[FEATURES])

        # Format predictions as a dictionary {row index: prediction} and return as JSON
        return jsonify({str(i): round(float(pred), 2) for i, pred in enumerate(predictions)})

    except KeyError as e:
        return jsonify({'error': f'Missing column(s) in CSV: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app on all available IPs, listening on port 7860
    app.run(host='0.0.0.0', port=7860)
