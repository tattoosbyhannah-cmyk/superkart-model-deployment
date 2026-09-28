
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model from the same directory
model = joblib.load('superkart_model.joblib')

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        # Convert JSON payload into a pandas DataFrame
        df = pd.DataFrame([data])
        # Generate prediction using the loaded model
        prediction = model.predict(df)
        # Return the prediction as a JSON response
        return jsonify({'prediction': float(prediction[0])})
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

        # Drop the target column if it exists in the test data
        if 'Product_Store_Sales_Total' in df.columns:
            df = df.drop(columns=['Product_Store_Sales_Total'])

        # Generate predictions for the batch
        predictions = model.predict(df)

        # Format predictions as a dictionary {index: prediction} and return as JSON
        return jsonify({str(i): float(pred) for i, pred in enumerate(predictions)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app on all available IPs, listening on port 7860
    app.run(host='0.0.0.0', port=7860)
