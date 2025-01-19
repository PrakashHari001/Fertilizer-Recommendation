from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load dataset and preprocess
file_path = r"C:\Users\harip\Desktop\MP2\fertilizer_recommendation_dataset.csv"
data = pd.read_csv(file_path)

# Label encode categorical columns
label_encoder_crop = LabelEncoder()
data['Crop Name'] = label_encoder_crop.fit_transform(data['Crop Name'])

label_encoder_fertilizer = LabelEncoder()
data['Recommended Fertilizer'] = label_encoder_fertilizer.fit_transform(data['Recommended Fertilizer'])

# Features and target separation
X = data.drop(columns=['Recommended Fertilizer'])
y = data['Recommended Fertilizer']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)


# Define recommendation logic
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse JSON data
        data = request.get_json()  # Expect JSON format

        # Extract input fields
        crop_name = data['crop_name']
        nitrogen = int(data['nitrogen'])
        phosphorus = int(data['phosphorus'])
        potassium = int(data['potassium'])
        pH = float(data['ph'])
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])
        humidity = float(data['humidity'])
        area_sqft = float(data['area_sqft'])

        # Validate input ranges
        if not (0 <= nitrogen <= 100 and 0 <= phosphorus <= 100 and 0 <= potassium <= 100):
            return jsonify({'error': 'Nutrient levels (N, P, K) should be between 0 and 100.'})
        if not (0 <= pH <= 14):
            return jsonify({'error': 'pH level should be between 0 and 14.'})
        if not (0 <= temperature <= 50):
            return jsonify({'error': 'Temperature should be between 0 and 50°C.'})
        if not (0 <= rainfall <= 1000):
            return jsonify({'error': 'Rainfall should be between 0 and 1000 mm.'})
        if not (0 <= humidity <= 100):
            return jsonify({'error': 'Humidity should be between 0 and 100%.'})

        # Encode crop name
        if crop_name not in label_encoder_crop.classes_:
            return jsonify({'error': 'Crop name not recognized. Please enter a valid crop name.'})
        crop_encoded = label_encoder_crop.transform([crop_name])[0]

        # Scale input features
        input_features = scaler.transform([[crop_encoded, nitrogen, phosphorus, potassium, pH, temperature, rainfall, humidity]])
        
        # Predict fertilizer
        fertilizer_encoded = model.predict(input_features)[0]
        fertilizer_name = label_encoder_fertilizer.inverse_transform([fertilizer_encoded])[0]
        
        # Calculate quantity
        base_quantity_per_sqft = 0.05  # Base rate, in kg per square foot
        nutrient_deficiency_factor = (100 - nitrogen + 100 - phosphorus + 100 - potassium) / 300
        quantity = base_quantity_per_sqft * (1 + nutrient_deficiency_factor) * area_sqft

        # Return recommendations
        return jsonify({
            'fertilizer': fertilizer_name,
            'quantity': f"{quantity:.2f} kg",
            'message': "Ensure balanced nutrient application for optimal growth based on current soil conditions."
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# Render the home page
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
