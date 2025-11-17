import os
import cv2
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, render_template, url_for

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load Your Saved Model and Scaler ---
# These files must be in the same directory as this app.py
try:
    model = joblib.load('rice_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("FATAL ERROR: 'rice_model.joblib' or 'scaler.joblib' not found.")
    print("Please run the 'train.py' script first to create these files.")
    # We can't run the app without the models.
    exit()

# --- Feature Extraction Function ---
def extract_features_from_image(image):
    """
    Analyzes an uploaded image to extract the required features.
    This is the same logic from your Streamlit app.
    """
    try:
        # Convert PIL Image to OpenCV format
        img_cv = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Binarize the image. This threshold (100) is crucial.
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Find Contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, "No contour detected in image."

        # Assume the largest contour is the rice grain
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = contours[0]
        
        # --- Calculate Features ---
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Avoid division by zero if height is 0
        aspect_ratio = float(w) / h if h > 0 else 0.0
        
        # Create a mask to calculate the mean color of only the grain
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [main_contour], 0, 255, -1)
        mean_color_values = cv2.mean(img_cv, mask=mask)
        mean_color = np.mean(mean_color_values[:3]) 
        
        features = [area, perimeter, aspect_ratio, mean_color]
        return features, None

    except Exception as e:
        return None, f"Error during feature extraction: {e}"

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main homepage (index.html)"""
    # This is the "render" function you were thinking of!
    return render_template('index.html', prediction_result=None, error_msg=None)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image upload, processing, and prediction.
    """
    if 'image' not in request.files:
        return render_template('index.html', prediction_result=None, error_msg="No file part. Please upload an image.")
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', prediction_result=None, error_msg="No image selected. Please upload an image.")
    
    if file:
        try:
            # Open the image using PIL
            image = Image.open(file.stream)
            
            # 1. Extract features
            features, error = extract_features_from_image(image)
            if error:
                return render_template('index.html', prediction_result=None, error_msg=error)

            # 2. Scale features
            features_array = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(features_array)
            
            # 3. Predict
            prediction = model.predict(scaled_features)
            prediction_proba = model.predict_proba(scaled_features)
            
            # Create a confidence dictionary
            confidence = dict(zip(model.classes_, prediction_proba[0]))
            
            result = {
                "prediction": prediction[0],
                "confidence": confidence
            }
            
            # Success! Render the page again, but this time with the result.
            return render_template('index.html', prediction_result=result, error_msg=None)

        except Exception as e:
            return render_template('index.html', prediction_result=None, error_msg=f"An error occurred: {e}")

# --- Run the App ---
if __name__ == '__main__':
    # Create a 'templates' folder in the same directory as this script.
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' folder. Please place 'index.html' there.")
        
    app.run(debug=True)