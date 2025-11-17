import os
import cv2
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, render_template, url_for, flash, redirect
import psycopg2 # --- Postgres Import ---
from urllib.parse import urlparse

# --- App Setup ---
app = Flask(__name__, template_folder='templates')
# You must set a secret key for "flash" messages to work
app.secret_key = 'your_super_secret_key_12345' 

# --- Database Connection ---
# This reads the secret variables you set on Render
DATABASE_URL = os.environ.get('DATABASE_URL')
SSL_MODE = os.environ.get('SSL_MODE', 'prefer') # Get SSL_MODE, default to 'prefer'

def get_db_connection():
    """Helper function to connect to the Postgres database."""
    if not DATABASE_URL:
        print("DATABASE_URL not set. DB features will be disabled.")
        return None
    try:
        # Use the SSL_MODE from the environment
        conn = psycopg2.connect(DATABASE_URL, sslmode=SSL_MODE)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def create_corrections_table():
    """Run this once on startup to make sure our table exists."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS corrections (
                        id SERIAL PRIMARY KEY,
                        feature_string TEXT UNIQUE NOT NULL,
                        correct_label TEXT NOT NULL
                    );
                """)
                conn.commit()
                print("Successfully connected to DB and verified 'corrections' table.")
        except Exception as e:
            print(f"Error creating table: {e}")
        finally:
            conn.close()
    else:
        print("Could not connect to DB to create table.")

def check_db_for_correction(feature_string):
    """Checks the database for a user-submitted correction."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT correct_label FROM corrections WHERE feature_string = %s", (feature_string,))
                result = cur.fetchone()
            if result:
                return result[0] # Return the correct label (e.g., "Basmati")
        except Exception as e:
            print(f"Error checking DB for correction: {e}")
        finally:
            conn.close()
    return None

def save_correction_to_db(feature_string, correct_label):
    """Saves a new correction. If one exists, it updates it."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO corrections (feature_string, correct_label)
                    VALUES (%s, %s)
                    ON CONFLICT (feature_string) 
                    DO UPDATE SET correct_label = EXCLUDED.correct_label;
                """, (feature_string, correct_label))
                conn.commit()
                print(f"Successfully saved correction: {feature_string} -> {correct_label}")
        except Exception as e:
            print(f"Error saving correction to DB: {e}")
        finally:
            conn.close()
    else:
        print(f"Could not save correction. DB connection failed.")

# --- Load Models (Unchanged) ---
try:
    model = joblib.load('rice_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("FATAL ERROR: 'rice_model.joblib' or 'scaler.joblib' not found.")
    print("Please run 'train.py' first and make sure files are pushed to GitHub.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model, scaler = None, None

# --- Feature Extraction (Unchanged) ---
def extract_features_from_image(image):
    try:
        img_cv = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, "No contour detected in image."
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = contours[0]
        
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [main_contour], 0, 255, -1)
        mean_color_values = cv2.mean(img_cv, mask=mask)
        mean_color = np.mean(mean_color_values[:3])
        
        # Round the features to be consistent
        features = [round(area), round(perimeter, 2), round(aspect_ratio, 3), round(mean_color, 2)]
        return features, None
    except Exception as e:
        return None, f"Error during feature extraction: {e}"

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main homepage (index.html)"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, checks database for corrections, 
    and then uses the model if no correction is found.
    """
    if not model or not scaler:
        flash("Server Error: Model is not loaded. Please check server logs.", "error")
        return redirect(url_for('home'))
        
    if 'image' not in request.files:
        flash("No file part. Please upload an image.", "error")
        return redirect(url_for('home'))
    
    file = request.files['image']
    if file.filename == '':
        flash("No image selected. Please upload an image.", "error")
        return redirect(url_for('home'))
    
    if file:
        try:
            image = Image.open(file.stream)
            features, error = extract_features_from_image(image)
            if error:
                flash(error, "error")
                return redirect(url_for('home'))
            
            # This string is the unique ID for the database
            features_string = ",".join(map(str, features))

            # 1. HERE IS THE "VETO" CHECK!
            corrected_prediction = check_db_for_correction(features_string)
            
            if corrected_prediction:
                # 2A. A CORRECTION WAS FOUND!
                print(f"Veto system: Found correction for {features_string}. Using '{corrected_prediction}'")
                prediction_result = {
                    "prediction": corrected_prediction,
                    "confidence": "N/A (User-corrected)",
                    "was_corrected": True
                }
            else:
                # 2B. NO CORRECTION WAS FOUND.
                print(f"No correction found for {features_string}. Using model.")
                features_array = np.array(features).reshape(1, -1)
                scaled_features = scaler.transform(features_array)
                prediction = model.predict(scaled_features)
                prediction_proba = model.predict_proba(scaled_features)
                confidence = dict(zip(model.classes_, prediction_proba[0]))
                
                prediction_result = {
                    "prediction": prediction[0],
                    "confidence": confidence,
                    "was_corrected": False
                }

            # Render the page with the results
            return render_template('index.html', 
                                   prediction_result=prediction_result, 
                                   all_labels=model.classes_,
                                   features_string=features_string)

        except Exception as e:
            flash(f"An error occurred: {e}", "error")
            return redirect(url_for('home'))

@app.route('/report_error', methods=['POST'])
def report_error():
    """Saves the user's correction to the permanent Neon database."""
    try:
        correct_label = request.form['correct_label']
        features_string = request.form['features']
        
        # This is the line that saves the correction
        save_correction_to_db(features_string, correct_label) 
        
        flash(f"Thank you! The correction '{correct_label}' has been saved.", "thank-you")
        
    except Exception as e:
        flash(f"Error logging correction: {e}", "error")

    return redirect(url_for('home'))

# --- Run the App ---
if __name__ == '__main__':
    # This runs when you type 'python flask_app.py'
    if not DATABASE_URL:
        print("-----------------------------------------------")
        print("WARNING: 'DATABASE_URL' is not set.")
        print("The app will run locally, but corrections will not be saved.")
        print("-----------------------------------------------")
    
    # Try to create the table on local startup
    create_corrections_table() 
    
    app.run(debug=True)
else:
    # This runs when Gunicorn starts the app on Render
    if DATABASE_URL:
        print("Gunicorn starting... creating table if it doesn't exist.")
        create_corrections_table() # Make sure our table exists