import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib # Import joblib to save your model
import os # Added os for checking file

# --- Load Your Data ---
# Make sure 'rice_data.csv' is the correct path to your data
data_file = 'rice_data.csv' # Define your dataset name here

if not os.path.exists(data_file):
    print(f"Error: '{data_file}' not found.")
    print("Please make sure your dataset is in the same folder as train.py")
    exit()

try:
    df = pd.read_csv(data_file) 
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Your Existing Code ---
# Update these to match your CSV's column names if they are different
features = ['area(px)', 'perimeter(px)', 'aspect_ratio', 'mean_color']
target = 'Name_of_rice'

# Check if features and target exist
if not all(f in df.columns for f in features):
    print(f"Error: One or more features not in CSV columns. Need: {features}")
    exit()
if target not in df.columns:
    print(f"Error: Target column '{target}' not in CSV.")
    exit()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- Evaluate the Model (Good to check) ---
y_pred = model.predict(X_test_scaled)
print("Model Training Complete. Evaluation metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- NEW: Save the Model and Scaler ---
model_filename = 'rice_model.pkl'
scaler_filename = 'scaler.pkl'

print(f"\nSaving model to '{model_filename}'...")
joblib.dump(model, model_filename)

print(f"Saving scaler to '{scaler_filename}'...")
joblib.dump(scaler, scaler_filename)

print("\nTraining complete. You can now run the 'flask_app.py' file.")