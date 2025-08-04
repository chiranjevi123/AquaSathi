from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sqlite3
import os

# Absolute path setup
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

# --- Database Setup ---
DB_PATH = os.path.join(basedir, 'water_quality.db')

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY,
                color REAL,
                turbidity REAL,
                ph REAL,
                label TEXT,
                latitude REAL,
                longitude REAL
            )
        ''')

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Load ML Model ---
MODEL_PATH = os.path.join(basedir, 'model.pkl')
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    model = None

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if request.method == "POST" and model:
        try:
            print("📥 Received POST data:", request.form)

            color = safe_float(request.form.get("color"))
            turbidity = safe_float(request.form.get("turbidity"))
            ph = safe_float(request.form.get("ph"))
            latitude = safe_float(request.form.get("latitude"))
            longitude = safe_float(request.form.get("longitude"))

            # Prediction
            input_data = np.array([[color, turbidity, ph]])
            prediction = model.predict(input_data)[0]
            result = "Safe" if prediction == 1 else "Unsafe"

            # Log to DB
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO submissions (color, turbidity, ph, label, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)",
                (color, turbidity, ph, result, latitude, longitude)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            result = f"❌ Error: {str(e)}"

    elif not model:
        result = "❌ Model not loaded. Please check server logs."

    return render_template("index.html", result=result)

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/map_data")
def map_data():
    conn = get_db_connection()
    data = conn.execute("SELECT latitude, longitude, label FROM submissions WHERE latitude IS NOT NULL AND longitude IS NOT NULL").fetchall()
    conn.close()
    return jsonify([{"lat": row["latitude"], "lon": row["longitude"], "label": row["label"]} for row in data])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
