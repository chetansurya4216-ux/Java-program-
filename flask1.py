from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ✅ DEFINE BASE_DIR FIRST
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ LOAD MODELS AFTER BASE_DIR
crop_model = joblib.load(os.path.join(BASE_DIR, "crop_model.pkl"))
yield_model = joblib.load(os.path.join(BASE_DIR, "yield_model.pkl"))
fert_model = joblib.load(os.path.join(BASE_DIR, "fertilizer_model.pkl"))
hist_model = joblib.load(os.path.join(BASE_DIR, "historical_crop_model.pkl"))



@app.route("/")
def home():
    return "Smart Agriculture Backend Running"

# ===============================
# CASE 1: SOIL + WEATHER
# ===============================
@app.route("/predict/case1", methods=["POST"])
def case1_predict():
    data = request.json

    X = np.array([[ 
        data["State"],
        data["District"],
        data["Soil_Type"],
        data["Nitrogen"],
        data["Phosphorus"],
        data["Potassium"],
        data["pH"],
        data["Temperature"],
        data["Rainfall"],
        data["Humidity"]
    ]])

    crop = int(crop_model.predict(X)[0])
    yield_val = float(yield_model.predict(X)[0])
    fertilizer = int(fert_model.predict(X)[0])

    return jsonify({
        "Predicted_Crop": crop,
        "Expected_Yield": round(yield_val, 2),
        "Recommended_Fertilizer": fertilizer
    })

# ===============================
# CASE 2: HISTORICAL
# ===============================
@app.route("/predict/case2", methods=["POST"])
def case2_predict():
    data = request.json

    X = np.array([[ 
        data["State"],
        data["Soil_Type"],
        data["Past_Crop"],
        data["Pesticide"],
        data["Temperature"]
    ]])

    crop = int(hist_model.predict(X)[0])

    return jsonify({
        "Recommended_Crop": crop
    })

if __name__ == "__main__":
    app.run(debug=True)