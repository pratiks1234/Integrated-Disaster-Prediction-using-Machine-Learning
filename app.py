from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
earthquake_model = joblib.load("earthquake_magnitude_model.pkl")
flood_model = joblib.load("flood_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        disaster_type = request.form["disaster_type"]

        if disaster_type == "earthquake":
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            depth = float(request.form["depth"])
            nst = float(request.form["nst"])
            gap = float(request.form["gap"])
            dmin = float(request.form["dmin"])
            rms = float(request.form["rms"])
            magNst = float(request.form["magNst"])

            input_data = np.array([[latitude, longitude, depth, nst, gap, dmin, rms, magNst]])
            prediction = earthquake_model.predict(input_data)[0]

            return jsonify({"prediction": f"Predicted Earthquake Magnitude: {prediction:.2f}"})

        elif disaster_type == "flood":
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            rainfall = float(request.form["rainfall"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            river_discharge = float(request.form["river_discharge"])
            water_level = float(request.form["water_level"])
            elevation = float(request.form["elevation"])
            land_cover = float(request.form["land_cover"])
            soil_type = float(request.form["soil_type"])
            population_density = float(request.form["population_density"])
            infrastructure = float(request.form["infrastructure"])
            historical_floods = float(request.form["historical_floods"])

            input_data = np.array([[latitude, longitude, rainfall, temperature, humidity,
                                    river_discharge, water_level, elevation, land_cover,
                                    soil_type, population_density, infrastructure, historical_floods]])
            
            prediction = flood_model.predict(input_data)[0]

            return jsonify({"prediction": f"Flood Prediction: {'Flood Expected' if prediction == 1 else 'No Flood'}"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
