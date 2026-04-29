# Disaster Prediction System: Earthquake Magnitude and Flood Occurrence

This project is a machine learning based disaster prediction system that supports two prediction tasks:

1. **Earthquake magnitude prediction** using historical earthquake event data.
2. **Flood occurrence prediction** using environmental, geographic, and historical flood-related features.

The project includes Jupyter notebooks for model development and comparison, trained model files saved as `.pkl`, and a Flask web application that allows users to enter input values and receive predictions through a simple browser interface.

---

## Project Overview

Natural disasters such as earthquakes and floods can cause major damage to human life, infrastructure, and the environment. This project applies supervised machine learning techniques to historical disaster-related datasets to estimate disaster risk and severity.

The earthquake module is designed as a **regression problem**, where the model predicts the expected earthquake magnitude. The flood module is designed as a **binary classification problem**, where the model predicts whether a flood is expected or not expected based on input conditions.

---

## Key Features

- Predicts earthquake magnitude from seismic and geographic inputs.
- Predicts flood occurrence using rainfall, humidity, river discharge, water level, elevation, land cover, soil type, and other risk factors.
- Includes exploratory data analysis, preprocessing, model training, and model comparison notebooks.
- Uses trained machine learning models saved with `joblib`/`pickle`.
- Provides a Flask web interface for real-time prediction.
- Supports separate input forms for earthquake and flood prediction.

---

## Project Structure

```text
MAJOR PROJECT FINAL/
│
├── app.py                              # Flask application for prediction
├── requirements.txt                    # Python dependencies
├── test.py                             # Test script for earthquake model prediction
│
├── dataset/
│   ├── flood_dataset.csv               # Flood prediction dataset
│   └── india_earthquakes.csv           # Earthquake magnitude dataset
│
├── templates/
│   └── index.html                      # Front-end UI for the Flask app
│
├── Earthquake_Final (1).ipynb          # Earthquake model training and evaluation notebook
├── Flood_Prediction_Final (1).ipynb    # Flood model training and evaluation notebook
│
├── earthquake_magnitude_model.pkl      # Trained earthquake magnitude prediction model used by app.py
├── earthquake_model.pkl                # Additional saved earthquake model
├── flood_model.pkl                     # Trained flood prediction model used by app.py
│
└── catboost_info/                      # Training logs/artifacts generated during experimentation
```

---

## Dataset Description

This project uses two tabular datasets stored in the `dataset/` folder.

### 1. Earthquake Dataset

**File:** `dataset/india_earthquakes.csv`

The earthquake dataset contains historical earthquake records related to India and nearby regions. It has **14,218 rows** and **22 columns**. Each row represents an earthquake event with geographic, seismic, and metadata attributes.

#### Important Columns

| Column | Description |
|---|---|
| `time` | Date and time of the earthquake event |
| `latitude` | Latitude of the earthquake location |
| `longitude` | Longitude of the earthquake location |
| `depth` | Depth of the earthquake event |
| `mag` | Earthquake magnitude; this is the target variable |
| `magType` | Magnitude measurement type |
| `nst` | Number of seismic stations used |
| `gap` | Azimuthal gap between seismic stations |
| `dmin` | Minimum distance to the seismic station |
| `rms` | Root mean square travel-time residual |
| `magNst` | Number of stations used to calculate magnitude |
| `place` | Location description |
| `type` | Event type, such as earthquake |
| `status` | Review status of the event |

#### Features Used for Earthquake Prediction

The final earthquake prediction model uses the following input features:

```text
latitude, longitude, depth, nst, gap, dmin, rms, magNst
```

The target variable is:

```text
mag
```

#### Preprocessing

- Missing values were handled by removing incomplete rows during final model training.
- Date/time fields were explored and converted during experimentation.
- The final model focuses on numerical seismic and geographic features.
- The model predicts earthquake magnitude as a continuous numerical value.

---

### 2. Flood Dataset

**File:** `dataset/flood_dataset.csv`

The flood dataset contains environmental and geographic conditions that can influence flood occurrence. It has **40,000 rows** and **14 columns**. Each row represents a location/condition record with flood-related risk indicators.

#### Important Columns

| Column | Description |
|---|---|
| `Latitude` | Latitude of the location |
| `Longitude` | Longitude of the location |
| `Rainfall` | Rainfall amount |
| `Temperature` | Temperature value |
| `Humidity` | Humidity percentage/value |
| `River_Discharge` | River discharge level |
| `Water_Level` | Water level measurement |
| `Elevation` | Elevation of the location |
| `Land_Cover` | Type of land cover |
| `Soil_Type` | Soil category |
| `Population_Density` | Population density of the area |
| `Infrastructure` | Infrastructure indicator |
| `Historical_Floods` | Previous flood history indicator |
| `Flood_Occurred` | Target variable; `1` means flood occurred and `0` means no flood occurred |

#### Features Used for Flood Prediction

The flood prediction model uses the following input features:

```text
Latitude, Longitude, Rainfall, Temperature, Humidity,
River_Discharge, Water_Level, Elevation, Land_Cover,
Soil_Type, Population_Density, Infrastructure, Historical_Floods
```

The target variable is:

```text
Flood_Occurred
```

#### Categorical Encoding

The flood dataset contains two categorical columns: `Land_Cover` and `Soil_Type`. These were encoded into numeric values before training.

`Land_Cover` encoding:

| Encoded Value | Original Category |
|---:|---|
| 0 | Agricultural |
| 1 | Desert |
| 2 | Forest |
| 3 | Urban |
| 4 | Water Body |

`Soil_Type` encoding:

| Encoded Value | Original Category |
|---:|---|
| 0 | Clay |
| 1 | Loam |
| 2 | Peat |
| 3 | Sandy |
| 4 | Silt |

#### Preprocessing

- Missing numerical values were filled using the mean value of each column.
- Categorical variables were label encoded.
- The target variable `Flood_Occurred` was used for binary classification.
- The final web app expects the categorical fields as encoded numeric values.

---

## Machine Learning Models

### Earthquake Magnitude Prediction

The earthquake prediction task was treated as a regression problem. Several regression models were tested, including:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regressor

The final model used in the project is a **Random Forest Regressor** with hyperparameter tuning using `RandomizedSearchCV`.

Final earthquake model performance from the notebook:

| Metric | Value |
|---|---:|
| Mean Squared Error | 0.0361 |
| Root Mean Squared Error | 0.1900 |
| Mean Absolute Error | 0.1432 |
| R-squared Score | 0.7580 |

---

### Flood Occurrence Prediction

The flood prediction task was treated as a binary classification problem. Multiple classification models were tested, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- K-Nearest Neighbors
- AdaBoost Classifier
- XGBoost Classifier
- LightGBM Classifier
- Voting Classifier

The saved `flood_model.pkl` file is used by the Flask application to classify whether a flood is expected or not expected.

---

## Web Application

The Flask application provides a simple user interface where users can select the disaster type and enter input values.

### Earthquake Prediction Inputs

```text
Latitude, Longitude, Depth, NST, Gap, Dmin, RMS, MagNst
```

The output is a predicted earthquake magnitude.

### Flood Prediction Inputs

```text
Latitude, Longitude, Rainfall, Temperature, Humidity,
River Discharge, Water Level, Elevation, Land Cover,
Soil Type, Population Density, Infrastructure, Historical Floods
```

The output is either:

```text
Flood Expected
```

or

```text
No Flood
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install flask joblib
```

If you update `requirements.txt`, include the following dependencies:

```text
flask
joblib
lightgbm
matplotlib
numpy
pandas
scikit-learn
seaborn
xgboost
```

### 4. Run the Flask App

```bash
python app.py
```

Then open the local server in your browser:

```text
http://127.0.0.1:5000/
```

---

## Example Usage

### Example Earthquake Input

```text
Latitude: 19.0911
Longitude: 72.5535
Depth: 10
NST: 70
Gap: 65
Dmin: 4.14
RMS: 0.8
MagNst: 5.2
```

Example output:

```text
Predicted Earthquake Magnitude: 5.00
```

### Example Flood Input

```text
Latitude: 19.04
Longitude: 72.86
Rainfall: 1000
Temperature: 31
Humidity: 30
River Discharge: 126.18
Water Level: 4.41
Elevation: 15
Land Cover: 4
Soil Type: 4
Population Density: 15627
Infrastructure: 0
Historical Floods: 0
```

Example output:

```text
Flood Prediction: No Flood
```

---

## Notes on Model Files

The repository includes trained `.pkl` model files so the Flask application can make predictions without retraining the models every time.

Because `flood_model.pkl` is a large file, GitHub may warn about file size limits. If needed, use **Git LFS** for storing large model files.

Also, saved `.pkl` files can depend on the scikit-learn version used during training. If you face loading errors, use the same scikit-learn version used during training or rerun the notebooks to regenerate the model files.

---

## Limitations

- This project is an academic/educational machine learning prototype.
- The predictions should not be used as official disaster warnings or emergency alerts.
- Model results depend heavily on the quality and accuracy of input values.
- The flood model expects encoded numeric values for `Land_Cover` and `Soil_Type`.
- Real-world disaster prediction requires more advanced geospatial, meteorological, and real-time sensor data.

---

## Future Improvements

- Add model versioning and save preprocessing pipelines together with the models.
- Improve the UI with clearer labels, dropdowns for categorical fields, and input validation.
- Add maps to visualize earthquake and flood risk locations.
- Deploy the Flask application using Render, Railway, AWS, or Heroku.
- Add API documentation for the `/predict` endpoint.
- Improve model evaluation using cross-validation and additional metrics.
- Store categorical encoders and preprocessing steps in a single pipeline.

---

## Disclaimer

This project is for educational and research purposes only. It should not be used as a replacement for official disaster monitoring systems, government alerts, meteorological agencies, or emergency management services.
