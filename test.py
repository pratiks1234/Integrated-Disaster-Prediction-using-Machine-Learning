import pickle
import numpy as np

# Load the trained earthquake model
try:
    with open("earthquake_model.pkl", "rb") as file:
        model = pickle.load(file)

    if not hasattr(model, "predict"):
        raise ValueError("❌ The loaded object is not a trained model.")

    # Check if model is trained
    if hasattr(model, "n_features_in_"):
        print(f"✅ Model is trained with {model.n_features_in_} features.")
    else:
        raise ValueError("❌ Model is NOT trained properly.")

    # Define test input
    test_input = np.array([[19.0911, 72.5235, 10, 70, 65, 4.14, 0.8, 5.2]])  

    # Ensure input shape matches model
    if test_input.shape[1] != model.n_features_in_:
        raise ValueError(f"❌ Model expects {model.n_features_in_} features but got {test_input.shape[1]}.")

    # Make prediction
    prediction = model.predict(test_input)[0]
    print(f"✅ Predicted Earthquake Magnitude: {prediction:.2f}")

except Exception as e:
    print(f"❌ Error: {e}")
