import json
import joblib
import pandas as pd

json_dummy={"total_area_sqm": 2000, "primary_energy_consumption_sqm":34}

def predict(input_data):
    """Predicts house prices from input data."""
    # Load the trained model
    artifacts = joblib.load("app/model/artifacts.joblib")
    print(artifacts)
    # Prepare the input data
    data = pd.DataFrame(input_data, index = [0])
    
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]
    
    data[num_features] = imputer.transform(data[num_features])
    data_cat = enc.transform(data[cat_features]).toarray()

    # Extract the used data
    data = data[num_features]

    # Make predictions
    predictions = model.predict(data)

    return predictions[0]

if __name__ == "__main__":
    pass
    # For testing purposes
    #input_data = {"living_area": 1500, "energy_consumption": 250}  # Example input data
    # prediction = predict(json_dummy)
    # print("Predicted house price:", prediction)