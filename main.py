# Import modules
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from app.model.predict import predict

app = FastAPI()


@app.get("/")
def health():
    return {"message": "Server is Running"}


@app.get("/predict")
def prediction_calculator():
    json_dummy={"total_area_sqm": 2000,
        "primary_energy_consumption_sqm":34}
    try:
        prediction = predict(json_dummy)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


print("this is an update inside dev")
