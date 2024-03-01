# Import modules
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from model.predict import predict

app = FastAPI()


class InputData(BaseModel):
    total_area_sqm: int
    primary_energy_consumption_sqm: int


@app.get("/")
def health():
    return {"message": "Server is Running"}


@app.post("/predict")
def prediction_calculator(input_data: InputData):
    try:
        prediction = predict(input_data.dict())
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


print("this is an update inside ss dev")
