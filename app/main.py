# Import modules
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel


# Dummy predict function
def predict(data):
    return {"prediction": "prediction"}


# class that defines the structure of the input data
class Property(BaseModel):
    LivingArea: int
    TypeOfProperty: str
    Bedrooms: int
    PostalCode: int
    SurfaceOfGood: Optional[int] = None
    Garden: Optional[bool] = None
    GardenArea: Optional[int] = None
    SwimmingPool: Optional[bool] = None
    Furnished: Optional[bool] = None
    Openfire: Optional[bool] = None
    Terrace: Optional[bool] = None
    NumberOfFacades: Optional[int] = None
    ConstructionYear: Optional[int] = None
    StateOfBuilding: Optional[str] = None
    Kitchen: Optional[str] = None


# initialized fastAPI instance
app = FastAPI()


# API endpoint to check if the API is alive
@app.get("/")
def health():
    return {"message": "Server is Running"}


# API endpoint to access the prediction
@app.post("/predict")
def prediction_calculator(property_data: Property):
    try:
        prediction = predict(property_data.dict())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


print("this is an update inside dev")
