# this is the industrial standard for ML endpoints usinf Fast API
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

# validate new data
class InputData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

@app.get("/")
def home():
    return {"message": "ML API running"}

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(data:InputData):
    custom_data = CustomData(
        gender=data.gender,
        race_ethnicity=data.race_ethnicity,
        parental_level_of_education=data.parental_level_of_education,
        lunch=data.lunch,
        test_preparation_course=data.test_preparation_course,
        reading_score=data.reading_score,
        writing_score=data.writing_score
    )

    pred_df = custom_data.get_data_as_frame()

    pred_pipeline = PredictPipeline()

    prediction = np.round(
        pred_pipeline.predict(pred_df),
        2
    )

    return {
        "prediction": float(prediction[0])
    }