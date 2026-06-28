from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from ml_pipeline import MLPipeline

app = FastAPI(title="Heart Disease ML API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline with the path to heart.csv
# We assume main.py is run from the 'backend' folder and heart.csv is in the parent 'Project 2' folder
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heart.csv")
pipeline = MLPipeline(DATA_PATH)

@app.get("/")
def read_root():
    return {"message": "Heart Disease ML API is running"}

@app.get("/api/step1_load")
def step1_load():
    try:
        result = pipeline.load_and_preprocess()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/step2_features")
def step2_features():
    try:
        result = pipeline.feature_engineering()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/step3_train")
def step3_train():
    try:
        result = pipeline.train_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/step4_evaluate")
def step4_evaluate():
    try:
        result = pipeline.evaluate_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/step5_best_model")
def step5_best_model():
    try:
        result = pipeline.get_best_model_analysis()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
