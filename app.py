'''
Run this file to start the FastAPI application.
'''
import os, sys
import certifi
import pymongo
import pandas as pd

from dotenv import load_dotenv
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object
from src.constant.training_pipeline import (
  DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
)
from src.utils.ml_utils.model.estimator import NetworkModel

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Get the mongoDB connection URI from environment variables
ca = certifi.where()
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URI")
print(f"MongoDB URL: {mongo_db_url}")

# Initialize mongo client
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
db = client[DATA_INGESTION_DATABASE_NAME]
collection = db[DATA_INGESTION_COLLECTION_NAME]

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

templates = Jinja2Templates(directory="./templates")

# Create get endpoint for the root path
@app.get("/", tags=["authentication"])
async def root():
  return RedirectResponse(url="/docs")

@app.get("/train")
async def train_model():
  try:
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()
    return Response("Training pipeline executed successfully!", status_code=200)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

@app.post("/predict", tags=["prediction"])
async def predict(request: Request, file: UploadFile = File(...)):
  try:
    df = pd.read_csv(file.file)
    preprocessor = load_object("final_model/preprocessor.pkl")
    model = load_object("final_model/model.pkl")
    
    network_model = NetworkModel(
      preprocessor=preprocessor,
      model=model
    )
    print(df.iloc[0])
    y_pred = network_model.predict(df)
    print(f"Prediction: {y_pred}")
    
    df["predicted_column"] = y_pred
    print(df["predicted_column"])
    
    df.to_csv("prediction_output/output.csv")
    
    table_html = df.to_html(classes="table table-striped")
    return templates.TemplateResponse(
      "index.html", 
      {
        "request": request,
        "table_html": table_html,
      }
    )
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# Run the FastAPI application
if __name__ == "__main__":
  app.run(
    app,
    host="localhost",
    port=8000,
  )