from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig

import os
import sys

if __name__ == "__main__":
  try:
    training_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    
    logging.info("Initiate the data ingestion")
    
    data_ing_artifact = data_ingestion.initiate_data_ingestion()
    print(data_ing_artifact)
  except Exception as e:
    raise NetworkSecurityException(e, sys)
