from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import (
  DataIngestionConfig, TrainingPipelineConfig, 
  DataValidationConfig, DataTransformationConfig, 
  ModelTrainerConfig
  )

import os
import sys

if __name__ == "__main__":
  try:
    training_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    
    logging.info("Initiate the data ingestion")
    
    data_ing_artifact = data_ingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed successfully")
    print(data_ing_artifact)
    
    # Initiate data validation
    data_validation_config = DataValidationConfig(training_config)
    data_validation = DataValidation(data_ing_artifact, data_validation_config)
    logging.info("Initiate the data validation")
    data_val_artifact = data_validation.initiate_data_validation()
    logging.info("Data validation completed successfully")
    print(data_val_artifact)
    
    # Initiate data transformation
    logging.info("Initiate the data transformation")
    data_transformation_config = DataTransformationConfig(training_pipeline_config=training_config)
    data_transformation = DataTransformation(
      data_validation_artifact=data_val_artifact,
      data_transformation_config=data_transformation_config
    )
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    print(data_transformation_artifact)
    logging.info("Data transformation completed successfully")
    
    # Initiate model training
    logging.info("Initiate the model training")
    model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_config)
    model_trainer = ModelTrainer(
      model_trainer_config=model_trainer_config,
      data_transformation_artifact=data_transformation_artifact,
    )
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    logging.info("Model training completed successfully")
  except Exception as e:
    raise NetworkSecurityException(e, sys)
