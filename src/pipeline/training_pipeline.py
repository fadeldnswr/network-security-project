'''
Training pipeline for the model.
'''
import os, sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
  DataIngestionConfig, DataValidationConfig,
  DataTransformationConfig, ModelTrainerConfig,
  TrainingPipelineConfig
)
from src.entity.artifact_entity import (
  DataIngestionArtifact, DataValidationArtifact,
  DataTransformationArtifact, ModelTrainerArtifact
)

# TrainingPipeline class to manage the entire training process
class TrainingPipeline:
  def __init__(self):
    try:
      self.training_pipeline_config = TrainingPipelineConfig()
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def start_data_ingestion(self):
    try:
      # Initialize Data Ingestion Config
      self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
      logging.info(f"Start data ingestion")
      
      # Initialize Data Ingestion component
      data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
      data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
      logging.info(f"Data ingestion completed successfully! {data_ingestion_artifact}")
      
      return data_ingestion_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
    try:
      # Initialize Data Validation Config
      data_validation_config = DataValidationConfig(
        training_pipeline_config=self.training_pipeline_config,
      )
      logging.info(f"Start data validation")
      
      data_validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config
      )
      data_validation_artifact = data_validation.initiate_data_validation()
      logging.info(f"Data validation completed successfully!")
      
      return data_validation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
    try:
      # Initialize Data Transformation
      data_transformation_config = DataTransformationConfig(
        training_pipeline_config=self.training_pipeline_config,
      )
      logging.info(f"Start data transformation")
      data_transformation = DataTransformation(
        data_transformation_config=data_transformation_config,
        data_validation_artifact=data_validation_artifact
      )
      data_transformation_artifact = data_transformation.initiate_data_transformation()
      logging.info(f"Data transformation completed successfully! {data_transformation_artifact}")
      
      return data_transformation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
    try:
      # Initialize Model Trainer
      self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
        training_pipeline_config=self.training_pipeline_config
      )
      logging.info(f"Start model training")
      model_trainer = ModelTrainer(
        data_transformation_artifact=data_transformation_artifact,
        model_trainer_config=self.model_trainer_config
      )
      model_trainer_artifact = model_trainer.initiate_model_trainer()
      logging.info(f"Model training completed successfully! {model_trainer_artifact}")
      return model_trainer_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def run_pipeline(self):
    try:
      # Run the entire training pipeline
      logging.info(f"Starting training pipeline")
      
      data_ingestion_artifact = self.start_data_ingestion()
      data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
      data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
      model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
      
      logging.info(f"Training pipeline completed successfully!")
      return model_trainer_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)