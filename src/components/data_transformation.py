'''
Data Transformation Module
This module handles the data transformation process, including loading the dataset,
imputing missing values, and saving the transformed data.
'''
import sys, os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from src.constant.training_pipeline import (
  TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
)
from src.entity.artifact_entity import (
  DataTransformationArtifact, DataValidationArtifact
)
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array, save_object

class DataTransformation:
  def __init__(self, data_validation_artifact: DataValidationArtifact,
    data_transformation_config: DataTransformationConfig):
    try:
      self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
      self.data_transformation_config:DataTransformationConfig = data_transformation_config
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  @staticmethod
  def read_data(file_path: str) -> pd.DataFrame:
    '''
    Reads a CSV file and returns a DataFrame.
    :param file_path: Path to the CSV file
    :return: DataFrame containing the data
    '''
    try:
      df = pd.read_csv(file_path)
      return df
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def get_data_transformer_object(cls) -> Pipeline:
    '''
    Creates a data transformation pipeline with KNN imputer.
    :return: A scikit-learn Pipeline object for data transformation
    '''
    logging.info("Creating data transformation pipeline with KNN imputer")
    try:
      imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
      processor:Pipeline = Pipeline([("imputer", imputer)])
      return processor
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def initiate_data_transformation(self) -> DataTransformationArtifact:
    logging.info("Initiating data transformation")
    try:
      # Load the validated train and test data
      logging.info("Loading validated train and test data")
      train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
      test_df = self.read_data(self.data_validation_artifact. valid_test_file_path)
      logging.info("Data loaded successfully")
      
      # Training dataframe
      input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
      target_feature_train_df = train_df[TARGET_COLUMN]
      target_feature_train_df = target_feature_train_df.replace(-1, 0)
      
      # Testing dataframe
      input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
      target_feature_test_df = test_df[TARGET_COLUMN]
      target_feature_test_df = target_feature_test_df.replace(-1, 0)
      
      # Create the data transformation pipeline
      preprocessor = self.get_data_transformer_object()
      preprocessor_obj = preprocessor.fit(input_features_train_df)
      transformed_input_train_feature = preprocessor_obj.transform(input_features_train_df)
      tranformed_input_test_feature = preprocessor_obj.transform(input_features_test_df)
      
      # Create the array
      train_arr = np.c_[transformed_input_train_feature, target_feature_train_df]
      test_arr = np.c_[tranformed_input_test_feature, target_feature_test_df]
      
      # Save the transformed data
      save_numpy_array(self.data_transformation_config.transformed_train_file_path, array=train_arr)
      save_numpy_array(self.data_transformation_config.transformed_test_file_path, array=test_arr)
      save_object(self.data_transformation_config.transformed_object_file_path, obj=preprocessor_obj)
      save_object("final_model/preprocessor.pkl", preprocessor_obj)
      
      # Prepare the data transformation artifact
      data_transformation_artifact = DataTransformationArtifact(
        transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
        transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
        transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
      )
      return data_transformation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)