'''
Data validation module for the application.
This module is responsible for validating the data
'''
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp

import os, sys
import pandas as pd

class DataValidation:
  def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
    '''
    Initializes the DataValidation class with the provided artifacts and configuration.
    :param data_ingestion_artifact: Artifact containing paths of ingested data files
    :param data_validation_config: Configuration for data validation
    :raises NetworkSecurityException: If the schema file does not exist
    '''
    try:
      self.data_ingestion_artifact = data_ingestion_artifact
      self.data_validation_config = data_validation_config
      self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  @staticmethod
  def read_data(file_pat: str) -> pd.DataFrame:
    '''
    Reads data from the specified file path.
    :param file_path: Path to the data file
    :return: DataFrame containing the data
    :raises NetworkSecurityException: If the file does not exist or cannot be read
    '''
    try:
      if not os.path.exists(file_pat):
        raise FileNotFoundError(f"The file {file_pat} does not exist.")
      return pd.read_csv(file_pat)
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def validate_nums_of_cols(self, df: pd.DataFrame) -> bool:
    '''
    Validates the number of columns in the DataFrame against the schema.
    :param df: DataFrame to validate
    :return: True if the number of columns matches the schema, False otherwise
    '''
    try:
      number_of_cols = len(self._schema_config)
      logging.info(f"Required number of columns: {number_of_cols}")
      logging.info(f"Dataframe has {len(df.columns)} columns.")
      if len(df.columns) == number_of_cols:
        return True
      else:
        return False
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def validate_column_names(self, df: pd.DataFrame) -> bool:
    '''
    Validates the column names in the DataFrame against the schema.
    :param df: DataFrame to validate
    :return: True if the column names match the schema, False otherwise
    '''
    try:
      schema_columns = list(self._schema_config.keys())
      logging.info(f"Schema columns: {schema_columns}")
      logging.info(f"Dataframe columns: {df.columns}")
      if set(df.columns) == set(schema_columns):
        return True
      else:
        return False
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def detect_data_drift(self, base_df, current_df, threshold=0.05) -> bool:
    '''
    Detects data drift between two DataFrames using the Kolmogorov-Smirnov test.
    :param base_df: Base DataFrame (e.g., training data)
    :param current_df: Current DataFrame (e.g., new data)
    :param threshold: Significance level for the KS test
    :return: True if data drift is detected, False otherwise
    '''
    try:
      status = True
      report = {}
      #  Check if both DataFrames have the same columns
      for column in base_df.columns:
        df_1 = base_df[column]
        df_2 = current_df[column]
        is_sample_dist = ks_2samp(df_1, df_2)
        
        # Check if the p-value is below the threshold
        if threshold < is_sample_dist.pvalue:
          is_found = False
        else:
          is_found = True
          status = False
        
        report.update({column: {
          "p_value": float(is_sample_dist.pvalue),
          "drift_status": is_found,
        }})
        drift_report_file_path = self.data_validation_config.drift_repost_dir
        
        # Create directory if it does not exist
        dir_path = os.path.dirname(drift_report_file_path)
        os.makedirs(dir_path, exist_ok=True)
        write_yaml_file(file_path=drift_report_file_path, content=report)
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def initiate_data_validation(self) -> DataValidationArtifact:
    '''
    Initiates the data validation process.
    :return: DataValidationArtifact containing validation results
    :raises NetworkSecurityException: If the schema file does not exist or if validation fails
    '''
    try:
      # Initialize paths for train and test files
      train_file_path = self.data_ingestion_artifact.trained_file_path
      test_file_path = self.data_ingestion_artifact.test_file_path
      
      # Read the data from train and test files
      train_df = DataValidation.read_data(train_file_path)
      test_df = DataValidation.read_data(test_file_path)
      
      # Validate the number of columns in train and test dataframes
      status = self.validate_nums_of_cols(train_df)
      if not status:
        error_msg = f"Train dataframe does not contain all the columns"
      
      status = self.validate_nums_of_cols(test_df)
      if not status:
        error_msg = f"Test dataframe does not contain all the columns"
      
      # Validate the column names in train and test dataframes
      status = self.validate_column_names(train_df)
      if not status:
        error_msg = f"Train dataframe does not contain all the columns as per schema"
      
      status = self.validate_column_names(test_df)
      if not status:
        error_msg = f"Test dataframe does not contain all the columns as per schema"
      
      # Check data drift using Kolmogorov-Smirnov test
      status = self.detect_data_drift(base_df=train_df, current_df=test_df)
      dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
      os.makedirs(dir_path, exist_ok=True)
      
      # Convert train and test DataFrames to CSV files
      train_df.to_csv(
        self.data_validation_config.valid_train_file_path,
        index=False,
        header=True
        )
      
      test_df.to_csv(
        self.data_validation_config.valid_test_file_path,
        index=False,
        header=True
      )
      
      # Create data validation artifact
      data_validation_artifact = DataValidationArtifact(
        validation_status=status,
        valid_train_file_path=self.data_validation_config.valid_train_file_path,
        valid_test_file_path=self.data_validation_config.valid_test_file_path,
        invalid_train_file_path=None,
        invalid_test_file_path=None,
        drift_report_file_path=self.data_validation_config.drift_repost_dir
      )
      return data_validation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
