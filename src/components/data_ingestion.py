import os
import sys
import pandas as pd
import numpy as np
import pymongo

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from typing import List
from sklearn.model_selection import train_test_split

# Import Mongo DB env
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URI")

# Create data ingestion class
class DataIngestion:
  '''
  This class is used for data ingestion process
  such as read data from db, transforming and split
  into train and test
  '''
  def __init__(self, data_ingestion_config:DataIngestionConfig):
    try:
      self.data_ingestion_config = data_ingestion_config
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def export_collection_as_df(self):
    try:
      # Initiate db and collection name
      db_name = self.data_ingestion_config.database_name
      collection_name = self.data_ingestion_config.collection_name
      
      # Create mongo db collection
      self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
      collection = self.mongo_client[db_name][collection_name]
      
      # Convert to dataframe
      df = pd.DataFrame(list(collection.find()))
      if "_id" in df.columns.to_list(): # Drop _id column
        df = df.drop(columns=["_id"], axis=1)
      
      # Replace the na value into NaN
      df.replace({"na": np.nan}, inplace=True)
      return df
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def export_feature_score(self, df: pd.DataFrame):
    try:
      feature_store_file_path = self.data_ingestion_config.feature_store_file_path
      
      # Create the folder
      dir_path = os.path.dirname(feature_store_file_path)
      os.makedirs(dir_path, exist_ok=True)
      
      # Convert dataframe into csv
      df.to_csv(feature_store_file_path, index=False, header=True)
      return df
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def split_data_to_train_test(self, df: pd.DataFrame):
    try:
      # Perform train and test split
      train_set, test_set = train_test_split(
        df, test_size=self.data_ingestion_config.train_test_split_ratio
      )
      logging.info("Performed train and test split on the dataframe")
      logging.info("Exited split_data_to_train_test method of DataIngestion class")
      
      # Create directory for train file path
      dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
      os.makedirs(dir_path, exist_ok=True)
      logging.info("Exporting train and test file path")
      
      # Convert train and test data into csv format
      train_set.to_csv(
        self.data_ingestion_config.training_file_path,
        index=False,
        header=True
      )
      test_set.to_csv(
        self.data_ingestion_config.test_file_path,
        index=False,
        header=True
      )
      logging.info("Exported train and test file path")
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def initiate_data_ingestion(self):
    try:
      # Initiate the df
      df = self.export_collection_as_df()
      
      # Export data to feature store
      df = self.export_feature_score(df)
      
      # Data preprocessing
      self.split_data_to_train_test(df)
      
      # Create data ingestion artifact
      data_ingestion_artifact = DataIngestionArtifact(
        trained_file_path=self.data_ingestion_config.training_file_path,
        test_file_path=self.data_ingestion_config.test_file_path
      )
      return data_ingestion_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)