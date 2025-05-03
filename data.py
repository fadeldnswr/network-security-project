import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from dotenv import load_dotenv
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

# Load the environment
load_dotenv()

# Create mongo db url
MONGO_DB_URL = os.getenv("MONGO_DB_URI")

# Create certifi instance
ca = certifi.where() 

# Create data extraction
class NetworkDataExtraction:
  def __init__(self):
    try:
      pass
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def csv_to_json(self, file_path):
    try:
      # Read and reset index of the dataframe
      data = pd.read_csv(file_path)
      data.reset_index(drop=True, inplace=True)
      
      # Convert the dataframe as a list of json
      records = list(json.loads(data.T.to_json()).values())
      return records
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def insert_data_to_db(self, records, db, collection):
    try:
      self.db = db
      self.collection = collection
      self.records = records
      
      # Create mongo db connection
      self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
      self.db = self.mongo_client[self.db]
      
      # Insert the collection to db
      self.collection = self.db[self.collection]
      self.collection.insert_many(self.records)
      
      return (len(self.records))
    except Exception as e:
      raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
  FILE_PATH = "network-data/phisingData.csv"
  DATABASE = "fadeldnswr"
  collection = "network-data"
  network_obj = NetworkDataExtraction()
  records = network_obj.csv_to_json(file_path=FILE_PATH)
  print(records)
  no_of_records = network_obj.insert_data_to_db(records, DATABASE, collection)
  print(no_of_records)