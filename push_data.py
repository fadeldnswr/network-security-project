import certifi, sys, os
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
import json
from dotenv import load_dotenv

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URI")

class NetworkDataExtract():
  def __init__(self):
    try:
      pass
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def csv_to_json_converter(self, file_path):
    try:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df.reset_index(drop=True, inplace=True)
      records = list(json.loads(df.T.to_json()).values())
      return records
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def push_data_to_mongo(self, records, db_name, collection_name):
    try:
      self.db = db_name
      self.collection = collection_name
      self.records = records
      
      self.mongo_client = pymongo.MongoClient(
        mongo_db_url
      )
      self.db = self.mongo_client[self.db]
      self.collection = self.db[self.collection]
      self.collection.insert_many(self.records)
      return len(self.records)
    except Exception as e:
      raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
  FILE_PATH = "data/network_data.csv"
  DATABASE = "network_security_db"
  COLLECTION = "network_data"
  network_obj = NetworkDataExtract()
  records = network_obj.csv_to_json_converter(FILE_PATH)
  print(records)
  no_of_records = network_obj.push_data_to_mongo(
    records=records,
    db_name=DATABASE,
    collection_name=COLLECTION
  )
  print(no_of_records)