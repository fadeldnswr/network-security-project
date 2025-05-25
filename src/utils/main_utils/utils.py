'''
Utility functions for various tasks.
'''

import yaml
import os,sys
import numpy as np
import dill
import pickle

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# read_yaml_file function created to read a YAML file and return its content
def read_yaml_file(file_path: str) -> dict:
  '''
  Reads a YAML file and returns its content as a dictionary.
  :param file_path: Path to the YAML file
  :return: Dictionary containing the content of the YAML file
  '''
  try:
    with open(file_path, "r") as file:
      return yaml.safe_load(file)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# write_yaml_file function created to write a dictionary to a YAML file
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
  '''
  Writes a dictionary to a YAML file.
  :param file_path: Path to the YAML file
  :param content: Dictionary to write to the file
  :param replace: If True, replaces the existing file; if False, appends to the file
  :raises NetworkSecurityException: If the file cannot be written
  '''
  try:
    if replace:
      if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
      yaml.dump(content, file)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# Save numpy array function created to save a numpy array to a file
def save_numpy_array(file_path: str, array: np.array):
  '''
  Saves a numpy array to a file.
  :param file_path: Path to the file where the array will be saved
  :param array: Numpy array to save
  :raises NetworkSecurityException: If the file cannot be saved
  '''
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file:
      np.save(file, array)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# save object function created to save an object to a file using dill
def save_object(file_path: str, obj: object) -> None:
  '''
  Saves an object to a file using dill.
  :param file_path: Path to the file where the object will be saved
  :param obk: Object to save
  :raises NetworkSecurityException: If the object cannot be saved
  '''
  try:
    logging.info(f"Saving object to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
      pickle.dump(obj, file)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# load object function created to load an object from a file using dill
def load_object(file_path: str) -> object:
  '''
  Loads an object from a file using dill.
  :param file_path: Path to the file from which the object will be loaded
  :return: Loaded object
  '''
  try:
    if not os.path.exists(file_path):
      raise Exception(f"File {file_path} does not exist.")
    with open(file_path, "rb") as file:
      print(file)
      return pickle.load(file)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# load numpy array data function created to load a numpy array from a file
def load_numpy_array_data(file_path: str) -> np.array:
  '''
  Loads a numpy array from a file.
  :param file_path: Path to the file from which the numpy array will be loaded
  :return: Loaded numpy array
  :raises NetworkSecurityException: If the file cannot be loaded
  '''
  try:
    with open(file_path, "rb") as file:
      return np.load(file)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

# evaluate models function created to evaluate multiple models and return the best one
def evaluate_models(
  X_train, y_train, 
  X_test, y_test, 
  models, param):
  '''
  Evaluates multiple machine learning models and returns the best one based on accuracy.
  :param X_train: Training features
  :param y_train: Training labels
  :param X_test: Testing features
  :param y_test: Testing labels
  :param models: Dictionary of models to evaluate
  :param params: Dictionary of parameters for each model
  :return: Tuple containing the best model, its name, and the accuracy score
  '''
  try:
    report = {}
  
    for i in range(len(list(models))):
      model = list(models.values())[i]
      params = param[list(models.keys())[i]]
      
      gs = GridSearchCV(model, params, cv=3)
      gs.fit(X_train, y_train)
      
      model.set_params(**gs.best_params_)
      model.fit(X_train, y_train)
      
      y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)
      
      train_model_score = r2_score(y_train, y_train_pred)
      test_model_score = r2_score(y_test, y_test_pred)
      
      report[list(models.keys())[i]] = test_model_score
    return report
  except Exception as e:
    raise NetworkSecurityException(e, sys)