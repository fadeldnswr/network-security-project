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