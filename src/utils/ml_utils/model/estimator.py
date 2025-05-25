'''
Estimator class for machine learning models.
'''
import os, sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.constant.training_pipeline import (
  SAVED_MODEL_DIR,
  MODEL_FILE_NAME
)

class NetworkModel:
  def __init__(self, preprocessor, model):
    '''
    Initialize the NetworkModel with a machine learning model.
    :param model: The machine learning model to be used
    '''
    try:
      self.preprocessor = preprocessor
      self.model = model
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def predict(self, x):
    '''
    Predict the output using the model.
    :param x: Input data for prediction
    :return: Predicted output
    '''
    try:
      x_transform = self.preprocessor.transform(x)
      y_hat = self.model.predict(x_transform)
      return y_hat
    except Exception as e:
      raise NetworkSecurityException(e, sys)