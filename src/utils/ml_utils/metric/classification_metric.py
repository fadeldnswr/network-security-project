'''
Classification metrics for evaluating model performance.
'''
import os, sys

from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception.exception import NetworkSecurityException
from sklearn.metrics import (
  f1_score,
  accuracy_score,
  precision_score,
  recall_score,
)

# get_classification_metrics function created to calculate classification metrics
def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
  '''
  Classification metrics for evaluating model performance.
  :param y_true: True labels
  :param y_pred: Predicted labels
  :return: ClassificationMetricArtifact containing the calculated metrics
  '''
  try:
    # Calculate classification metrics
    model_f1_score = f1_score(y_true, y_pred)
    model_recall_sscore = recall_score(y_true, y_pred)
    model_precision_score = precision_score(y_true, y_pred)
    
    # Create and return the ClassificationMetricArtifact
    classification_metric = ClassificationMetricArtifact(
      f1_score=model_f1_score,
      recall_score=model_recall_sscore,
      precision_score=model_precision_score,
    )
    return classification_metric
  except Exception as e:
    raise NetworkSecurityException(e, sys)