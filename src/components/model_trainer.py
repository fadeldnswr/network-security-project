'''
Model Trainer Component
This component is responsible for training the model using the preprocessed data.
'''
import os, sys
import mlflow

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from src.utils.main_utils.utils import (
  save_object,
  load_object,
  load_numpy_array_data,
  evaluate_models
)
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
  RandomForestClassifier,
  GradientBoostingClassifier,
  AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score


class ModelTrainer:
  def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
    '''
    Model Trainer component constructor
    Initializes the model trainer with the necessary configurations.
    '''
    try:
      # Initialize the model trainer with configurations
      self.model_trainer_config = model_trainer_config
      self.data_transformation_artifact = data_transformation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def track_mlflow(self, best_model, classification_metric):
    '''
    Tracks the model and its metrics using MLFlow.
    :param model: The trained machine learning model
    :param classification_metric: The classification metrics of the model
    :raises NetworkSecurityException: If there is an error during tracking
    '''
    try:
      with mlflow.start_run():
        # Initialize the MLFlow run
        f1_score = classification_metric.f1_score
        precision_score = classification_metric.precision_score
        recall_score = classification_metric.recall_score
        
        # Log the model and its parameters
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("precision_score", precision_score)
        mlflow.log_metric("recall_score", recall_score)
        
        mlflow.sklearn.log_model(best_model, "model")
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def train_model(self, x_train, y_train, x_test, y_test):
    '''
    Trains the machine learning model using the training data.
    :param x_train: Training features
    :param y_train: Training labels
    :param x_test: Testing features
    :param y_test: Testing labels
    :return: Trained model
    '''
    try:
      # Initialize the models and their parameters
      models = {
        "Random Forest": RandomForestClassifier(verbose=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        "Logistic Regression": LogisticRegression(verbose=1),
        "AdaBoost": AdaBoostClassifier(),
      }
      params= {
        "Decision Tree": {
          'criterion':['gini', 'entropy', 'log_loss'],
          # 'splitter':['best','random'],
          # 'max_features':['sqrt','log2'],
        },
        "Random Forest":{
          # 'criterion':['gini', 'entropy', 'log_loss'],
          # 'max_features':['sqrt','log2',None],
          'n_estimators': [8,16,32,128,256]
        },
        "Gradient Boosting":{
          # 'loss':['log_loss', 'exponential'],
          'learning_rate':[.1,.01,.05,.001],
          'subsample':[0.6,0.7,0.75,0.85,0.9],
          # 'criterion':['squared_error', 'friedman_mse'],
          # 'max_features':['auto','sqrt','log2'],
          'n_estimators': [8,16,32,64,128,256]
        }, 
        "Logistic Regression":{},
        "AdaBoost":{
          'learning_rate':[.1,.01,.001],
          'n_estimators': [8,16,32,64,128,256]
        }
      }
      # Initialize the best model and its score
      model_report: dict = evaluate_models(
        X_train=x_train, y_train=y_train, 
        X_test=x_test, y_test=y_test,
        models=models, param=params
      )
      
      # Get the best model based on the report
      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      best_model = models[best_model_name]
      
      # Train the best model with the training data
      y_train_pred = best_model.predict(x_train)
      classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
      
      # Function to track MLFlow metrics
      self.track_mlflow(best_model, classification_train_metric)
      
      # Predict on the test set using the best model
      y_test_pred = best_model.predict(x_test)
      classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
      
      self.track_mlflow(best_model=best_model, classification_metric=classification_test_metric)
      
      # Load the preprocessor from the data transformation artifact
      preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
      model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
      os.makedirs(model_dir_path, exist_ok=True)
      
      # Create a NetworkModel instance with the preprocessor and the best model
      network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
      
      # Save the trained model using the save_object function
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=network_model
      )
      
      # Save the best model to a specific directory
      save_object("final_model/model.pkl", best_model)
      
      # ModelTrainerArtifact to store the trained model and metrics
      model_trainer_artifact = ModelTrainerArtifact(
        trained_model_file_path=self.model_trainer_config.trained_model_file_path,
        train_metric_artifact=classification_train_metric,
        test_metric_artifact=classification_test_metric,
      )
      logging.info(f"Model training completed successfully. Best model: {best_model_name} with score: {best_model_score}")
      return model_trainer_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
  
  def initiate_model_trainer(self) -> ModelTrainerArtifact:
    '''
    Initiates the model training process.
    Loads the preprocessed data, trains the model, and saves the trained model.
    :return: ModelTrainerArtifact containing the trained model and metrics
    '''
    try:
      train_file_path = self.data_transformation_artifact.transformed_train_file_path
      test_file_path = self.data_transformation_artifact.transformed_test_file_path
      
      # Load the preprocessed training and testing data
      train_arr = load_numpy_array_data(file_path=train_file_path)
      test_arr = load_numpy_array_data(file_path=test_file_path)
      
      # Split the data into features and labels
      x_train, y_train, x_test, y_test = (
        train_arr[:, :-1], train_arr[:, -1],
        test_arr[:, :-1], test_arr[:, -1]
      )
      
      # Create a model instance
      model = self.train_model(x_train, y_train, x_test, y_test)
      return model
    except Exception as e:
      raise NetworkSecurityException(e, sys)