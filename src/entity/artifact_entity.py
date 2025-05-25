from dataclasses import dataclass

# Data ingestion artifact class to store paths of ingested data files
@dataclass
class DataIngestionArtifact:
  trained_file_path:str
  test_file_path:str

# Data validation artifact class to store paths of validated data files and validation status
@dataclass
class DataValidationArtifact:
  validation_status: bool
  valid_train_file_path: str
  valid_test_file_path: str
  invalid_train_file_path: str
  invalid_test_file_path: str
  drift_report_file_path: str

# Data transformation artifact class to store paths of transformed data files
@dataclass
class DataTransformationArtifact:
  transformed_object_file_path: str
  transformed_train_file_path: str
  transformed_test_file_path: str

# Classification metric artifact class to store model evaluation metrics
@dataclass
class ClassificationMetricArtifact:
  f1_score: float
  precision_score: float
  recall_score: float

# Model trainer artifact class to store paths of trained model files
@dataclass
class ModelTrainerArtifact:
  trained_model_file_path: str
  train_metric_artifact: ClassificationMetricArtifact
  test_metric_artifact: ClassificationMetricArtifact