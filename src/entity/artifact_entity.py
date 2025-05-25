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