import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidaton
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidaionConfig,
)
from src.entity.artifact_entity import DataIngestionArtifact, DataValidatinArtifact

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingesion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingesion_config)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initate_data_ingestion()
        logging.info("Data ingestion initiation completed")
        print(data_ingestion_artifact)
        data_validation_config = DataValidaionConfig(training_pipeline_config)
        data_validation = DataValidaton(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_arifact = data_validation.initiate_data_validation()
        logging.info("Data validation initiation completed")
        print(data_validation_arifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
