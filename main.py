import sys

from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion(DataIngestionConfig(TrainingPipelineConfig()))
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)