import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransfomation
from src.components.model_trainer import ModelTrainer
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidatinArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingesion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingesion_config)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initate_data_ingestion()
        logging.info("Data ingestion initiation completed")
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation initiation completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransfomation(
            data_validation_artifact, data_transformation_config
        )
        logging.info("Initiate the data transformation")
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        logging.info("Data transformation initiation completed")
        print(data_transformation_artifact)

        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        logging.info("Initiate the model trainer")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training initiation completed")
        print(model_trainer_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
