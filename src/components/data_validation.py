import os, sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact, DataValidatinArtifact
from src.entity.config_entity import DataValidaionConfig
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd


class DataValidaton:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidaionConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column: {
                            "p_value": float(is_same_dist.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(drift_report_file_path, report)
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataIngestionArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_pth = self.data_ingestion_artifact.test_file_path

            # read the data from train and test
            train_dataframe = DataValidaton.read_data(train_file_path)
            test_dataframe = DataValidaton.read_data(test_file_pth)

            # validate number of columns
            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                raise Exception("Train dataframe doesn't contain all columns.")
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                raise Exception("Test dataframe doesn't contain all columns.")

            train_numerical = train_dataframe.select_dtypes(include=["number"])
            test_numerical = test_dataframe.select_dtypes(include=["number"])

            if train_numerical.shape[1] == 0:
                raise Exception("No numerical columns found in train dataset.")
            if test_numerical.shape[1] == 0:
                raise Exception("No numerical columns found in test dataset.")

            # check datadrift
            status = self.detect_dataset_drift(train_dataframe, test_dataframe)
            if status:
                dir_path = os.path.dirname(
                    self.data_validation_config.valid_train_file_path
                )
                os.makedirs(dir_path, exist_ok=True)
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path,
                    index=False,
                    header=True,
                )
                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path,
                    index=False,
                    header=True,
                )
            else:
                dir_path = os.path.dirname(
                    self.data_validation_config.invalid_train_file_path
                )
                os.makedirs(dir_path, exist_ok=True)
                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path,
                    index=False,
                    header=True,
                )
                test_dataframe.to_csv(
                    self.data_validation_config.invalid_test_file_path,
                    index=False,
                    header=True,
                )

            data_validation_aritfact = DataValidatinArtifact(
                status,
                self.data_ingestion_artifact.train_file_path,
                self.data_ingestion_artifact.test_file_path,
                None,
                None,
                self.data_validation_config.drift_report_file_path,
            )
            return data_validation_aritfact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
