import os
import sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from src.utils.ml_utils.model.estimator import NetworkModel
from src.utils.ml_utils.metric.classificaiton_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_arifact = data_transformation_artifact
        except Exception(e) as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metrics):
        with mlflow.start_run():
            f1_score = classification_metrics.f1_score
            precision_score = classification_metrics.precision_score
            recall_score = classification_metrics.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        models = {
            "Logistic Regression": LogisticRegression(verbose=1),
            "KNeighbors Classifier": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(verbose=1),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Adaboost": AdaBoostClassifier(),
        }
        params = {
            "Logistic Regression": {},
            "KNeighbors Classifier": {
                "n_neighbors": [3, 4, 5, 6, 7, 8],
                "algorithm": ["auto", "ball_tree", "kd_tree"],
            },
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_features": ["sqrt", "log2"],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_features": ["sqrt", "log2", None],
            },
            "Gradient Boosting": {
                "loss": ["log_loss", "exponential"],
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "criterion": ["squred_error", "friedman_mse"],
                "max_features": ["auto", "sqrt", "log2"],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Adaboost": {
                "learning_rate": [0.1, 0.5, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

        model_report: dict = evaluate_models(
            X_train, y_train, X_test, y_test, models, params
        )

        logging.info(model_report)

        # to get the best model score from dict
        best_model_score = max(model_report.values())

        # to get the best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        logging.info(f"best model name: {best_model_name}")

        best_model = models[best_model_name]

        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_train, y_train_pred)

        ## track the experiments with mlflow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_test, y_test_pred)

        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(
            self.data_transformation_arifact.transformed_object_file_path
        )

        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor, best_model)
        save_object(self.model_trainer_config.trained_model_file_path, network_model)

        save_object("final_model/model.pkl", best_model)

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            self.model_trainer_config.trained_model_file_path,
            classification_train_metric,
            classification_test_metric,
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_arifact.transformed_train_file_path
            )
            test_file_path = self.data_transformation_arifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
