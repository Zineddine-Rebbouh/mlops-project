from typing import Dict, Any, Optional, Tuple
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator

import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingModel:
    """
    A class for training machine learning models with MLflow tracking and experiment management.
    
    This class handles data splitting, model training, evaluation, and MLflow integration
    for house price prediction using Linear Regression.
    """
    
    # Constants
    DEFAULT_EXPERIMENT_NAME = "House price prediction"
    DEFAULT_EXPERIMENT_URI = "http://localhost:5000"
    DEFAULT_MODEL_PATH = "./models/best_model.pkl"
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_CV_FOLDS = 5
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        test_size: float = DEFAULT_TEST_SIZE, 
        random_state: int = DEFAULT_RANDOM_STATE,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        experiment_uri: str = DEFAULT_EXPERIMENT_URI
    ):
        """
        Initialize the TrainingModel.
        
        Args:
            data: The dataset containing features and target
            target_column: Name of the target column
            test_size: Proportion of dataset to use for testing
            random_state: Random state for reproducibility
            experiment_name: Name of the MLflow experiment
            experiment_uri: URI of the MLflow tracking server
        """
        self._validate_inputs(data, target_column, test_size)
        
        self.data = data.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.experiment_name = experiment_name
        self.experiment_uri = experiment_uri
        
        # Model-related attributes
        self.model: Optional[BaseEstimator] = None
        self.best_model: Optional[BaseEstimator] = None
        self.grid_search: Optional[GridSearchCV] = None
        
        # Data splits
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
        # Initialize data splits
        self._split_data()
        
        # Configure MLflow
        self._initialize_mlflow()
    
    def _validate_inputs(self, data: pd.DataFrame, target_column: str, test_size: float) -> None:
        """Validate input parameters."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
    
    def _initialize_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.experiment_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            raise
    
    def _split_data(self) -> None:
        """Split the data into training and testing sets."""
        try:
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            logger.info(f"Data split - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise

    def train_model(self) -> None:
        """
        Train a Linear Regression model with GridSearchCV.
        
        Raises:
            RuntimeError: If training fails
        """
        try:
            logger.info("Starting model training...")
            
            self.model = LinearRegression()
            param_grid = {
                'fit_intercept': [True, False],
            }

            self.grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=self.DEFAULT_CV_FOLDS, 
                scoring=make_scorer(r2_score),
                n_jobs=-1
            )
            
            self.grid_search.fit(self.X_train, self.y_train)
            self.best_model = self.grid_search.best_estimator_
            
            logger.info(f"Training completed. Best params: {self.grid_search.best_params_}")
            logger.info(f"Best CV score: {self.grid_search.best_score_:.4f}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}")

    def _save_model(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """Save the best model to disk."""
        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, "wb") as f:
                pickle.dump(self.best_model, f)
            
            logger.info(f"Best model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, "rb") as f:
                self.best_model = pickle.load(f)
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        return self.best_model.predict(X)

    def _calculate_metrics(self, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

    def evaluate_and_log_model(self) -> Dict[str, float]:
        """
        Evaluate the model and log results to MLflow.
        
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        try:
            with mlflow.start_run():
                y_pred = self.predict(self.X_test)
                metrics = self._calculate_metrics(y_pred)
                
                # Log parameters and metrics
                mlflow.log_param("model_type", "LinearRegression")
                mlflow.log_param("test_size", self.test_size)
                mlflow.log_param("random_state", self.random_state)
                
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                signature = infer_signature(self.X_train, self.predict(self.X_train))
                log_model(self.best_model, "linear_regression_model", signature=signature)
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Model logged in MLflow with run ID: {run_id}")
                logger.info(f"Evaluation metrics: {metrics}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to evaluate and log model: {e}")
            raise
    
    def get_best_model(self) -> Optional[BaseEstimator]:
        """Get the best trained model."""
        return self.best_model
    
    def configure_mlflow(self) -> Any:
        """
        Configure MLflow experiment.
        
        Returns:
            MLflow experiment object
        """
        try:
            mlflow.set_tracking_uri(uri=self.experiment_uri)
            
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is not None:
                mlflow.set_experiment(experiment_id=exp.experiment_id)
            else:
                exp_id = mlflow.create_experiment(name=self.experiment_name)
                mlflow.set_experiment(experiment_id=exp_id)
            
            logger.info(f"MLflow experiment configured: {self.experiment_name}")
            return mlflow.get_experiment_by_name(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to configure MLflow: {e}")
            raise
    
    def _log_gridsearch_runs(self) -> None:
        """Log individual GridSearch runs to MLflow."""
        if self.grid_search is None:
            raise ValueError("GridSearch has not been performed yet.")
        
        for i, params in enumerate(self.grid_search.cv_results_["params"]):
            try:
                with mlflow.start_run(run_name=f"child_run_{i}", nested=True):
                    # Get metrics from cv_results_
                    mean_test_score = self.grid_search.cv_results_["mean_test_score"][i]
                    std_test_score = self.grid_search.cv_results_["std_test_score"][i]

                    # Log parameters and cross-validation metrics
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_cv_score", mean_test_score)
                    mlflow.log_metric("std_cv_score", std_test_score)

                    # Refit model with current parameters
                    temp_model = self.grid_search.estimator.set_params(**params)
                    temp_model.fit(self.X_train, self.y_train)
                    y_pred = temp_model.predict(self.X_test)
                    
                    # Log test metrics
                    test_metrics = self._calculate_metrics(y_pred)
                    mlflow.log_metrics(test_metrics)
                    mlflow.sklearn.log_model(temp_model, "model")

                    logger.info(f"Logged run {i} with params: {params}, "
                              f"mean_cv_score: {mean_test_score:.4f}, "
                              f"std_test_score: {std_test_score:.4f}")
                              
            except Exception as e:
                logger.warning(f"Failed to log run {i}: {e}")

    def register_model(self) -> Any:
        """
        Register the best model in MLflow with comprehensive logging.
        
        Returns:
            MLflow model info object
        """
        if self.best_model is None or self.grid_search is None:
            raise ValueError("Model must be trained before registration.")
        
        try:
            with mlflow.start_run(run_name="LinearReg_GridSearch_Best", log_system_metrics=True) as run:
                # Log the best parameters and metrics
                best_params = self.grid_search.best_params_
                best_score = self.grid_search.best_score_
                
                mlflow.log_params(best_params)
                mlflow.log_metric("best_mean_cv_score", best_score)
                
                # Calculate and log test metrics
                y_pred = self.predict(self.X_test)
                test_metrics = self._calculate_metrics(y_pred)
                mlflow.log_metrics(test_metrics)

                # Define signature
                signature = infer_signature(
                    np.array(self.X_train), 
                    np.array(self.predict(self.X_test))
                )

                # Log all GridSearch runs
                self._log_gridsearch_runs()

                # Log and Register best model
                model_info = log_model(
                    sk_model=self.best_model,
                    artifact_path="house_model",
                    signature=signature,
                    input_example=self.X_train.head(5),
                    registered_model_name="house_price_prediction",
                )
                
                logger.info(f"Model registered successfully with run ID: {run.info.run_id}")
                return model_info
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the trained model.
        
        Returns:
            Dictionary containing model summary information
        """
        if self.best_model is None:
            return {"status": "Model not trained"}
        
        summary = {
            "model_type": type(self.best_model).__name__,
            "training_samples": len(self.X_train) if self.X_train is not None else 0,
            "test_samples": len(self.X_test) if self.X_test is not None else 0,
            "features": list(self.X_train.columns) if self.X_train is not None else [],
            "target_column": self.target_column,
            "best_params": self.grid_search.best_params_ if self.grid_search else None,
            "best_cv_score": self.grid_search.best_score_ if self.grid_search else None
        }
        
        return summary