from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pickle
import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "House price prediction"
EXPERIMENT_URI = "http://localhost:5000"

class TrainingModel:
    def __init__(self, data, target_column, test_size=0.2, random_state=42):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.best_model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        
        mlflow.set_tracking_uri(EXPERIMENT_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    def split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_model(self):
        self.model = LinearRegression()
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring=make_scorer(r2_score))
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        
        
        with open("../models/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)
        logger.info("Best model saved to ../models/best_model.pkl")

    def load_model(self, model_path="../models/best_model.pkl"):
        with open(model_path, "rb") as f:
            self.best_model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")


    def evaluate_and_log_model(self):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        with mlflow.start_run():
            y_pred = self.best_model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            
            signature = infer_signature(self.X_train, self.best_model.predict(self.X_train))
            log_model(self.best_model, "linear_regression_model", signature=signature)
            
            logger.info(f"Model logged in MLflow with run ID: {mlflow.active_run().info.run_id}")
            logger.info(f"Evaluation metrics - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}") 
    
    def get_best_model(self):
        return self.best_model
    
    def configure_mlflow(self):
        mlflow.set_tracking_uri(uri= EXPERIMENT_URI)
        try:
            exp= mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if (exp is not None):
                mlflow.set_experiment(experiment_id=exp.experiment_id)
        except:
            exp_id = mlflow.create_experiment(name =EXPERIMENT_NAME)
            mlflow.set_experiment(experiment_id=exp_id)
        finally:
            return mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    def register_model(self, model_name):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        mlflow.register_model("runs:/{}/linear_regression_model".format(mlflow.active_run().info.run_id), model_name)
        print(f"Model registered as {model_name}")