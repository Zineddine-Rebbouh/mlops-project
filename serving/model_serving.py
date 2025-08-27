import bentoml
import pandas as pd

class BentoModel:
    def __init__(self):
        self.model_name = None
        self._model = None  # cache the loaded model

    def import_model(self, name: str, model_uri: str):
        """Import an MLflow model into BentoML."""
        model = bentoml.mlflow.import_model(name, model_uri)
        self.model_name = f"{model.tag.name}:{model.tag.version}"
        return self.model_name

    def load_model(self):
        """Load the BentoML model into memory."""
        if self.model_name is None:
            raise ValueError("Model name is not set. Please import a model first.")
        if self._model is None:  # only load once
            self._model = bentoml.mlflow.load_model(self.model_name)
        return self._model

    def predict(self, input_data):
        """Run predictions using the model."""
        model = self.load_model()
        # MLflow pyfunc models expect a pandas DataFrame
        if not isinstance(input_data, (pd.DataFrame, pd.Series)):
            input_data = pd.DataFrame(input_data)
        prediction = model.predict(input_data)
        return prediction
