import logging
from abc import ABC, abstractmethod 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataEncoding(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

class BinaryCustomEncoding(DataEncoding):
    def encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logger.info("Applying Binary Custom Encoding.")
        for column in columns:
            df[column] = df[column].apply(lambda x: 1 if str(x).lower() in ["yes", "1", "true"] else 0)
        return df

class OneHotEncoding(DataEncoding):
    def encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logger.info("Applying One Hot Encoding.")
        return pd.get_dummies(df, columns=columns, drop_first=False)   # keep all categories

class LabelEncoding(DataEncoding):
    def encode(self, df: pd.DataFrame , columns: list) -> pd.DataFrame:
        logger.info("Applying Label Encoding.")
        le = LabelEncoder()
        for column in columns:
            df[column] = le.fit_transform(df[column])
        return df

class NumericalScaling(DataEncoding):
    def encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logger.info("Applying Numerical Scaling.")
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

class DataEncoderFactory:
    @staticmethod
    def get_encoder(encoding_type: str) -> DataEncoding:
        if encoding_type == "binary_custom":
            return BinaryCustomEncoding()
        elif encoding_type == "one_hot":
            return OneHotEncoding()
        elif encoding_type == "label":
            return LabelEncoding()
        elif encoding_type == "numerical":
            return NumericalScaling()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
