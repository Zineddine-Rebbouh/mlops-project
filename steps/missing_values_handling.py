import logging
from abc import ABC, abstractmethod
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingValuesHandler(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DropMissingValues(MissingValuesHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values using Drop Missing Values.")
        # Implement dropping logic here
        return df.dropna()

class FillingMissingValuesWithMean(MissingValuesHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values using Filling with Mean.")
        # Implement filling logic here
        return df.fillna(df.mean())

class FillingMissingValuesWithMedian(MissingValuesHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values using Filling with Median.")
        # Implement filling logic here
        return df.fillna(df.median())

class FillingMissingValuesWithMode(MissingValuesHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values using Filling with Mode.")
        # Implement filling logic here
        return df.fillna(df.mode().iloc[0])

class FillingMissingValuesWithCustomValue(MissingValuesHandler):
    def __init__(self, fill_value):
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Handling missing values using Filling with Custom Value: {self.fill_value}.")
        # Implement filling logic here
        return df.fillna(self.fill_value)

class MissingValuesProcessor:
    def __init__(self, strategy: MissingValuesHandler):
        self.strategy = strategy
        
    def set_strategy(self, strategy: MissingValuesHandler):
        self.strategy = strategy
    
    def execute(self, df):
        return self.strategy.handle(df)