#strategy pattern
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalysis(ABC):
    @abstractmethod
    def analyze(self , df: pd.Dataframe  , features ):
        pass

class CorrelationAnalysis(DataAnalysis):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Analyze correlations between numerical features."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"correlations": {}, "error": "Empty DataFrame"}
            
            logger.info("Performing correlation analysis")
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) < 2:
                return {"correlations": {}, "warning": "Less than 2 numerical columns for correlation"}
            
            corr_matrix = df[numerical_cols].corr(method='pearson').to_dict()
            return {"correlations": corr_matrix}
        except Exception as e:
            logger.error(f"Error in CorrelationAnalysis: {str(e)}")
            return {"error": str(e)}

class CardinalityAnalysis(DataAnalysis):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Analyze cardinality of categorical columns."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"cardinality": {}, "error": "Empty DataFrame"}
            
            logger.info("Analyzing cardinality of categorical columns")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            result = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                unique_values = df[col].unique().tolist()[:10]  # Limit to first 10 for brevity
                result[col] = {
                    "unique_count": unique_count,
                    "unique_values_sample": unique_values,
                    "high_cardinality": unique_count > 0.05 * len(df)  # Threshold: 5% of rows
                }
            return {"cardinality": result}
        except Exception as e:
            logger.error(f"Error in CardinalityAnalysis: {str(e)}")
            return {"error": str(e)}    