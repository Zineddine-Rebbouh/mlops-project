# strategy pattern
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any  # safer typing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, analyze_results: dict = None, features: list = None) -> Dict[str, Any]:
        """Abstract method for data analysis strategies"""
        pass

class CorrelationAnalysis(DataAnalysis):
    def analyze(self, df: pd.DataFrame, analyze_results: dict = None, features: list = None) -> dict:
        """Analyze correlations between numerical features."""
        try:
            if df is None or df.empty:
                logger.warning("DataFrame is empty or None.")
                return {"correlations": {}, "error": "Empty DataFrame"}
            
            logger.info("Performing correlation analysis")
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) < 2:
                logger.warning("Less than 2 numerical columns for correlation analysis.")
                return {"correlations": {}, "warning": "Less than 2 numerical columns"}
            
            corr_matrix = df[numerical_cols].corr(method='pearson').to_dict()
            return {"correlations": corr_matrix}
        except Exception as e:
            logger.error(f"Error in CorrelationAnalysis: {str(e)}")
            return {"error": str(e)}

class CardinalityAnalysis(DataAnalysis):
    def analyze(self, df: pd.DataFrame, analyze_results: dict = None, features: list = None) -> dict:
        """Analyze cardinality of categorical columns."""
        try:
            if df is None or df.empty:
                logger.warning("DataFrame is empty or None.")
                return {"cardinality": {}, "error": "Empty DataFrame"}
            
            logger.info("Analyzing cardinality of categorical columns")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            result = {}
            for col in categorical_cols:
                unique_count = df[col].nunique(dropna=True)
                unique_values = df[col].dropna().unique().tolist()[:10]  # Limit to first 10 for brevity
                result[col] = {
                    "unique_count": unique_count,
                    "unique_values_sample": unique_values,
                    "high_cardinality": unique_count > 0.05 * len(df)  # Threshold: 5% of rows
                }
            return {"cardinality": result}
        except Exception as e:
            logger.error(f"Error in CardinalityAnalysis: {str(e)}")
            return {"error": str(e)}

class DataAnalyzer:
    def __init__(self, strategies: List[DataAnalysis] = None):
        self.strategies = strategies or [
            CorrelationAnalysis(),
            CardinalityAnalysis()
        ]

    def set_strategy(self, strategies: List[DataAnalysis]):
        """Replace existing strategies with new ones"""
        self.strategies = strategies

    def execute_all_strategies(self, df: pd.DataFrame, analyze_results: dict = None) -> dict:
        """Run all analysis strategies and return combined results"""
        results = {}
        if df is None or df.empty:
            logger.warning("DataFrame is empty or None. Skipping analysis.")
            return results

        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            logger.info(f"Using strategy: {strategy_name}")
            result = strategy.analyze(df, analyze_results)
            results[strategy_name] = result
        return results
