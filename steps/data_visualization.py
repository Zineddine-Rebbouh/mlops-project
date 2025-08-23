from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import numpy as np
from steps.data_ingestion import *
    

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataVisualization(ABC):
    @abstractmethod
    def visualize(self, df: pd.DataFrame, inspection_results: dict = None) -> None:
        pass

class MissingValuesVisualization(DataVisualization):
    def visualize(self, df: pd.DataFrame, inspection_results: dict = None) -> None:
        try:
            if df.empty or not inspection_results:
                logger.warning("DataFrame or inspection results are empty.")
                return
            
            logger.info("Visualizing missing values")
            missing_data = inspection_results.get('MissingValues', {}).get('missing_counts', {})
            if not missing_data:
                logger.warning("No missing values data to visualize.")
                return
            
            plt.figure(figsize=(10, 6))
            plt.bar(missing_data.keys(), missing_data.values(), color='#1f77b4')
            plt.title('Missing Values by Column')
            plt.xlabel('Columns')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('missing_values.png')
            plt.close()
            logger.info("Missing values plot saved as 'missing_values.png'")
        except Exception as e:
            logger.error(f"Error in MissingValuesVisualization: {str(e)}")

class DistributionVisualization(DataVisualization):
    def visualize(self, df: pd.DataFrame, inspection_results: dict = None) -> None:
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return
            
            logger.info("Visualizing data distributions")
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if not numerical_cols:
                logger.warning("No numerical columns to visualize.")
                return
            
            for col in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[col].dropna(), kde=True, color='#1f77b4')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(f'distribution_{col}.png')
                plt.close()
                logger.info(f"Distribution plot for {col} saved as 'distribution_{col}.png'")
        except Exception as e:
            logger.error(f"Error in DistributionVisualization: {str(e)}")

class CorrelationHeatmapVisualization(DataVisualization):
    def visualize(self, df: pd.DataFrame, inspection_results: dict = None) -> None:
        try:
            if df.empty or not inspection_results:
                logger.warning("DataFrame or inspection results are empty.")
                return
            
            logger.info("Visualizing correlation matrix")
            corr_data = inspection_results.get('CorrelationAnalysis', {}).get('correlations', {})
            if not corr_data:
                logger.warning("No correlation data to visualize.")
                return
            
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) < 2:
                logger.warning("Less than 2 numerical columns for correlation visualization.")
                return
            
            corr_matrix = pd.DataFrame(corr_data)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png')
            plt.close()
            logger.info("Correlation heatmap saved as 'correlation_heatmap.png'")
        except Exception as e:
            logger.error(f"Error in CorrelationHeatmapVisualization: {str(e)}")

class DataVisualizer:
    def __init__(self, strategies: list[DataVisualization] = None):
        self.strategies = strategies or [
            MissingValuesVisualization(),
            DistributionVisualization(),
            CorrelationHeatmapVisualization()
        ]

    def add_strategy(self, strategy: DataVisualization):
        """Add a new visualization strategy."""
        self.strategies.append(strategy)

    def execute_all_visualizations(self, df: pd.DataFrame, inspection_results: dict = None) -> None:
        """Execute all visualization strategies."""
        try:
            if not isinstance(df, pd.DataFrame):
                logger.error("Input must be a pandas DataFrame")
                return
            
            logger.info("Executing all visualization strategies")
            for strategy in self.strategies:
                strategy_name = strategy.__class__.__name__
                logger.info(f"Running visualization: {strategy_name}")
                strategy.visualize(df, inspection_results)
        except Exception as e:
            logger.error(f"Error in DataVisualizer: {str(e)}")

# Example usage
if __name__ == "__main__":
    
    file_path = "C:\\Users\\mkrym\\Downloads\\archive.zip"

    Ingestor = DataIngestorFactory.get_data_ingestor(file_path)   
    df = Ingestor.ingest(file_path)
    
    # Sample inspection results (in practice, these would come from data_inspection.py)
    inspection_results = {
        'MissingValues': {
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        },
        'CorrelationAnalysis': {
            'correlations': df.select_dtypes(include=['float64', 'int64']).corr().to_dict()
        }
    }
    
    # Initialize visualizer and execute
    visualizer = DataVisualizer()
    visualizer.execute_all_visualizations(df, inspection_results)