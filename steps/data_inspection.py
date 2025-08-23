from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from scipy import stats
from datetime import datetime
from steps.data_ingestion import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataInspection(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> dict:
        """Abstract method to inspect a DataFrame and return results as a dictionary."""
        pass

class DataTypeInspection(DataInspection):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Inspect data types of each column."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"data_types": {}, "error": "Empty DataFrame"}
            
            logger.info("Performing data type inspection")
            dtypes = df.dtypes.to_dict()
            type_counts = df.dtypes.value_counts().to_dict()
            
            return {
                "data_types": {col: str(dtype) for col, dtype in dtypes.items()},
                "type_summary": {str(k): v for k, v in type_counts.items()}
            }
        except Exception as e:
            logger.error(f"Error in DataTypeInspection: {str(e)}")
            return {"error": str(e)}

class SummaryStatistics(DataInspection):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for numerical and categorical features."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"numerical_summary": {}, "categorical_summary": {}, "error": "Empty DataFrame"}
            
            logger.info("Generating summary statistics")
            numerical_summary = df.describe().transpose().to_dict()
            categorical_summary = df.select_dtypes(include=['object', 'category']).describe().transpose().to_dict()
            
            return {
                "numerical_summary": numerical_summary,
                "categorical_summary": categorical_summary
            }
        except Exception as e:
            logger.error(f"Error in SummaryStatistics: {str(e)}")
            return {"error": str(e)}

class MissingValues(DataInspection):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Analyze missing values in the DataFrame."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"missing_values": {}, "error": "Empty DataFrame"}
            
            logger.info("Analyzing missing values")
            missing = df.isnull().sum().to_dict()
            missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
            
            return {
                "missing_counts": missing,
                "missing_percentage": {col: f"{perc:.2f}%" for col, perc in missing_percentage.items()}
            }
        except Exception as e:
            logger.error(f"Error in MissingValues: {str(e)}")
            return {"error": str(e)}

class OutlierDetection(DataInspection):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Detect outliers in numerical columns using IQR method."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"outliers": {}, "error": "Empty DataFrame"}
            
            logger.info("Detecting outliers")
            result = {}
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                result[col] = {
                    "outlier_count": len(outliers),
                    "outlier_values": outliers.tolist(),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }
            return {"outliers": result}
        except Exception as e:
            logger.error(f"Error in OutlierDetection: {str(e)}")
            return {"error": str(e)}

class DuplicateDetection(DataInspection):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Check for duplicate rows in the DataFrame."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"duplicates": {}, "error": "Empty DataFrame"}
            
            logger.info("Checking for duplicates")
            duplicate_count = df.duplicated().sum()
            duplicate_rows = df[df.duplicated(keep=False)].to_dict()
            
            return {
                "duplicate_count": int(duplicate_count),
                "duplicate_rows": duplicate_rows
            }
        except Exception as e:
            logger.error(f"Error in DuplicateDetection: {str(e)}")
            return {"error": str(e)}

class CorrelationAnalysis(DataInspection):
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

class CardinalityAnalysis(DataInspection):
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

class DataDistribution(DataInspection):
    def inspect(self, df: pd.DataFrame) -> dict:
        """Analyze statistical properties of numerical columns."""
        try:
            if df.empty:
                logger.warning("DataFrame is empty.")
                return {"distributions": {}, "error": "Empty DataFrame"}
            
            logger.info("Analyzing data distributions")
            result = {}
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numerical_cols:
                skewness = stats.skew(df[col].dropna())
                kurtosis = stats.kurtosis(df[col].dropna())
                result[col] = {
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "is_normal": abs(skewness) < 0.5 and abs(kurtosis) < 0.5
                }
            
            return {"distributions": result}
        except Exception as e:
            logger.error(f"Error in DataDistribution: {str(e)}")
            return {"error": str(e)}

class DataInspector:
    def __init__(self, strategies: list[DataInspection] = None):
        """Initialize with a list of inspection strategies."""
        self.strategies = strategies or [
            DataTypeInspection(),
            SummaryStatistics(),
            MissingValues(),
            OutlierDetection(),
            DuplicateDetection(),
            CorrelationAnalysis(),
            CardinalityAnalysis(),
            DataDistribution()
        ]
        self.results = {}

    def add_strategy(self, strategy: DataInspection):
        """Add a new inspection strategy."""
        self.strategies.append(strategy)

    def execute_all_strategies(self, df: pd.DataFrame) -> dict:
        """Execute all inspection strategies and compile results."""
        try:
            if not isinstance(df, pd.DataFrame):
                logger.error("Input must be a pandas DataFrame")
                return {"error": "Invalid input: must be a pandas DataFrame"}
            
            logger.info("Executing all inspection strategies")
            self.results = {}
            for strategy in self.strategies:
                strategy_name = strategy.__class__.__name__
                self.results[strategy_name] = strategy.inspect(df)
            
            return self.results
        except Exception as e:
            logger.error(f"Error in DataInspector: {str(e)}")
            return {"error": str(e)}

    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive report of inspection results."""
        try:
            logger.info("Generating inspection report")
            report = f"Data Inspection Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += "=" * 80 + "\n\n"
            
            for strategy_name, result in self.results.items():
                report += f"{strategy_name.replace('Inspection', '')}:\n"
                report += "-" * 50 + "\n"
                if "error" in result:
                    report += f"Error: {result['error']}\n\n"
                    continue
                
                for key, value in result.items():
                    report += f"{key.replace('_', ' ').title()}:\n"
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            report += f"  {sub_key}: {sub_value}\n"
                    else:
                        report += f"  {value}\n"
                report += "\n"
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_file}")
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"

# Example usage
if __name__ == "__main__":
    
    file_path = "C:\\Users\\mkrym\\Downloads\\archive.zip"

    Ingestor = DataIngestorFactory.get_data_ingestor(file_path)   
    df = Ingestor.ingest(file_path)
    
    # Initialize inspector and execute
    inspector = DataInspector()
    results = inspector.execute_all_strategies(df)
    
    # Generate and print report
    report = inspector.generate_report("inspection_report.txt")
    print(report)