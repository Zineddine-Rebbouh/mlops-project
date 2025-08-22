from abc import ABC, abstractmethod
import os
import pandas as pd
import zipfile

#Factory design pattern for data ingestion

class DataIngestion(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass
    
class ZipFileIngestion(DataIngestion):
    def ingest(self , file_path:str ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            return ValueError(f"File {file_path} does not exist.")
        
        if not file_path.endswith('.zip'):
            return ValueError(f"File {file_path} is not a zip file.")
        
        with zipfile.ZipFile(file_path, 'r') as files:
            files.extractall("data")

        extracted_files = os.listdir("data")
        csv_files = [files for files in extracted_files if files.endswith('.csv')]
        
        if len(csv_files) == 0:
            return ValueError("No CSV files found in the zip archive.")
        elif len(csv_files) > 1:
            return ValueError("Multiple CSV files found in the zip archive.Please provide a zip file with a single CSV file.")
        
        path = os.path.join("data" , csv_files[0])
        return pd.read_csv(path)

class CSVFileIngestion(DataIngestion):
    def ingest(self, file_path):
        
        if not os.path.exists(file_path):
            return ValueError(f"File {file_path} does not exist.")

        if not file_path.endswith('.csv'):
            return ValueError(f"File {file_path} is not a csv file.")

        return pd.read_csv(file_path)
    
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestion:
        if file_path.endswith('.zip'):
            return ZipFileIngestion()
        elif file_path.endswith('.csv'):
            return CSVFileIngestion()
        else:
            raise ValueError(f"Unsupported file type for {file_path}. Only .zip and .csv files are supported.")