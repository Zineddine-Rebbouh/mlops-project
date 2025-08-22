from abc import ABC , abstractmethod 
import pandas as pd


class DataInspection(ABC):
    @abstractmethod
    def inspect(self , df):
        pass

class DataTypeInspection(DataInspection):
    def inspect(self, df):
        print("Data types Analysis:")
        print(df.dtypes)

class SummaryStatistics(DataInspection):
    def inspect(self, df):
        print("Summary Statistics for numerical features:")
        print(df.describe().transpose())
        print("Summary for categorical features :")
        print(df.describe(include=['object']))

class MissingValues(DataInspection):
    def inspect(self, df):
        print("Missing Values Analysis:")
        print(df.isnull().sum())
        

# class OutlierDetection(DataInspection):
#     def inspect(self, df):
#         pass
 

class DataInspector:
    @staticmethod
    def __init__(self , strategy : DataTypeInspection): 
        self.strategy = strategy
        
    def set_strategy(self, strategy: DataInspection):
        self.strategy = strategy
    
    def execute_strategy(self , df:pd.DataFrame):
        self.strategy.inspect(df)
        