#strategy pattern
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis(ABC):
    @abstractmethod
    def analyze(self , df: pd.Dataframe  , features ):
        pass