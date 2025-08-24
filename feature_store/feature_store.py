import os
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
import pandas as pd
from datetime import datetime, timedelta


class FeastFeatureStore:
    def __init__(self, path):
        self.store = FeatureStore(repo_path=path)
        self.retrievalJob = None
        
