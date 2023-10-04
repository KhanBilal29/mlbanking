import sys 
import dataclasses as dataclass 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            pass

        except Exception as e:
            raise CustomException(e, sys)
            
      