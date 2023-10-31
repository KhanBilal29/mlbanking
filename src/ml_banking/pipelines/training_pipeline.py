import os
import sys
import pandas as pd
from src.ml_banking.exception import CustomException
from src.ml_banking.utils import load_object
from src.ml_banking.components.data_transformation import DataTransformation

class TrainingPipeline:
    def __init__(self, model_path, preprocessor_path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def train_model(self, train_data_path, test_data_path):
        try:
            model_trainer = DataTransformation(self.model_path, self.preprocessor_path)
            train_arr, test_arr = model_trainer.initiate_data_transformation(train_data_path, test_data_path)
            accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(f"Model trained successfully with accuracy: {accuracy}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    model_path = "path/to/your/model.pkl"
    preprocessor_path = "path/to/your/preprocessor.pkl"

    training_pipeline = TrainingPipeline(model_path, preprocessor_path)
    training_pipeline.train_model(train_data_path, test_data_path)
