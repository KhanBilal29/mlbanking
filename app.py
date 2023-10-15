from src.ml_banking.logger import logging
from src.ml_banking.exception import CustomException
from src.ml_banking.components.data_ingestion import DataIngestion
from src.ml_banking.components.data_ingestion import DataIngestionConfig
from src.ml_banking.components.data_transformation import DataTransformation
from src.ml_banking.components.data_transformation import DataTransformationConfig
from src.ml_banking.components.model_tranier import ModelTrainerConfig,ModelTrainer
from sklearn.model_selection import GridSearchCV
import time


import sys

start_time = time.time()

if __name__=="__main__":
    logging.info("The execution has started")


    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr =data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)    
    

end_time = time.time()    

execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")




