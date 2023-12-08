import time
import sys
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from src.ml_banking.components.data_ingestion import DataIngestion
from src.ml_banking.components.data_transformation import DataTransformation
from src.ml_banking.components.model_tranier import ModelTrainer
from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
from src.ml_banking.pipelines.prediction_pipeline import CustomData, PredictPipeline
from pydantic import BaseModel

class MyModel(BaseModel):
    my_model_alias: str = 'default_value'  # No conflict now

application = Flask(__name__)
app = application

# ... (Your existing routes and functions)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for handling autocomplete requests and form submissions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # If the request method is GET, render the home page
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get data from the form using Pydantic
            data = CustomData(
                TRANSACTION_ID=request.form.get('TRANSACTION_ID'),
                TX_DATETIME=request.form.get('TX_DATETIME'),
                CUSTOMER_ID=request.form.get('CUSTOMER_ID'),
                TERMINAL_ID=request.form.get('TERMINAL_ID'),
                TX_AMOUNT=request.form.get('TX_AMOUNT'),
                TX_TIME_SECONDS=request.form.get('TX_TIME_SECONDS'),
                TX_TIME_DAYS=request.form.get('TX_TIME_DAYS'),
                TX_FRAUD_SCENARIO=request.form.get('TX_FRAUD_SCENARIO')
            )

            # Convert data to a DataFrame
            new_data = data.get_data_as_data_frame()

            # Use the prediction pipeline to get results
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(new_data)

            # Render the home page with the results
            return render_template('home.html', results=results[0])

        except Exception as e:
            # Log the exception and render an error page
            logging.exception("Error in prediction")
            return render_template('error.html', error_message=str(e))


# Add the block to measure execution time and log details
if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    logging.info("The execution has started")  # Log the start of execution

    try:
        # Initialize DataIngestion, perform data ingestion, and get paths
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Initialize DataTransformation, perform data transformation, and get arrays
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Initialize ModelTrainer, perform model training, and print the results
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

        # Optionally, you might have model monitoring here
        # model_monitoring = ModelMonitoring()
        # model_monitoring.monitor_model()

    except Exception as e:
        # Log any exceptions and raise a CustomException
        logging.exception("Custom Exception")
        raise CustomException(e, sys)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"Execution Time: {execution_time} seconds")  # Print the execution time

    app.run(host="0.0.0.0")