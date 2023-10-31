# prediction_pipeline.py
import sys
import pandas as pd
from src.ml_banking.exception import CustomException
from src.ml_banking.utils import load_object

class PredictPipeline:
    def __init__(self, model_path, preprocessor_path):
        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)

    def predict(self):
        try:
            # Accept input from the user
            gender = input("Enter gender: ")
            race_ethnicity = input("Enter race/ethnicity: ")
            parental_level_of_education = input("Enter parental level of education: ")
            lunch = input("Enter lunch type: ")
            test_preparation_course = input("Enter test preparation course: ")
            reading_score = int(input("Enter reading score: "))
            writing_score = int(input("Enter writing score: "))

            # Create a CustomData instance with user input
            custom_data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            input_data = custom_data.get_data_as_data_frame()

            input_features_transformed = self.preprocessor.transform(input_data)
            predictions = self.model.predict(input_features_transformed)
            return predictions
        except Exception as e:
            raise CustomException(str(e))

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(str(e))

if __name__ == "__main__":
    model_path = "path/to/your/model.pkl"
    preprocessor_path = "path/to/your/preprocessor.pkl"

    # Make predictions using the PredictPipeline
    prediction_pipeline = PredictPipeline(model_path, preprocessor_path)
    predictions = prediction_pipeline.predict()
    print("Predictions:", predictions)
