import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.util import save_object
from src.util import evaluate_model


from dataclasses import dataclass
import sys
import os 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array, test_array):
        try:

            X_train, y_train, X_test, y_test = (
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
            models = {
                "LinearRegression":LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
            }
        
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n==========================================================\n')
            logging.info(f'Model Report: {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)   
                ]
            
            best_model = models[best_model_name]

            print(f"Best modle found, Model name: {best_model_name}")
            print('\n==========================================================\n')
            logging.info(f'Best modle found, Model name: {best_model_name}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

        except Exception as e:
            logging.info('Exception occure in model training')
            raise Exception(e, sys)