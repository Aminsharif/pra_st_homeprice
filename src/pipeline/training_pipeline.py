import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer


if __name__ == "__main__":
    ob = DataIngestion()
    train_data_path, test_data_path = ob.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)