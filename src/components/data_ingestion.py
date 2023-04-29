import os 
import sys 
from src.exception import CustomException 
from src.logger import logging 
from datasets import load_dataset 
import pandas as pd

from dataclasses import dataclass 
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = str=os.path.join('artifacts', "train_split.csv")
    test_data_path = str=os.path.join('artifacts', "test_split.csv")
    validation_data_path = str=os.path.join('artifacts', "validation_split.csv") 
    raw_data_path = str=os.path.join('artifacts', "raw_data.csv")

class DataIngestion: 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\emotion_dataset.csv') 
            logging.info('Read the dataset as DataFrame') 

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) 

            logging.info("Train, Test and Validation data is initiated")

            train_set = pd.read_csv('notebook\data\\train_split.csv')
            test_set = pd.read_csv('notebook\data\\test_split.csv')
            validation_set = pd.read_csv('notebook\data\\validation_split.csv') 

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True) 

            logging.info("Data Ingestion is completed!") 

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path, 
                self.ingestion_config.validation_data_path
            )
        except Exception as e: 
            raise CustomException(e,sys) 

if __name__=="__main__":
    
    obj=DataIngestion()
    train_path, test_path, validation_path = obj.initiate_data_ingestion() 
    logging.info("Data ingestion is done!")

    data_transform = DataTransformation()
    train_transformed, test_transformed, validation_transformed = data_transform.convert_to_tensor(train_path, test_path, validation_path)
    logging.info("Data Transformation is done!")

    model = ModelTrainer()
    model.train_model(train_transformed, test_transformed) 
    logging.info(f"Model Training Completed")
