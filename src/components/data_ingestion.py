import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_tranformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass ## if only defining variable then we can use data class
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str =os.path.join('artifacts','test.csv')
    raw_data_path : str =os.path.join('artifacts','raw.csv')

    
class DataIngestion: 

    def  __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            # reading dataset from semewhere (may be local, database, mongobd etc)
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('read the dataset as dataframe')

            # creating directory for train data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # saving data to raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('TRain test split initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed')

            # returning train and test set , coz next step will be data tranformation and 
            # and these data will be used in that 
            return (self.ingestion_config.train_data_path,
                     self.ingestion_config.test_data_path
                    )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    modeltrainer = ModelTrainer()
    r2_sqaure = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(r2_sqaure)