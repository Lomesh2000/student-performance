## here we will tranform data, feature engineering , handle continous variable, categorical data
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer # tranforamtion in a pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig: ## for giving input to the data transformation component
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_object(self):
        # this funtion is responsible for data transforamtion based on the different types of data

        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 
                                    'race_ethnicity', 
                                    'parental_level_of_education', 
                                    'lunch',
                                    'test_preparation_course']
            
            ## creating pipeline which will do two things
            ## 1. handle missing values
            ## 2. perform standar scaling
            ## this pipeline will run on the training dataset 
            numerical_pipepline = Pipeline(
                                steps=[
                                    ('imputer', SimpleImputer(strategy="median")),
                                    ('scaler', StandardScaler())
                                ]

                                )
            logging.info(f'NUmeical columns {numerical_columns} ..... scaling completed')

            categorical_pipeline = Pipeline(
                                steps = [
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('one_hot_encoder', OneHotEncoder(sparse=False)),
                                    ('scaler',StandardScaler())
                                ]
                                )

            logging.info(f'categorical columns {categorical_columns} ...... encoding completed')

            # columns tranformer whihc is a comabination of numerical and catgorical pipeline
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipepline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )   

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('read train and test data completed')
            logging.info('Obtaining pre processing object')

            preprocessing_object = self.get_data_tranformer_object()

            target_column = 'math_score'
            numerical_columns = ['reading_score', 'writing_score'] 

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            # print(type(input_feature_train_arr)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path  
            )


        except Exception as e:
            raise CustomException(e, sys)
