import sys 
from dataclasses import dataclass 
import os 

import numpy as np 
import pandas as pd 

from src.logger import logging 
from src.exception import CustomException 

import tensorflow as tf 
from transformers import AutoTokenizer 

import datasets 
from datasets import Dataset

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased") 

class DataTransformation:
    def __init__(self):
        pass 
    def read_dataset(self, train_path, test_path, validation_path):
        try:
            train_set = pd.read_csv(train_path)
            test_set = pd.read_csv(test_path)
            validation_set = pd.read_csv(validation_path) 
            return(
                train_set, test_set, validation_set  
            )
        except Exception as e: 
            raise CustomException(e,sys)

    def convert_to_dataset(self, train_set, test_set, validation_set):
        try:
            train_1 = Dataset.from_dict(train_set)
            test_1 = Dataset.from_dict(test_set) 
            validation_1 = Dataset.from_dict(validation_set)
            emotions_dataset = datasets.DatasetDict({"train":train_1, "test":test_1, "validation":validation_1})
            return emotions_dataset 
        except Exception as e:
            raise CustomException(e,sys)

    def tokenize(self, batch):
        try:
            return tokenizer(batch["text"], padding=True, truncation=True)
        except Exception as e:
            raise CustomException(e,sys)
            
    def encode(self, emotions_dataset):
        try:
            emotions_encoded = emotions_dataset.map(self.tokenize, batched=True, batch_size=None)
            logging.info("Tokenize method is executed!")
            return emotions_encoded 
        except Exception as e:
            raise CustomException(e,sys) 
        
    def order(self, inp):
        try:
            data = list(inp.values())
            return{
                'input_ids':data[1],
                'attention_mask':data[2],
                'token_type_ids':data[3]
            }, data[0]
        except Exception as e:
            raise CustomException(e,sys)

    def convert_to_tensor(self, train_path, test_path, validation_path):
        try:
            train_set, test_set, val_set = self.read_dataset(train_path, test_path, validation_path)
            logging.info("Train, test and val data is read")
            emotions_dataset = self.convert_to_dataset(train_set, test_set, val_set)
            logging.info("CSV format is converted into dataset")
            emotions_encoded = self.encode(emotions_dataset)
            logging.info("The required dataset is encoded")

            emotions_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
            BATCH_SIZE=64 
                
            train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['train'][:])
            train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
            train_dataset = train_dataset.map(self.order, num_parallel_calls=tf.data.AUTOTUNE)
            logging.info("Train dataset in converted into Tensors")
            test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:])
            test_dataset = test_dataset.batch(BATCH_SIZE).shuffle(1000)
            test_dataset = test_dataset.map(self.order, num_parallel_calls=tf.data.AUTOTUNE) 
            logging.info("Test dataset is converted into Tensors")
            validation_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['validation'][:])
            validation_dataset = validation_dataset.batch(BATCH_SIZE).shuffle(1000)
            validation_dataset = validation_dataset.map(self.order, num_parallel_calls=tf.data.AUTOTUNE) 
            logging.info("Validation dataset is converted into Tensors")
            return(
                train_dataset, 
                test_dataset, 
                validation_dataset  
            )
        except Exception as e:
            raise CustomException(e, sys) 


