import sys 
import os 
import pandas as pd 
from datasets import Dataset 
from transformers import TFAutoModel 
import tensorflow as tf 
from src.logger import logging 
from src.exception import CustomException

model = TFAutoModel.from_pretrained("bert-base-uncased")

class BERTForClassification(tf.keras.Model):

    def __init__(self, bert_model, num_classes):
        try:
            super().__init__()
            self.bert = bert_model 
            self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        except Exception as e:
            raise CustomException(e, sys)
        
    def call(self, inputs):
        try:
            x = self.bert(inputs)[1]
            return self.fc(x)
        except Exception as e:
            raise CustomException(e,sys)
class ModelTrainer:
    def __init__(self):
        pass 
    def train_model(self, train_set, test_set):
        try:
            classifier = BERTForClassification(model, num_classes=6)
            logging.info("Compilation done!")
            classifier.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            history = classifier.fit(
                train_set, 
                epochs=5 
            )
            logging.info("Training done")
            loss, accuracy = classifier.evaluate(test_set) 
            logging.info(f"Evaluation has been completed with loss: {loss} and accuracy: {accuracy}")
            classifier.save('artifacts\saved_model') 
        except Exception as e:
            raise CustomException(e,sys)
    