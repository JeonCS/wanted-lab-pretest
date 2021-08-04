import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from constants import CATEGORIES, WORD_MAX_LEN
from preprocessor import Preprocessor


class ModelDependency:
    def __init__(self, config):
        print("initializing model dependencies....")
        self.model = load_model(f'../model_deps/models/{config["model_version"]}/model.h5')
        self.data_preprocessor = Preprocessor()
        with open(f'../model_deps/tokenizers/{config["tokenizer_version"]}/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def prepare_text(self, jd):
        feature_cols = ["position", "requirements", "main_tasks", "preferred_points"]
        t = ""
        for col in feature_cols:
            if col in jd:
                t = t + jd[col] + " "
        x = self.data_preprocessor.split_text(t)
        x = self.tokenizer.texts_to_sequences([t])
        x = pad_sequences(x, maxlen=WORD_MAX_LEN)
        return x    
