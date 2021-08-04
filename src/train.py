from constants import CATEGORIES, WORD_MAX_LEN
from preprocessor import Preprocessor

# model training + tokenizing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPool1D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# pre, post training
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# visualization
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic') 
import seaborn as sns

# save, load from system
import pickle
import os.path
import json

# data manipulation
import numpy as np
import pandas as pd


# all model configuration can go here
model_config = {
  "model_version": "v1",
  "tokenizer_version": "v1",
  "vocab_size": 3000
}

pp = Preprocessor()
cat_to_index = {cat: i for i, cat in enumerate(CATEGORIES)}
index_to_cat = {i: cat for i, cat in enumerate(CATEGORIES)}

def prepare_data():
  print("preparing data...")
  df = pd.read_json("../data/jd_7632.json", encoding="utf-8", dtype={"wd_id": int, "position": str, "main_tasks": str, "requirements" : str, "preferred_points": str, "category": str})
  feature_cols = ["position", "main_tasks", "requirements", "preferred_points"]

  if os.path.isfile("../data/df_clean.csv"):
    df = pd.read_csv("../data/df_clean.csv", usecols= ["text_sum", "text_sum_tokenized", "category"], converters={"text_sum_tokenized": lambda x: x.strip("[]").replace("'", "").split(", ")})  
  else:
    print("no df clean file found. cleaning text from original df. may take few minutes...")
    df = df.copy()
    df["text_sum"] = df[feature_cols].sum(axis=1)
    df["text_sum_tokenized"] = df["text_sum"].apply(pp.split_text)
    df = df[["text_sum", "text_sum_tokenized", "category"]]

    # bottleneck, takes ~3 mins for 8000 rows
    df.to_csv("../data/df_clean.csv")
  return df

def prepare_sentence(df):
  print("preparing sentences...")
  return df.text_sum_tokenized.values

def tokenize(sentences):
  print("tokenizing sentences...")
  tokenizer = Tokenizer(num_words= model_config["vocab_size"], oov_token='oov')
  tokenizer.fit_on_texts(sentences)
  with open(f'../model_deps/tokenizers/{model_config["tokenizer_version"]}/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return tokenizer

def prepare_input_data(sentences, df):
  print("preparing train test data....")
  data_X = np.array(tokenizer.texts_to_sequences(sentences), dtype=object)
  data_y = to_categorical(df.category.replace(cat_to_index).values)
  X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
  X_train = pad_sequences(X_train, maxlen=WORD_MAX_LEN) 
  X_test = pad_sequences(X_test, maxlen=WORD_MAX_LEN)
  return (X_train, X_test, y_train, y_test)

def prepare_model():
  print("preparing model....")
  model = Sequential()
  model.add(Embedding(model_config["vocab_size"], 16))
  # model.add(Conv1D(filters=8, kernel_size=3, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPool1D(pool_size=2))
  model.add(LSTM(50, dropout=0.2))

  model.add(Dense(4, activation='softmax'))
  opt = Adam(learning_rate=3e-4)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
  model.summary()
  return model
  
def train_model(model, input_data):
  print("training model....")
  X_train, X_test, y_train, y_test = input_data
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
  mc = ModelCheckpoint(f'../model_deps/models/{model_config["model_version"]}/model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
  history = model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[es, mc], validation_data=(X_test, y_test))
  log_path = f'../logs/{model_config["model_version"]}.json' 
  
  with open(log_path, 'w') as f:
    print(f'saving to {log_path}')
    json.dump(history.history, f)
  return model

def evaluate(model, input_data):
    _, X_test, _, y_test = input_data
    y_pred = model.predict(X_test)
    
    amt = np.argmax(y_test, axis=1)
    amp = np.argmax(y_pred, axis=1)

    print(classification_report(amt, amp, target_names=CATEGORIES))

    cm = confusion_matrix(amt, amp)

    ax = sns.heatmap(pd.DataFrame(cm), xticklabels=CATEGORIES, yticklabels=CATEGORIES, annot=True, cmap="Blues", fmt="d")
    ax.set(xlabel='predicted', ylabel='true')
    plt.show()



if __name__ == "__main__":
  df = prepare_data()
  sentences = prepare_sentence(df)
  tokenizer = tokenize(sentences)
  input_data = prepare_input_data(sentences, df)
  model = prepare_model()
  model = train_model(model, input_data)
  evaluate(model, input_data)


