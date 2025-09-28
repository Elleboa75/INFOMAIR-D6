import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Neural Network Model
    ## add predict
    ## standardize preprocess
    ## standardize eval
class MachineModelOne():
    def __init__(self,
                 dataset_location,
                 max_tokens = 100,
                 sequence_length = 50,
                 model = None
                 ):
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.dataset_location = dataset_location
        self.data = [] #Final data of the form [train_data, test_data]
        self.label_encoder = LabelEncoder()
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens = max_tokens,
            output_mode = "int",
            output_sequence_length = sequence_length
        )       
        self.model = model


    def preprocess(self):
        with open(self.dataset_location, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)  # split only on the first space
            if len(parts) == 2:
                dialog_act, sentence = parts
            else:  # case where line has only dialog_act and no sentence
                dialog_act, sentence = parts[0], ""
            data.append((dialog_act, sentence))

        # Create DataFrame
        df = pd.DataFrame(data, columns=["dialog_act", "sentence"])
        df = df.drop_duplicates(subset = ["sentence"]) # drop deduplication

        train_df, test_df = train_test_split(
            df,
            test_size=0.15,
            random_state=42,
            #stratify=df['dialog_act']
        )

        train_labels = train_df['dialog_act'].values
        test_labels = test_df['dialog_act'].values

        train_labels = self.label_encoder.fit_transform(train_labels)
        test_labels = self.label_encoder.transform(test_labels)

        self.vectorize_layer.adapt(train_df["sentence"].values)

        train_data = tf.data.Dataset.from_tensor_slices((train_df["sentence"], train_labels)).batch(16)
        test_data = tf.data.Dataset.from_tensor_slices((test_df["sentence"], test_labels)).batch(16)

        train_data = train_data.map(lambda x, y: (self.vectorize_layer(x), y))
        test_data = test_data.map(lambda x, y: (self.vectorize_layer(x), y))

        self.data.append(train_data)
        self.data.append(test_data)

    def train_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim = self.max_tokens, output_dim = 128),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(15, activation = "softmax")
        ])
   

        model.compile(
            loss = "sparse_categorical_crossentropy",
            optimizer = "adam",
            metrics = ["accuracy"]
        )

        history = model.fit(
            self.data[0],
            validation_data = self.data[1],
            epochs = 50
        
        )
        #print(history.history)
        model.save("models/NN_model.keras")
        
    def eval_model(self):   
        train_results = self.model.evaluate(self.data[0])
        test_results = self.model.evaluate(self.data[1])
        print("DNN Train Loss, DNN Train Accuracy:", train_results) 
        print("DNN Test Loss, DNN Test Accuracy:", test_results)
        

    
# Decision Tree Model
    ## add predict
    ## standardize preprocess
    ## standardize eval
class MachineModelTwo():
    def __init__(self, dataset_location, max_features=100, max_depth=None, model=None):
        self.dataset_location = dataset_location
        self.max_features = max_features
        self.max_deph = max_depth
        self.data = []   
        self.label_encoder = LabelEncoder()    
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.model = model
        
    def preprocess(self):
        with open(self.dataset_location, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)  # split only on the first space
            if len(parts) == 2:
                dialog_act, sentence = parts
            else:  # case where line has only dialog_act and no sentence
                dialog_act, sentence = parts[0], ""
            data.append((dialog_act, sentence))

        # Create DataFrame
        df = pd.DataFrame(data, columns=["dialog_act", "sentence"])

        train_df, test_df = train_test_split(
            df,
            test_size=0.15,
            random_state=42,
            stratify=df['dialog_act']
        )

        train_labels = train_df['dialog_act'].values
        test_labels = test_df['dialog_act'].values
        
        train_sentences = train_df["sentence"].values
        test_sentences = test_df["sentence"].values
        
        self.vectorizer.fit(train_sentences)
        
        train_vectors = self.vectorizer.transform(train_sentences)
        test_vectors = self.vectorizer.transform(test_sentences)

        train_labels = self.label_encoder.fit_transform(train_labels)
        test_labels = self.label_encoder.transform(test_labels)

        train_data = (train_vectors, train_labels)
        test_data = (test_vectors, test_labels)

        self.data.append(train_data)
        self.data.append(test_data)
    
    def train_model(self):
        train_vectors, train_labels = self.data[0]
        clf = tree.DecisionTreeClassifier(max_depth=self.max_deph, random_state=42)
        model = clf.fit(train_vectors, train_labels)
        self.model = model
        dt_file = open('models/DT_model.pkl', 'ab')
        pickle.dump(model, dt_file)
        dt_file.close()

    def eval_model(self): 
        train_vectors, train_labels = self.data[0]
        test_vectors, test_labels = self.data[1]

        train_predictions = self.model.predict(train_vectors)
        test_predictions = self.model.predict(test_vectors)
        
        train_acc = accuracy_score(train_labels, train_predictions)
        test_acc = accuracy_score(test_labels, test_predictions)

        print(f"DT Training accuracy: {train_acc:.4f}")
        print(f"DT Test accuracy: {test_acc:.4f}")

# Logistic Regression model
class MachineModelThree:
    def __init__(self, dataset_location, model=None, max_tokens=50000, sequence_length=50, model_path="models/LR_model.keras"):
        self.dataset_location = dataset_location
        self.model = model
        self.model_path = model_path
        self.label_encoder = LabelEncoder()
        self.data = []
        self.sequence_length = sequence_length

        # Integer token IDs -> Embedding model
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=sequence_length
        )

    def preprocess(self):
        with open(self.dataset_location, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        pairs = []
        for line in lines:
            parts = line.split(maxsplit=1)
            dialog_act, sentence = (parts + [""])[:2]
            pairs.append((dialog_act, sentence))

        df = pd.DataFrame(pairs, columns=["dialog_act", "sentence"])
        df = df.drop_duplicates(subset=["dialog_act", "sentence"]).reset_index(drop=True)

        try:
            train_df, test_df = train_test_split(
                df, test_size=0.15, random_state=42, stratify=df["dialog_act"]
            )
        except ValueError:
            train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

        y_train = self.label_encoder.fit_transform(train_df["dialog_act"].values)
        y_test  = self.label_encoder.transform(test_df["dialog_act"].values)

        self.vectorize_layer.adapt(train_df["sentence"].values)

        AUTOTUNE = tf.data.AUTOTUNE

        train_data = (tf.data.Dataset
            .from_tensor_slices((train_df["sentence"].values, y_train))
            .shuffle(len(train_df), seed=42, reshuffle_each_iteration=True)
            .batch(32)
            .map(lambda x, y: (self.vectorize_layer(x), y))
            .cache()
            .prefetch(AUTOTUNE))

        test_data = (tf.data.Dataset
            .from_tensor_slices((test_df["sentence"].values, y_test))
            .batch(32)
            .map(lambda x, y: (self.vectorize_layer(x), y))
            .cache()
            .prefetch(AUTOTUNE))

        self.data = [train_data, test_data]
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, epochs=10):
        if not self.data:
            raise RuntimeError("Call preprocess() before train_model().")
        train_data, test_data = self.data

        num_classes = len(self.label_encoder.classes_)
        vocab_size  = self.vectorize_layer.vocabulary_size()
        seq_len     = self.sequence_length
        emb_dim     = 128

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(seq_len,)),
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
        ]

        model.fit(train_data, validation_data=test_data, epochs=epochs, callbacks=callbacks, verbose=2)

        # save properly (NO pickle)
        Path(os.path.dirname(self.model_path) or ".").mkdir(parents=True, exist_ok=True)
        model.save(self.model_path)

        self.model = model

    def _ensure_model_loaded(self):
        if isinstance(self.model, tf.keras.Model):
            return
        if self.model is None or isinstance(self.model, (str, os.PathLike)):
            path = self.model if isinstance(self.model, (str, os.PathLike)) else self.model_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}. Train first or set a correct path.")
            self.model = tf.keras.models.load_model(path)
            return
        raise TypeError(f"self.model is not a Keras model. Got: {type(self.model)}. "
                        f"Did you assign a file handle? Use tf.keras.models.load_model(...) instead.")

    def eval_model(self, detailed_report=True):

        if not self.data:
            raise RuntimeError("Call preprocess() before eval_model().")

        self._ensure_model_loaded()

        test_loss, test_acc = self.model.evaluate(self.data[1], verbose=0)
        print(f"Test  Loss: {test_loss:.4f}  |  Test  Acc: {test_acc:.4f}")

        if not detailed_report:
            return

        y_pred_probs = self.model.predict(self.data[1], verbose=0)
        y_pred = y_pred_probs.argmax(axis=1)
        y_true = self.y_test

        all_labels = np.arange(len(self.label_encoder.classes_))
        print("\nClassification report Logistic Regression:\n",
              classification_report(
                  y_true, y_pred,
                  labels=all_labels,
                  target_names=self.label_encoder.classes_,
                  zero_division=0
              ))


"""
if __name__ == "__main__":
    dataset_location = "dialog_acts.dat"
    machine_model = MachineModelOne(dataset_location = dataset_location)
    machine_model.preprocess()
    machine_model.model()
    
    machine_model = MachineModelTwo(dataset_location = dataset_location)
    machine_model.preprocess()
    machine_model.model()
"""