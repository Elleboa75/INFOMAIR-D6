import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
    def __init__(
        self,
        dataset_location,
        model=None,
        max_features=3000,
        model_path="models/LR_model.pkl",
    ):
        self.dataset_location = dataset_location
        self.model = model
        self.model_path = model_path
        self.label_encoder = LabelEncoder()
        self.data = []  # [(X_train, y_train), (X_test, y_test)]

        # Bag-of-Words (TF-IDF) vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),          # unigrams + bigrams
            stop_words=None,             # keep function words
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            min_df=2,
            max_df=0.95,
        )

    # ---------- Loading ----------
    def _ensure_model_loaded(self):
        # Already a fitted sklearn model in memory?
        if isinstance(self.model, LogisticRegression):
            return

        path = self.model if isinstance(self.model, (str, os.PathLike)) else self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Train first or set a correct path.")

        loaded = joblib.load(path)

        # Case A: full bundle dict
        if isinstance(loaded, dict):
            if "model" not in loaded:
                raise ValueError(f"Loaded dict from {path} is missing the 'model' key.")
            self.model = loaded["model"]
            if "vectorizer" in loaded and loaded["vectorizer"] is not None:
                self.vectorizer = loaded["vectorizer"]
            if "label_encoder" in loaded and loaded["label_encoder"] is not None:
                self.label_encoder = loaded["label_encoder"]
            return

        # Case B: raw sklearn model (backward compatibility)
        if isinstance(loaded, LogisticRegression):
            self.model = loaded
            # vectorizer/label_encoder remain as currently set in self (likely unfitted)
            return

        raise TypeError(
            f"Unsupported artifact loaded from {path}: {type(loaded)}. "
            f"Expected dict bundle or LogisticRegression."
        )

    # ---------- Data ----------
    def preprocess(self):
        with open(self.dataset_location, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        pairs = []
        for line in lines:
            parts = line.split(maxsplit=1)
            dialog_act, sentence = (parts + [""])[:2]
            pairs.append((dialog_act, sentence))

        df = pd.DataFrame(pairs, columns=["dialog_act", "sentence"]) \
               .drop_duplicates(subset=["dialog_act", "sentence"]) \
               .reset_index(drop=True)

        try:
            train_df, test_df = train_test_split(
                df, test_size=0.15, random_state=42, stratify=df["dialog_act"]
            )
        except ValueError:
            train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

        # LabelEncoder: fit only if not already fitted
        if hasattr(self.label_encoder, "classes_"):
            y_train = self.label_encoder.transform(train_df["dialog_act"].values)
            y_test  = self.label_encoder.transform(test_df["dialog_act"].values)
        else:
            y_train = self.label_encoder.fit_transform(train_df["dialog_act"].values)
            y_test  = self.label_encoder.transform(test_df["dialog_act"].values)

        # Vectorizer: fit only if not already fitted
        if getattr(self.vectorizer, "vocabulary_", None) is None:
            self.vectorizer.fit(train_df["sentence"].values)

        X_train = self.vectorizer.transform(train_df["sentence"].values)
        X_test  = self.vectorizer.transform(test_df["sentence"].values)

        self.data = [(X_train, y_train), (X_test, y_test)]

    # ---------- Training ----------
    def train_model(self, C=1.0, max_iter=1000):
        if not self.data:
            raise RuntimeError("Call preprocess() before train_model().")

        (X_train, y_train), _ = self.data

        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",      # multinomial by default as of sklearn 1.5+
            max_iter=max_iter
        )
        model.fit(X_train, y_train)
        self.model = model

        # Save bundle (recommended)
        bundle = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
        }
        Path(os.path.dirname(self.model_path) or ".").mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)
        print(f"Model saved at: {self.model_path}")

    # ---------- Evaluation ----------
    def eval_model(self, detailed_report=True):
        if not self.data:
            raise RuntimeError("Call preprocess() before eval_model().")

        # Ensure we have a model (and optionally vectorizer/encoder) loaded
        try:
            self._ensure_model_loaded()
        except FileNotFoundError:
            # No saved model yet — that's fine if you just trained in this session
            if not isinstance(self.model, LogisticRegression):
                raise

        (X_train, y_train), (X_test, y_test) = self.data

        y_train_pred = self.model.predict(X_train)
        y_test_pred  = self.model.predict(X_test)
        print("-------------------------")
        print("Logistic Regression Accuracy")

        print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Validation/Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

        if not detailed_report:
            return

        all_labels = np.arange(len(self.label_encoder.classes_))
        print("\nClassification report (validation):\n",
              classification_report(
                  y_test, y_test_pred,
                  labels=all_labels,
                  target_names=self.label_encoder.classes_,
                  zero_division=0
              ))

        # Optional: show any labels missing in test
        missing = set(all_labels) - set(np.unique(y_test))
        if missing:
            print("Classes missing in test set:",
                  [self.label_encoder.classes_[i] for i in sorted(missing)])
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