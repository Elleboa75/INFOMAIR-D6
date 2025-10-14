import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pickle

def plot_confusion(y_true, y_pred, class_names, title="Confusion Matrix", show_all_classes=True):
    """
    Plot confusion matrix with proper label alignment and model title.
    Automatically handles cases where some classes are missing from y_true/y_pred.
    """

    # Ensure label alignment (include missing classes if requested)
    if show_all_classes:
        labels_idx = np.arange(len(class_names))
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        display_labels = class_names
    else:
        labels_idx = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        display_labels = np.array(class_names)[labels_idx]

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    print(f"\n{title}:")
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", ax=ax, values_format="d")

    # Add title (model name)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 1️⃣ Neural Network Model (TF-IDF + Dense)
# ------------------------------------------------------
class MachineModelOne():
    def __init__(self,
                 dataset_location,
                 max_features=3000,
                 model=None,
                 model_path="models/saved/NN_model.keras"):
        self.dataset_location = dataset_location
        self.model = model
        self.model_path = model_path

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        self.label_encoder = LabelEncoder()
        self.data = []

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
        y_test = self.label_encoder.transform(test_df["dialog_act"].values)

        self.vectorizer.fit(train_df["sentence"].values)
        X_train = self.vectorizer.transform(train_df["sentence"].values)
        X_test = self.vectorizer.transform(test_df["sentence"].values)

        self.data = [(X_train, y_train), (X_test, y_test)]

    def train_model(self, epochs=50, batch_size=32, hidden_units=128, l2=1e-4):
        if not self.data:
            raise RuntimeError("Call preprocess() before train_model().")

        (X_train_sparse, y_train), (X_val_sparse, y_val) = self.data
        X_train = X_train_sparse.toarray().astype("float32")
        X_val = X_val_sparse.toarray().astype("float32")

        num_classes = len(self.label_encoder.classes_)
        vocab_size = X_train.shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(vocab_size,)),
            tf.keras.layers.Dense(hidden_units, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]

        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  verbose=2)

        Path(os.path.dirname(self.model_path) or ".").mkdir(parents=True, exist_ok=True)
        model.save(self.model_path)
        self.model = model

        joblib.dump(
            {"vectorizer": self.vectorizer, "label_encoder": self.label_encoder},
            os.path.join(os.path.dirname(self.model_path), "NN_vectorizer_le.pkl")
        )

    def _ensure_model_loaded(self):
        if isinstance(self.model, tf.keras.Model):
            return
        path = self.model if isinstance(self.model, (str, os.PathLike)) else self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}.")
        self.model = tf.keras.models.load_model(path)
        aux = os.path.join(os.path.dirname(path), "NN_vectorizer_le.pkl")
        if os.path.exists(aux):
            bundle = joblib.load(aux)
            self.vectorizer = bundle.get("vectorizer", self.vectorizer)
            self.label_encoder = bundle.get("label_encoder", self.label_encoder)

    def eval_model(self, detailed_report=True):
        if not self.data:
            raise RuntimeError("Call preprocess() before eval_model().")
        self._ensure_model_loaded()

        (X_train_sparse, y_train), (X_test_sparse, y_test) = self.data
        X_train = X_train_sparse.toarray().astype("float32")
        X_test = X_test_sparse.toarray().astype("float32")

        y_train_pred = self.model.predict(X_train, verbose=0).argmax(axis=1)
        y_test_pred = self.model.predict(X_test, verbose=0).argmax(axis=1)

        print("-------------------------")
        print("Neural Network Accuracy")
        print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Validation/Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

        if detailed_report:
            all_labels = np.arange(len(self.label_encoder.classes_))
            print("\nClassification report:\n",
                  classification_report(
                      y_test, y_test_pred,
                      labels=all_labels,
                      target_names=self.label_encoder.classes_,
                      zero_division=0
                  ))

            plot_confusion(y_test, y_test_pred, self.label_encoder.classes_,
                           title="Confusion Matrix (Neural Network)")

    def predict(self, texts):
        self._ensure_model_loaded()
        X = self.vectorizer.transform(texts).toarray().astype("float32")
        y_pred = self.model.predict(X, verbose=0).argmax(axis=1)
        return self.label_encoder.inverse_transform(y_pred)[0]


# ------------------------------------------------------
# 2️⃣ Decision Tree Model (CountVectorizer)
# ------------------------------------------------------
class MachineModelTwo():
    def __init__(self, dataset_location, max_features=100, max_depth=None, model=None):
        self.dataset_location = dataset_location
        self.max_features = max_features
        self.max_depth = max_depth
        self.data = []
        self.label_encoder = LabelEncoder()
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.model = model

    def preprocess(self):
        with open(self.dataset_location, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            dialog_act, sentence = (parts + [""])[:2]
            data.append((dialog_act, sentence))

        df = pd.DataFrame(data, columns=["dialog_act", "sentence"])
        train_df, test_df = train_test_split(
            df, test_size=0.15, random_state=42, stratify=df['dialog_act']
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
        self.data = [(train_vectors, train_labels), (test_vectors, test_labels)]

    def train_model(self):
        train_vectors, train_labels = self.data[0]
        clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        self.model = clf.fit(train_vectors, train_labels)
        Path("saved").mkdir(exist_ok=True)
        with open('saved/DT_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def eval_model(self, detailed_report=True):
        if not self.data or self.model is None:
            raise RuntimeError("Call preprocess() and train_model() before eval_model().")

        (train_vectors, train_labels), (test_vectors, test_labels) = self.data
        train_preds = self.model.predict(train_vectors)
        test_preds = self.model.predict(test_vectors)

        print("-------------------------")
        print("Decision Tree Accuracy")
        print(f"Train accuracy: {accuracy_score(train_labels, train_preds):.4f}")
        print(f"Validation/Test accuracy: {accuracy_score(test_labels, test_preds):.4f}")

        if detailed_report:
            all_labels = np.arange(len(self.label_encoder.classes_))
            print("\nClassification report:\n",
                  classification_report(
                      test_labels, test_preds,
                      labels=all_labels,
                      target_names=self.label_encoder.classes_,
                      zero_division=0
                  ))
            plot_confusion(test_labels, test_preds, self.label_encoder.classes_,
                           title="Confusion Matrix (Decision Tree)")

    def predict_label(self, sentence):
        return self.label_encoder.inverse_transform(
            self.model.predict(self.vectorizer.transform(sentence))
        )[0]


# ------------------------------------------------------
# 3️⃣ Logistic Regression Model
# ------------------------------------------------------
class MachineModelThree:
    def __init__(
        self,
        dataset_location,
        model=None,
        max_features=3000,
        model_path="saved/LR_model.pkl",
    ):
        self.dataset_location = dataset_location
        self.model = model
        self.model_path = model_path
        self.label_encoder = LabelEncoder()
        self.data = []
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            min_df=2,
            max_df=0.95,
        )

    def _ensure_model_loaded(self):
        if isinstance(self.model, LogisticRegression):
            return
        path = self.model if isinstance(self.model, (str, os.PathLike)) else self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Train first.")
        loaded = joblib.load(path)
        if isinstance(loaded, dict):
            self.model = loaded["model"]
            self.vectorizer = loaded.get("vectorizer", self.vectorizer)
            self.label_encoder = loaded.get("label_encoder", self.label_encoder)
        else:
            self.model = loaded

    def preprocess(self):
        with open(self.dataset_location, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        pairs = [(ln.split(maxsplit=1)[0], ln.split(maxsplit=1)[1] if len(ln.split(maxsplit=1)) > 1 else "") for ln in lines]
        df = pd.DataFrame(pairs, columns=["dialog_act", "sentence"])
        train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["dialog_act"])

        y_train = self.label_encoder.fit_transform(train_df["dialog_act"].values)
        y_test = self.label_encoder.transform(test_df["dialog_act"].values)
        self.vectorizer.fit(train_df["sentence"].values)
        X_train = self.vectorizer.transform(train_df["sentence"].values)
        X_test = self.vectorizer.transform(test_df["sentence"].values)
        self.data = [(X_train, y_train), (X_test, y_test)]

    def train_model(self, C=1.0, max_iter=1000):
        (X_train, y_train), _ = self.data
        model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter)
        model.fit(X_train, y_train)
        self.model = model
        Path(os.path.dirname(self.model_path) or ".").mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "vectorizer": self.vectorizer, "label_encoder": self.label_encoder}, self.model_path)
        print(f"Model saved at: {self.model_path}")

    def eval_model(self, detailed_report=True):
        (X_train, y_train), (X_test, y_test) = self.data
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        print("-------------------------")
        print("Logistic Regression Accuracy")
        print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Validation/Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

        if detailed_report:
            all_labels = np.arange(len(self.label_encoder.classes_))
            print("\nClassification report:\n",
                  classification_report(
                      y_test, y_test_pred,
                      labels=all_labels,
                      target_names=self.label_encoder.classes_,
                      zero_division=0
                  ))
            plot_confusion(y_test, y_test_pred, self.label_encoder.classes_,
                           title="Confusion Matrix (Logistic Regression)")

    def predict_labels(self, sentence):
        return self.label_encoder.inverse_transform(
            self.model.predict(self.vectorizer.transform(sentence))
        )[0]
