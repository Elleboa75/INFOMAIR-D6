import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Neural Network Model
class MachineModelOne():
    def __init__(self,
                 dataset_location,
                 max_tokens = 100,
                 sequence_length = 50
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
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim = self.max_tokens, output_dim = 128),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(15, activation = "softmax")
        ])
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim = self.max_tokens, output_dim = 128),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256, activation = "relu"),
            tf.keras.layers.Dense(512, activation = "relu"),
            tf.keras.layers.Dense(1024, activation = "relu"),
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
        
        


    
# Decision Tree Model
class MachineModelTwo():
    def __init__(self, dataset_location, max_features=100, max_depth=None):
        self.dataset_location = dataset_location
        self.max_features = max_features
        self.max_deph = max_depth
        self.data = []   
        self.label_encoder = LabelEncoder()    
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.model = None
        
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

        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        


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