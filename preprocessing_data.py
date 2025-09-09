# first split it in labels and actual text

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("dialog_acts.dat", "r", encoding="utf-8") as f:
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

#print(df.head())

train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df['dialog_act']
)

#print(train_df.head())
#print(test_df.head())


train_labels = train_df['dialog_act'].values
test_labels = test_df['dialog_act'].values

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)


max_tokens = 10000
sequence_length = 100

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(train_df["sentence"].values)

train_data = tf.data.Dataset.from_tensor_slices((train_df["sentence"], train_labels)).batch(16)
test_data = tf.data.Dataset.from_tensor_slices((test_df["sentence"], test_labels)).batch(16)

train_data = train_data.map(lambda x, y: (vectorize_layer(x), y))
test_data = test_data.map(lambda x, y: (vectorize_layer(x), y))

print(train_data)