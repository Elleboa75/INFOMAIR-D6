## split the csv into lists
import re

import pandas as pd

# reading csv file
df = pd.read_csv("assets/restaurant_info.csv")
# print(df)

df_food = df['food']
df_price_range = df['pricerange']
df_area = df['area']

# o.b.v. keywords uithalen

user_input = "I want a restaurant serving Chinese food".lower()

# split the input
split_input = user_input.split()
text_columns = ['food', 'pricerange', 'area']
match = []
for column in text_columns:
    mask = df[column].str.contains("chinese")

    match.append(mask)
print('----------')
for column in text_columns:
    for split in split_input:
        match = df.loc[df[column].str.contains(split)]

        print(f"match: {split} {match[column]}")
        print(column)

        ## full match
        # nu doet ie contains, als het letter in een woord zit.