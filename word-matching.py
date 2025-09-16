import Levenshtein
import pandas as pd

# reading csv file
# df = pd.read_csv("assets/restaurant_info.csv")
# # print(df)
#
# df_food = df['food']
# df_price_range = df['pricerange']
# df_area = df['area']
#
#
#
# # o.b.v. keywords uithalen
#
# user_input = "I want a restaurant serving Chinese food".lower()
#
# # split the input
# split_input = user_input.split()
# text_columns = ['food', 'pricerange', 'area']
# match = []
# for column in text_columns:
#     mask = df[column].str.contains("chinese")
#
#     match.append(mask)
# print('----------')
# for column in text_columns:
#     for split in split_input:
#         match = df.loc[df[column].str.contains(split)]
#
#         print(f"match: {split} {match[column]}")
#         print(column)
#
#         ## full match
#         # nu doet ie contains, als het letter in een woord zit.

## Extract preference of the user.



df = pd.read_csv("assets/restaurant_info.csv")

# user input
user_input = "I want Spanihh food".lower()
split_input = user_input.split()

text_columns = ['food', 'pricerange', 'area']

# start with all False
mask = pd.Series(False, index=df.index)

# check each column and each word
for column in text_columns:
    for split in split_input:
        # exact word match using regex with word boundaries
        mask |= df[column].str.contains(rf"\b{split}\b", case=False, na=False)

# filter rows where any match occurred
df_matches = df[mask]

#remove duplicates from the dataframe
if df_matches.empty:
    # Keep the three columns where the match could be found.
    df_food = df['food'].drop_duplicates()
    df_price_range = df['pricerange'].drop_duplicates()
    df_area = df['area'].drop_duplicates()

    possible_matches = []

    # ---- text_columns ----
    for item in df_food:
        for split in split_input:
            dist_calc = Levenshtein.distance(split, str(item).lower())
            if dist_calc < 4:
                possible_matches.append([split, item, dist_calc])

    for item in df_price_range:
        for split in split_input:
            dist_calc = Levenshtein.distance(split, str(item).lower())
            if dist_calc < 4:
                possible_matches.append([split, item, dist_calc])

    for item in df_area:
        for split in split_input:
            dist_calc = Levenshtein.distance(split, str(item).lower())
            if dist_calc < 4:
                possible_matches.append([split, item, dist_calc])

    # ---- remaining columns ----
    remaining_columns = [c for c in df.columns if c not in text_columns]
    for column in remaining_columns:
        unique_values = df[column].drop_duplicates()
        for item in unique_values:
            for split in split_input:
                dist_calc = Levenshtein.distance(split, str(item).lower())
                if dist_calc < 4:
                    possible_matches.append([split, item, dist_calc])

    # create DataFrame with input word, match, and score
    df_possible_matches = pd.DataFrame(
        possible_matches,
        columns=["input_word", "possible_match", "levenshtein_distance"]
    )

    print(df_possible_matches)


