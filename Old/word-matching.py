import Levenshtein
import pandas as pd


df = pd.read_csv("assets/restaurant_info.csv")

# user input
user_input = "I want to go turkishh ".lower()
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

# --- print exact matches first ---
if not df_matches.empty:
    print(f"Found {len(df_matches)} exact keyword matches")
    cols_to_show = [c for c in ["restaurantname", "food", "pricerange", "area"] if c in df.columns]
    print(df_matches[cols_to_show].head(10))
else:
    #remove duplicates from the dataframe
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

    def get_match_with_smallest_distance(df):
        df.sort_values(by=['levenshtein_distance'], inplace=True)
        return df.iloc[0]['possible_match']

    if df_possible_matches.empty:
        print("No matches found")
    else:
        best_match = get_match_with_smallest_distance(df_possible_matches)
        print("Best fuzzy match:", best_match)
