import pandas as pd
import random


def get_restaurant_recommendation(file_path, preferences):
    """
    Recommends a restaurant based on user preferences.

    Args:
        file_path (str): The path to the CSV file containing restaurant data.
        preferences (dict): A dictionary of user preferences (e.g., {'cuisine': 'Italian', 'city': 'New York'}).

    Returns:
        tuple: A tuple containing the recommended restaurant (dict) and a list of other matching restaurants (list of dicts).
               Returns (None, None) if no matching restaurants are found.
    """
    try:
        df = pd.read_csv(file_path)

        for key in preferences:
            preferences[key] = preferences[key].lower()

        # Filter the DataFrame based on user preferences
        mask = pd.Series([True] * len(df))
        for key, value in preferences.items():
            if key in df.columns:
                mask &= (df[key].str.lower() == value)
            else:
                # Handle cases where a preference key doesn't exist in the data
                print(f"Warning: Preference '{key}' not found in the dataset.")
                return None, None

        matching_restaurants = df[mask].to_dict('records')

        if matching_restaurants:
            # Select one restaurant at random
            recommended_restaurant = random.choice(matching_restaurants)

            # Store the rest for possible follow-up requests
            other_matches = [res for res in matching_restaurants if res != recommended_restaurant]

            return recommended_restaurant, other_matches
        else:
            # Handle no matches
            print("Sorry, no restaurants matched your preferences.")
            return None, None

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

