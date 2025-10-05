import pandas as pd
import numpy as np

# new property values
food_quality = ['good', 'average', 'bad']
crowdedness = ['busy', 'not busy']
length_of_stay = ['short', 'medium', 'long']

# path of input and output CSV files
input_csv_path = 'assets/restaurant_info.csv'
output_csv_path = 'assets/restaurant_info_additional_data.csv'

# load CVS
df = pd.read_csv(input_csv_path)

# Add new columns with random values
df['foodquality'] = np.random.choice(food_quality, size=len(df))
df['crowdedness'] = np.random.choice(crowdedness, size=len(df))
df['lengthstay'] = np.random.choice(length_of_stay, size=len(df))

df.to_csv(output_csv_path, index=False)
print(f"New data with additional properties saved to {output_csv_path}")
