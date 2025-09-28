import pandas as pd
from utils.dialog_manager import DialogManager

if __name__ == "__main__":
    df = pd.read_csv("restaurant_info.csv")
    dm = DialogManager(df)   # pass dataframe (and optionally config path, model)
    dm.run_dialog()
