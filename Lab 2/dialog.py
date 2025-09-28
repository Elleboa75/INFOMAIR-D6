import pandas as pd
from utils.dialog_manager import DialogManager

if __name__ == "__main__":
    df = pd.read_csv("data/restaurant_info_additional_data.csv")
    dm = DialogManager(df, config_path = "utils/dialog_config.json")   # pass dataframe (and optionally config path, model)
    dm.run_dialog()
