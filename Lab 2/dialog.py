import pandas as pd
from utils.dialog_manager import DialogManager
from machine_learning import MachineModelOne
import tensorflow as tf
if __name__ == "__main__":
    df = pd.read_csv("data/restaurant_info_additional_data.csv")
    NNmodel = MachineModelOne(dataset_location="dialog_acts.dat",
                              model=tf.keras.models.load_model("models/NN_model.keras"))
    
    print(NNmodel.predict("thanks"))
    dm = DialogManager(df, model = NNmodel, config_path = "utils/dialog_config.json",
                       all_caps = False,
                       allow_restarts = True,
                       delay = 0,
                       formal = True)   # pass dataframe (and optionally config path, model)
    dm.run_dialog()
