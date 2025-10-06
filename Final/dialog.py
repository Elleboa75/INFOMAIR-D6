import pandas as pd
from utils.dialog_manager import DialogManager
from models.machine_learning import MachineModelOne
import tensorflow as tf
if __name__ == "__main__":
    df = pd.read_csv("data/restaurant_info_additional_data.csv")
    NNmodel = MachineModelOne(dataset_location="data/dialog_acts.dat",
                              model=tf.keras.models.load_model("models/saved/NN_model.keras"))
    NNmodel.preprocess()
    dm = DialogManager(df, model = NNmodel, config_path = "utils/dialog_config.json",
                       all_caps = False, # Type all system messages in caps
                       allow_restarts = False, # Allow restarts
                       delay = 10, # Delay in seconds,
                       formal = True # Formal prompts toggle
                       )   
    dm.run_dialog()
