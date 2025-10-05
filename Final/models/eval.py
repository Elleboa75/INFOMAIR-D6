import pickle
from baseline import Train_Baseline_1, Train_Baseline_2
from data_preprocess import format_data
from machine_learning import MachineModelOne, MachineModelTwo, MachineModelThree
import tensorflow as tf 
if __name__ == "__main__":
    labels, text = format_data("../data/dialog_acts.dat")
    baseline_1_model = Train_Baseline_1()
    baseline_2_model = Train_Baseline_2()

    DT_file = open('saved/DT_model.pkl', 'rb')
    DTModel_file = pickle.load(DT_file)
    DT_file.close()

    DT_model = MachineModelTwo(dataset_location = "../data/dialog_acts.dat", max_features=100, max_depth=10, model=DTModel_file)
    DT_model.preprocess()

    NNmodel = MachineModelOne(dataset_location = "../data/dialog_acts.dat", model = tf.keras.models.load_model("saved/NN_model.keras"))
    NNmodel.preprocess()

    LRmodel = MachineModelThree("../data/dialog_acts.dat", model=None, model_path="saved/LR_model.pkl")
    LRmodel.preprocess()
    LRmodel.eval_model()

    ## Baseline Models
    baseline_1_model.evaluate(text, labels)
    baseline_2_model.evaluate(text, labels)   

    ## Decision Tree Model
    DT_model.eval_model() 

    ## Neural Network Model 
    NNmodel.eval_model()

    LRmodel.eval_model()
