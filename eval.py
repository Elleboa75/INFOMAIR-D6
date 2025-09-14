import pickle
from baseline import Train_Baseline_1, Train_Baseline_2
from data_preprocess import format_data
from machine_learning import MachineModelOne, MachineModelTwo   
import tensorflow as tf 
if __name__ == "__main__":
    labels, text = format_data("dialog_acts.dat")
    baseline_1_model = Train_Baseline_1()
    baseline_2_model = Train_Baseline_2()

    DT_file = open('models/DT_model.pkl', 'rb')
    DTModel_file = pickle.load(DT_file)
    DT_file.close()

    DT_model = MachineModelTwo(dataset_location = "dialog_acts.dat", max_features=100, max_depth=10, model=DTModel_file)
    DT_model.preprocess()

    NNmodel = MachineModelOne(dataset_location = "dialog_acts.dat", model = tf.keras.models.load_model("models/NN_model.keras"))
    NNmodel.preprocess()

    ## Baseline Models
    baseline_1_model.evaluate(text, labels)
    baseline_2_model.evaluate(text, labels)   

    ## Decision Tree Model
    DT_model.eval_model() 

    ## Neural Network Model 
    NNmodel.eval_model()    


"""
def inference_model(model, input_data):
    model.predict(input_data) ## output for each model is different

"""