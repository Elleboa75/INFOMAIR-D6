from baseline import Train_Baseline_1, Train_Baseline_2
from machine_learning import MachineModelOne, MachineModelTwo, MachineModelThree
import pickle
import tensorflow as tf

baseline_1_model = Train_Baseline_1()
baseline_2_model = Train_Baseline_2()

DT_file = open('saved/DT_model.pkl', 'rb')
DTModel_file = pickle.load(DT_file)
DT_file.close()

DT_model = MachineModelTwo(dataset_location = "../data/dialog_acts.dat", max_features=100, max_depth=10, model=DTModel_file)
DT_model.preprocess()

NNmodel = MachineModelOne(dataset_location = "../data/dialog_acts.dat", model = tf.keras.models.load_model("saved/NN_model.keras"))
NNmodel.preprocess()


LRmodel = MachineModelThree("../data/dialog_acts.dat", model_path="saved/LR_model.pkl")
LRmodel.preprocess()
LRmodel._ensure_model_loaded()

if __name__ == "__main__":
    while True:
        print('Please type the sentence to classify')
        sentence = input()
        if sentence.lower() == 'exit':
            break
        print('Baseline 1 prediction:', baseline_1_model.predict())
        print('Baseline 2 prediction:', baseline_2_model.predict_label(sentence))
        print('Decision Tree prediction:', DT_model.predict_label([sentence]))
        print('Neural Network prediction:', NNmodel.predict([sentence]))
        print('Logisitc Regression prediction:', LRmodel.predict_labels([sentence]))
