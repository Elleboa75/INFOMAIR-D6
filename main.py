import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from machine_learning import MachineModelOne, MachineModelTwo
from baseline import baseline_1, baseline_2
from data_preprocess import format_data



if __name__ == "__main__":
    #labels, text = format_data("dialog_acts.dat")
    NNModel = MachineModelOne(dataset_location = "dialog_acts.dat")
    DecisionTreeModel = MachineModelTwo(dataset_location = "dialog_acts.dat", max_features=100, max_depth=10)

    NNModel.preprocess()    
    NNModel.train_model()
    
    #DecisionTreeModel.preprocess()
    #DecisionTreeModel.train_model()

















