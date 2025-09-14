from machine_learning import MachineModelOne, MachineModelTwo

## Train Models
if __name__ == "__main__":
    #labels, text = format_data("dialog_acts.dat")
    NNModel = MachineModelOne(dataset_location = "dialog_acts.dat")
    DecisionTreeModel = MachineModelTwo(dataset_location = "dialog_acts.dat", max_features=100, max_depth=10)

    NNModel.preprocess()    
    NNModel.train_model()
    
    DecisionTreeModel.preprocess()
    DecisionTreeModel.train_model()

















