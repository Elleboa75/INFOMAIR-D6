from machine_learning import MachineModelOne, MachineModelTwo, MachineModelThree

## Train Models
if __name__ == "__main__":
    #labels, text = format_data("dialog_acts.dat")
    dataset_location = "dialog_acts.dat"
    NNModel = MachineModelOne(dataset_location = dataset_location)
    DecisionTreeModel = MachineModelTwo(dataset_location = dataset_location, max_features=100, max_depth=10)
    LogisticRegressionModel = MachineModelThree(dataset_location=dataset_location, max_tokens=50000, sequence_length=50)

    LogisticRegressionModel.preprocess()
    LogisticRegressionModel.train_model(epochs=33)

    NNModel.preprocess()    
    NNModel.train_model()
    
    DecisionTreeModel.preprocess()
    DecisionTreeModel.train_model()

















