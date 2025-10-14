from machine_learning import MachineModelOne, MachineModelTwo, MachineModelThree

if __name__ == "__main__":
    """
    Training file for the machine learning models, also saves them in /saved/
    """
    dataset_location = "data/dialog_acts.dat"
    NNModel = MachineModelOne(dataset_location = dataset_location)
    DecisionTreeModel = MachineModelTwo(dataset_location = dataset_location, max_features=100, max_depth=10)
    LogisticRegressionModel = MachineModelThree(dataset_location=dataset_location)

    LogisticRegressionModel.preprocess()
    LogisticRegressionModel.train_model()

    NNModel.preprocess()    
    NNModel.train_model()
    
    DecisionTreeModel.preprocess()
    DecisionTreeModel.train_model()