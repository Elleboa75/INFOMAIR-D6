import pandas as pd

class Classifier():
    def __init__(self, classifier_type, data_location):
        self.classifier_type = classifier_type
        self.data_location = data_location

    def load_data(self):
        dialog_act = []
        utterance_content = []
        with open(self.data_location, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                act = split_line[0]
                utterance = ' '.join(split_line[1:])
                dialog_act.append(act)
                utterance_content.append(utterance)

            f.close()

        return dialog_act, utterance_content