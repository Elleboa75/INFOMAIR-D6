import numpy as np
from collections import Counter     
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_preprocess import format_data
class BaselineModel_1:  
    def __init__(self, labels):
        self.most_common = Counter(labels).most_common()[0][0]

    def predict(self, sentence):
         return self.most_common    

    def evaluate(self, sentences, labels):
        eval_labels = [self.predict(sentence) for sentence in sentences]
        print('Baseline 1 accuracy:', accuracy_score(labels, eval_labels))


class BaselineModel_2:  
    def __init__(self, keywords):
        self.keywords = keywords

    def predict(self, sentence):
        counter_ = np.zeros(15)
        for cat in enumerate(self.keywords):
            ct = cat[0]
            for keyword in cat[1]:
                if keyword in sentence:
                    counter_[ct] += 1
        return counter_

    def evaluate(self, sentences, labels):
        eval_labels = []
        for sentence in sentences:
            count_ = self.predict(sentence)
            if sum(count_) > 0:
                eval_labels.append(keyword_dict[np.argmax(count_)])
            else:
                eval_labels.append('inform')

        print('Baseline 2 accuracy:', accuracy_score(labels, eval_labels))


labels, text = format_data("dialog_acts.dat")

keyword_ack = ['kay', 'okay', 'good', 'fine']
keyword_affirm = ['yes', 'yeah']
keyword_bye = ['good bye', 'bye']
keyword_confirm = ['does it serve']
keyword_deny = ["not any more", "i dont want", "wrong", "not", "dont want that", "can you change", "that is not important"]
keyword_hello = ["hi", "hello", "halo"]
keyword_inform = ["looking for", "any area", "i dont care"]
keyword_negate = ["not any more", "no"]
keyword_null = ["noise", "sil", "unintelligible", "cough", "system", "inaudible"]
keyword_repeat = ["repeat", "go back", "again"]
keyword_reqalts = ["is there anything else", "anything else", "how about"]
keyword_reqmore = ["more", "request", 'suggestions', "additional"]
keyword_request = ["post code", "information", "postcode", 'address', "what is", "where is", "what are", "where are",'phone', 'phone number']
keyword_restart = ["restart", "start over", "again", "repeat"]
keyword_thankyou = ["thankyou", "thank you", "thanks", "appreciate", 'thank you good bye', 'thank you goodbye','thank you bye', 'thank you good']
keywords = [keyword_ack, keyword_affirm, keyword_bye, keyword_confirm,
            keyword_deny, keyword_hello, keyword_inform, keyword_negate,
            keyword_null, keyword_repeat, keyword_reqalts,
            keyword_reqmore, keyword_request, keyword_restart, keyword_thankyou]

keyword_dict = {0: 'ack', 1: 'affirm', 2: 'bye', 3: 'confirm', 4: 'deny',
                5: 'hello', 6: 'inform', 7: 'negate', 8: 'null', 9: 'repeat',
                10: 'reqalts', 11: 'reqmore', 12: 'request', 13: 'restart', 14: 'thankyou'}


def Train_Baseline_1():
    model = BaselineModel_1(labels)
    return model

def Train_Baseline_2():
    model = BaselineModel_2(keywords)
    return model

"""
eval_labels = []
for sentence in text:
    count_ = baseline_2(sentence, keywords)
    if sum(count_) > 0:
        eval_labels.append(keyword_dict[np.argmax(count_)])
    else:
        eval_labels.append('inform')



print('accuracy:', accuracy_score(labels, eval_labels))
cm = confusion_matrix(labels, eval_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

"""