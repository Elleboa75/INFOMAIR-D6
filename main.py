import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def format_data(data):
    dialog_act = []
    utterance_content = []
    with open(data) as f:
        for line in f:
            split_line = line.split(' ')
            act = split_line[0]
            utterance = ' '.join(split_line[1:])
            dialog_act.append(act)
            utterance_content.append(utterance)

        f.close()
    return dialog_act, utterance_content


labels, text = format_data("dialog_acts.dat")

def baseline_1(labels):
    return Counter(labels).most_common()[0][0]


def baseline_2(sentence, keywords):
    counter_ = np.zeros(15)
    for cat in enumerate(keywords):
        ct = cat[0]
        for keyword in cat[1]:
            if keyword in sentence:
                counter_[ct] += 1
    return counter_


keyword_ack = ['kay', 'okay', 'good', 'fine']
keyword_affirm = ['yes', 'yeah']
keyword_bye = ['good bye', 'bye']
keyword_confirm = ['does it serve']

keyword_deny = ["not any more", "i dont want", "wrong", "not", "dont want that", "can you change",
                "that is not important"]
keyword_hello = ["hi", "hello", "halo"]
keyword_inform = ["looking for", "any area", "i dont care"]
keyword_negate = ["not any more", "no"]

keyword_null = ["noise", "sil", "unintelligible", "cough", "system", "inaudible"]

keyword_repeat = ["repeat", "go back", "again"]
keyword_reqalts = ["is there anything else", "anything else", "how about"]

keyword_reqmore = ["more", "request", 'suggestions', "additional"]
keyword_request = ["post code", "information", "postcode", 'address', "what is", "where is", "what are", "where are",
                   'phone', 'phone number']
keyword_restart = ["restart", "start over", "again", "repeat"]
keyword_thankyou = ["thankyou", "thank you", "thanks", "appreciate", 'thank you good bye', 'thank you goodbye',
                    'thank you bye', 'thank you good']

keywords = [keyword_ack, keyword_affirm, keyword_bye, keyword_confirm,
            keyword_deny, keyword_hello, keyword_inform, keyword_negate,
            keyword_null, keyword_repeat, keyword_reqalts,
            keyword_reqmore, keyword_request, keyword_restart, keyword_thankyou]

keyword_dict = {0: 'ack', 1: 'affirm', 2: 'bye', 3: 'confirm', 4: 'deny',
                5: 'hello', 6: 'inform', 7: 'negate', 8: 'null', 9: 'repeat',
                10: 'reqalts', 11: 'reqmore', 12: 'request', 13: 'restart', 14: 'thankyou',

                }

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






