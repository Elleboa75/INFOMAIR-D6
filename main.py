import pandas as pd


def format_data(data):
    dialog_act  = []
    utterance_content = []
    with open(data) as f:
        for line in f:
            split_line = line.split(' ')
            act = split_line[0]
            utterance =  ' '.join(split_line[1:])
            dialog_act.append(act)
            utterance_content.append(utterance)


        f.close()
    return dialog_act, utterance_content

    
def baseline_1(labels):
    return Counter(labels).most_common()[0][0]


    
    

    

if __name__ == '__main__':
    labels, text = format_data("dialog_acts.dat" )
    print(baseline_1(labels))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



