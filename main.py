import pandas as pd
dialog_act = []
utterance_content = []

with open("all_dialogs.txt") as f:
    for line in f:
        split_line = line.split(':')
        if len(split_line) > 1:
            act = split_line[0]
            utterance = split_line[1]
            dialog_act.append(act)
            utterance_content.append(utterance)

    f.close()

dialog_act

if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



