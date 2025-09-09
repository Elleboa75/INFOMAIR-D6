import pandas as pd
dialog_act = []
utterance_content = []

with open("dialog_acts.dat") as f:
    for line in f:
        split_line = line.split(' ')
        act = split_line[0]
        utterance =  ' '.join(split_line[1:])
        dialog_act.append(act)
        utterance_content.append(utterance)
        
        
    f.close()
    
    

    

if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



