def format_data(data):
    """
    Simple function to get the data in the correct format for training. Splits the data in utterances (feature) and dialog_acts (labels)
    """
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
