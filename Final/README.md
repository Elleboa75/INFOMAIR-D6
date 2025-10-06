## Summary of all code files
### Root Folder
- `dialog.py`: Main file to run the dialog system 

### Interfaces
All the interfaces are define the standard functions that we use for each class. They are not necessary to run the actual code, but they do provide some quick overview of what functions the classes contain.

### Models
- `models/main.py`: Main file to train and save the dialog act classifier models
- `models/baseline.py`: File where both baseline model classes are defined
- `models/machine_learning.py`: This file contains the three machine learning models classes.  
- `models/data_preprocess.py`: Helper function that splits the dialog_act data into a utterance (input) and dialog act (label)
- `models/eval.py`: Running this file will run an evaluation on all four models, printing their respective scores
- `models/user_input.py`: Running this file will run inferences on utterances input by users (CLI)

### Utils
- `utils/dialog_manager.py`: This file contains the main dialog state logic. State changes, and dialog interaction management is handled here. 
- `utils/extractor.py`: This file handles the preference extraction from user utterances (e.g. extracts italian food)
- `utils/parser.py`: This file parses the user utterance when there are not matches found (so changes to preferences are asked). E.g., it parses the user input (like food), and returns the state that needs to be changed
- `utils/validator.py`: The validator validates whether the extracted preferences are in present in the data. E.g. if italian is extracted it will check whether it exists in the df['food'] column.
- `utils/restaurant_matcher.py`: This file handles the spell correction of preferences using Levenshtein distance .
- `utils/suggestion_engine.py`: The suggestion engine uses the extracted preferences to return a filtered dataset with the preferences applied. If extra preferences are found, inference rules are applied.
- `utils/match_result.py`: Helper class for the Levenhstein based spell correction. Give a standard format to found matches.

## How to run:
`models/main.py`: Run main.py in a Terminal. The file will automatically train and saved the models to models/saved/
`models/eval.py`: Run eval.py after models are trained and saved (only need training once). The terminal will automatically print all the evaluation results.
`models/user_input.py`: Run user_input.py in the terminal if you want to run inferences on all models using CLI input.
`dialog.py`: Run this file if you want to start the dialog system.
---


## Group members:
- Yazan Mousa
- Julian Nobbe
- David Vanghelescu
- Justin Nguyen (7085842)
