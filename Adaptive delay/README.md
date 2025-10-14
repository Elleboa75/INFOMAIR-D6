# Summary of all code files
### Root Folder
- `dialog.py`: Main file to run the dialog system 

### Interfaces
All the interfaces define the standard functions that we use for each class. They are not necessary to run the actual code, but they do provide some quick overview of what functions the classes should contain.

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

### How to run:
- `models/main.py`: Run main.py in a Terminal from the root folder (e.g.: py .\models\eval.py). The file will automatically train and saved the models to models/saved/
- `models/eval.py`: Run eval.py after models are trained and saved (only need training once). The terminal will automatically print all the evaluation results. Run the file from the root folder (e.g.: py .\models\eval.py)
```
Baseline 1 accuracy: 0.39841574840202343
Baseline 2 accuracy: 0.8337320105093918
-------------------------
Decision Tree Accuracy
Train accuracy: 0.8313
Validation/Test accuracy: 0.8348
[Confusion Matrix]
-------------------------
Neural Network Accuracy
Train accuracy: 0.9844
Validation/Test accuracy: 0.9266
[Confusion Matrix]
-------------------------
Logistic Regression Accuracy
Train accuracy: 0.9800
Validation/Test accuracy: 0.9739
[Confusion Matrix]
```
- `models/user_input.py`: Run user_input.py in the terminal from the root folder (e.g.: py .\models\user_input.py), if you want to run inferences on all models using CLI input.

```
Please type the sentence to classify
> Hello how are you
Baseline 1 prediction: inform
Baseline 2 prediction: inform
Decision Tree prediction: inform
Neural Network prediction: hello
Logistic Regression prediction: reqalts
```
- `dialog.py`: Run this file if you want to start the dialog system.
```
> Good day, and welcome to the Cambridge restaurant reservation system. I would be delighted to assist you in finding restaurants based on your preferences for area, price range, or cuisine type. How may I be of service today?
> User: ___
```

## Configurability
-  Display all system messages in uppercase: in the dialog.py file set the all_caps bool to True or False
```
> GOOD DAY, AND WELCOME TO THE CAMBRIDGE RESTAURANT RESERVATION SYSTEM. I WOULD BE DELIGHTED TO ASSIST YOU IN FINDING RESTAURANTS BASED ON YOUR PREFERENCES FOR AREA, PRICE RANGE, OR CUISINE TYPE. HOW MAY I BE OF SERVICE TODAY?
> User: ___
```
- Disable / enable the systems ability to restart once we are in the no_alternatives state: in the dialog.py file set the allow_restarts bool to True or False
```
> System: Unfortunately, no restaurants match your current preferences. Would you like to modify a criterion like: area, cuisine type, or price range? Or would you like to restart?
> User: restart
> System: I apologize, but system restarts are not currently permitted. Would you prefer to modify one of the following: area, cuisine type, or price range?
```
- Make the system use formal / informal language: in the dialog.py file set the formal bool to True or False
```
Informal
> Hey there! Welcome to the Cambridge restaurant finder! I can help you find places to eat by area, price, or food type. What're you in the mood for?
> User: ___ 

Formal
> Good day, and welcome to the Cambridge restaurant reservation system. I would be delighted to assist you in finding restaurants based on your preferences for area, price range, or cuisine type. How may I be of service today?
> User: ___

```
- System thinking delay: In dialog.py change the delay (in seconds) of thinking.
```
> Good day, and welcome to the Cambridge restaurant reservation system. I would be delighted to assist you in finding restaurants based on your preferences for area, price range, or cuisine type. How may I be of service today?
> User: I would like a italian restaurant in the west area with moderate price
> Thinking...
```


---


## Group members:
- Yazan Mousa (4971698)
- Julian Nobbe (4772318)
- David Vanghelescu (1674560)
- Justin Nguyen (7085842)
