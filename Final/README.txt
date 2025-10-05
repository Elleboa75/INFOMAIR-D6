How to run:
Just run the dialog.py file

Notes (still to do + things done):
1. We have to change the way extra-requirements are asked (now its just implemented as a 3 question sequence-- > suggestion).
   It should be a 1-line anwser with automatic extraction --> suggestion

2. Changed the diagram

3. Changed the code format

3. Added the following configurability features: You can enable/disably them in the dialog.py file (
- OUTPUT IN ALL CAPS OR NOT
- Introduce a delay before showing system responses
- Allow dialog restarts or not
- Use formal or informal phrases in system utterances

4. Even the best trained model still wrongly classifies dialog acts, this breaks the flow and sometimes stop the dialog from ending/progressing

