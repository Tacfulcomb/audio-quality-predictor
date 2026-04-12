This is the distributed version of the scoring program used in the VoiceMOS Challenge.
The goal is to make sure the score calculated locally is consistent with the score calculated by the competition platform.

Usage:
1. Replace the `answer.txt` file in `example/res` with the answer file you wish to submit.
2. Execute `python scoring_program/evaluate.py example <out directory>`. `<out directory>` can be of arbritrary name. If it does not exist, it will be automatically created. If exists, output file will be overwritten.
3. You should then find the output score file: `<out directory>/scores.txt`.

An example submission file is already in `example/res`. Feel free to try an example run by executing the command in step 2.

Should you have any questions, contact `wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp` or `voicemos2022@nii.ac.jp`.