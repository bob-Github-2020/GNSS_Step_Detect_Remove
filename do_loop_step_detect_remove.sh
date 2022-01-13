#!/bin/sh
## 1-4-2022, Bob, loop the Python mondule "Step_detect_remove.py" for all NEU files (*.col).
## The output from the modle is *.col_StepFree" file

##get list of all *.col file
for file in `ls *.col`; do
echo $file > process.ctl
./Step_detect_remove.py
rm $file
done

