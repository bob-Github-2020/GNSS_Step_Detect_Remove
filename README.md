# GNSS Step Detection and Remove

Last updated on 1-18-2022, by Bob Wang

Useage: Download the following files into your working directory

Step_detect_remove.py (Detect and remove "obvious" steps in the GNSS ENU time series)

do_loop_step_detect_remove.sh  (Loop the module 'Step_detect_remove.py' for a group of ENU time series files)

*.col (ENU time series, Year  NS  EW  UD ...)

Run: 
  
# ./do_loop_step_detect_remove.sh


Output: *.col_StepFree, and several plots

For components, contact bob.g.wang@gmail.com

Sample output plot (black: original ENU time series; blue: step-free time series)

