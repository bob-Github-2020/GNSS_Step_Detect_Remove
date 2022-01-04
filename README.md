# GNSS_Step_Detect_Remove

Last updated on 1-4-2022, by Bob Wang

Useage:
Step_detect_remove.py (Detect and remove "obvious" steps in the GNSS ENU time series)
do_loop_step_detect_remove (Loop the module (Step_detect_remove.py) on a group of ENU time series files)
*.col (ENU time series)

Run: 
  
  ./do_loop_step_detect_remove

Input:
*.col: Three component of GNSS ENU time series, Year, NS, EW, UD, ...

Output: *.col_StepFree, and several plots

THIS PROGRAM RUNS VERY SLOW!

For components, contact bob.g.wang@gmail.com

