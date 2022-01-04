# GNSS_Step_Detect_Remove

Last updated on 1-4-2022, by Bob Wang

Useage: Download the following files into your working directory

Step_detect_remove.py (Detect and remove "obvious" steps in the GNSS ENU time series)

do_loop_step_detect_remove (Loop the module (Step_detect_remove.py) on a group of ENU time series files)

*.col (ENU time series, Year  NS  EW  UD ...)

Run: 
  
# ./do_loop_step_detect_remove


Output: *.col_StepFree, and several plots

THIS PROGRAM RUNS VERY SLOW!

For components, contact bob.g.wang@gmail.com

Sample output plot (black: original ENU time series; blue: step-free time series)

![MSFX_step](https://user-images.githubusercontent.com/65426380/148094103-f2837d48-aba0-4fe5-8e6d-47633b7102c9.png)
