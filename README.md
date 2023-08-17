# GNSS Step Detection and Remove

Last updated on 1-18-2022, by Bob Wang

The Python module was used in the following publication:

Wang, G.; Greuter, A.; Petersen, C.M.; Turco, M.J. Houston GNSS Network for Subsidence and Faulting Monitoring: Data Analysis Methods and Products. J. Surv. Eng. 2022, 148, doi:10.1061/(asce)su.1943-5428.0000399. https://ascelibrary.org/doi/10.1061/%28ASCE%29SU.1943-5428.0000399

[2022_HoustonNet_Data_Processing.pdf](https://github.com/bob-Github-2020/GNSS_Step_Detect_Remove/files/9993924/2022_HoustonNet_Data_Processing.pdf)


Useage: Download the following files into your working directory

GNSS_step_detection_remove.py (Detect and remove "obvious" steps in the GNSS ENU time series)

do_loop_GNSS_step_detection_remove.sh  (Loop the module 'GNSS_step_detection_remove.py' for a group of ENU time series files)

*.col (ENU time series, Year  NS  EW  UD ...)

Run: 
  
# ./do_loop_GNSS_step_detection_remove.sh


Output: *.col_StepFree, and several plots

For components, contact bob.g.wang@gmail.com

Samples of output plots (black: original ENU time series; blue: step-free time series)

![MSFX_step_remove](https://user-images.githubusercontent.com/65426380/149859503-7d11dacb-28d5-45ca-b2b7-79b2b5f88f88.png)
![MSGB_step_remove](https://user-images.githubusercontent.com/65426380/149859532-2d993dea-ae39-4b16-bc61-d9f8911e579a.png)
![MSLU_step_remove](https://user-images.githubusercontent.com/65426380/149859575-e21cf7ea-e43d-41ae-9217-9437d1057755.png)
![MSPK_step_remove](https://user-images.githubusercontent.com/65426380/149859622-c4047600-8f6c-4658-b7ee-0095b33c166a.png)
