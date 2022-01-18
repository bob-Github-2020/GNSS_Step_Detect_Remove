#!/usr/bin/python3
## Last updated 1-18-2022
## updated with rpt.KernelCPD. It is much FASTer now!!
## Decide to use rpt.KernelCPD, https://centre-borelli.github.io/ruptures-docs/examples/kernel-cpd-performance-comparison/
## rpt.Pelt--rbf is too slow, almost can not be used for long GNSS time series
## l1 and L2 are better, but still slow!!
## Decide to use rpt.KernelCPD
## algo1 = rpt.KernelCPD(kernel="linear", min_size=mz).fit(ts10), AMAZING!!!! Fast and same results with rpt.Pelt

# https://medium.com/dataman-in-ai/finding-the-change-points-in-a-time-series-95a308207012

## Useage, put the following files in your working directory:
# Step_detect_remove.py, do_loop_step_detect_remove, *.col
# RUN: ./do_loop_step_detect_remove.sh

# The main output file is: *.col_StepFree

import os
import math
import csv
import numpy as np
import pandas as pd
import random
import ruptures as rpt
import changefinder
import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

from pandas import read_csv
plt.rcParams.update({'font.size': 14})


# Input GPS data
ts = []

  # fin = 'MSFX_GOM20_neu_cm.col'

# Read the "fin" from a file, I use Bshell to loop
f = open('process.ctl', 'r')
ftxt = f.readline()
fin = ftxt.rstrip('\n')
f.close()

print (fin)
gnss = fin[0:4]
print (gnss)

# ts1 = pd.read_csv (fin, delimiter=r"\s+", header=1, index_col=0, usecols=[0, 1])
# ts = pd.read_csv (fin, delimiter=r"\s+", header=1, index_col=0, usecols= [0, 2])
ts = pd.read_csv (fin, header=0, delim_whitespace=True)
# print (ts)
xx = ts.iloc[:,0]
ts1 = ts.iloc[:,1]
ts2 = ts.iloc[:,2]
ts3 = ts.iloc[:,3]

# print (ts1)

# Use the offline rupture module
# https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html
# http://dev.ipol.im/~truong/ruptures-docs/build/html/general-info.html#user-guide
# https://gist.githubusercontent.com/dataman-git/2bd0c16250c775576a0fd200de724550/raw/6cb7b879d2ffab7edc7b38fe328c5589880ad9b4/ruptures
# https://techrando.com/2019/08/14/a-brief-introduction-to-change-point-detection-using-python/

# https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/

##--------------------------------------------------------------------
# Detect the change points
## --------------------------------------------------------------------
# model = "l1"  # "l2", "rbf"  (L1 and L2 are much faster than rbf)
mod = "l1"
pn = 60
mz = 60
# algo1 = rpt.Pelt(model="rbf").fit(ts10)
# lgo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
# min_size=10, all change points will be at least 10 samples apart.
# jump controls the grid of possible change points; for instance, if jump=k, only changes at k, 2*k, 3*k,... are considered.

ts10 = ts1.values.reshape(-1,1)
#algo1 = rpt.Pelt(model=mod, min_size=mz).fit(ts10)
algo1 = rpt.KernelCPD(kernel="linear", min_size=mz).fit(ts10)
change_location1 = algo1.predict(pen=pn)

ts20 = ts2.values.reshape(-1,1)
#algo2 = rpt.Pelt(model=mod,min_size=mz).fit(ts20)
algo2 = rpt.KernelCPD(kernel="linear", min_size=mz).fit(ts20)
change_location2 = algo2.predict(pen=pn)

ts30 = ts3.values.reshape(-1,1)
#algo3 = rpt.Pelt(model=mod,min_size=mz).fit(ts30)
algo3 = rpt.KernelCPD(kernel="linear", min_size=mz).fit(ts30)
change_location3 = algo3.predict(pen=pn)

print (change_location1)
print (change_location2)
print (change_location3)

# write CPD to a file
file1 = open(gnss+'.CPD', 'w')
# Writing a string to file
file1.write("NS: "+str(change_location1)+'\n')
file1.write("EW: "+str(change_location2)+'\n')
file1.write("UD: "+str(change_location3)+'\n')
# Closing file
file1.close()

#--------------------------------------------------------------------------------------------------------
## ---Remove identified STEPS-----
## --------------------------------------------------------------------------------------------------------
def cal_step(y,istep,nav):
    y1=np.zeros(nav)
    y2=np.zeros(nav)
    y1=y[(istep-nav):(istep-1)]
    y2=y[(istep+1):(istep+nav)]
    sy1=sum(y1)/len(y1)
    sy2=sum(y2)/len(y2)
    step=sy1-sy2
    return step
    
nx=len(xx)    
x=np.reshape(xx,nx)
nav=15
ngap=60/365.25

nend=len(ts10)
ts100=np.reshape(ts10,nend)
for ist in change_location1:
    ystep=np.zeros(len(ts100))
    if ist < nend-nav:
       step=cal_step(ts100,ist,nav)
       xgap=x[ist+1]-x[ist-1]
       if abs(step) > 0.5:     
          if abs(step) < 1 and xgap > ngap:
             print('Gap related CPD, no adjust!')
          else:
             ystep[ist:nend]=step
             ts100=ts100+ystep      
       else:
          print('Step < 5 mm') 
    else: 
       print('End CPD1!')

nend=len(ts20)
ts200=np.reshape(ts20,nend)
for ist in change_location2:
    ystep=np.zeros(len(ts200))
    if ist < nend-nav:
       step=cal_step(ts200,ist,nav)
       xgap=x[ist+1]-x[ist-1]
       if abs(step) > 0.5:
          if abs(step) < 1 and xgap > ngap:
             print('Gap related CPD, no adjustment!')
          else:
             ystep[ist:nend]=step
             ts200=ts200+ystep
       else:
          print('Step < 5 mm')
    else: 
       print('End CPD2!')
       
nend=len(ts30) 
ts300=np.reshape(ts30,nend)
for ist in change_location3:
    ystep=np.zeros(len(ts300))
    if ist < nend-nav:
       step=cal_step(ts300,ist,nav)
       xgap=x[ist+1]-x[ist-1]
       if abs(step) > 0.8:
          if abs(step) < 1.5 and xgap > ngap:
             print('Gap related CPD, no adjust!')
          else:
             ystep[ist:nend]=step
             ts300=ts300+ystep
       else:
          print('Step < 8 mm') 
    else: 
       print('End CPD3!')

## output Step-Free ENU time series
xt=pd.DataFrame(xx)
yns=pd.DataFrame(ts100)
yew=pd.DataFrame(ts200)
yud=pd.DataFrame(ts300)

fout = fin + "_StepFree"
df = pd.concat([xt, yns,yew,yud], axis=1)
df.columns = ['Year', 'NS-cm', 'EW-cm','UD-cm']
df.to_csv(fout, header=True, index=None, sep=' ', mode='w', float_format='%.5f')
              

##-----------------------------------------------------------------------------------
# Plot the change points
##-----------------------------------------------------------------------------------

# plot_change_points(ts1,change_location1)
fig, (fig1, fig2, fig3) = plt.subplots(3, figsize=(16,10))
fig.subplots_adjust(hspace=0.3)
fig.suptitle('Detecting and Removing Steps: '+gnss, size=15,  y=0.91);
  
# fig1.plot(ts1)
fig1.scatter(xx,ts1,color='black',marker='o',s=5)
fig1.scatter(xx,ts100,color='blue',marker='+',s=3)
for x in change_location1: 
    year = xx[x-1]
    fig1.axvline(year, lw=2, color='red')

fig2.scatter(xx,ts2,color='black', marker='o', s=5)
fig2.scatter(xx,ts200,color='blue', marker='+',s=3)
for x in change_location2:
    year = xx[x-1]
    fig2.axvline(year, lw=2, color='red')
    print (year)
    
fig3.scatter(xx,ts3,color='black',marker='o', s=5)
fig3.scatter(xx,ts300,color='blue',marker='+', s=3)
for x in change_location3:
    year = xx[x-1]
    fig3.axvline(year, lw=2, color='red')

fig1.set_ylabel('NS-Dis. (cm)')
fig2.set_ylabel('EW-Dis. (cm)')
fig3.set_ylabel('UD-Dis. (cm)')

# fig1.set_xlabel('Year')
# fig2.set_xlabel('Year')
fig3.set_xlabel('Year')

# Save the full figure...

fig.savefig(gnss + '_step_remove.png')
fig.savefig(gnss + '_step_remove.pdf')

plt.close()
# plt.show()       
      

