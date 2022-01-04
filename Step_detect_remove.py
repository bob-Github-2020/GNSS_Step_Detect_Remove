#!/usr/bin/python3
# 1-3-2022, add REMOVE the steps, output StepFree ENU time series
# 11-04-2021
# https://medium.com/dataman-in-ai/finding-the-change-points-in-a-time-series-95a308207012
# 9-15-2021, Detect steps from GNSS-derived NEU time series
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
# model = "l1"  # "l2", "rbf"
mod = "rbf"
pn = 60
mz = 60
# algo1 = rpt.Pelt(model="rbf").fit(ts10)
# lgo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
# min_size=10, all change points will be at least 10 samples apart.
# jump controls the grid of possible change points; for instance, if jump=k, only changes at k, 2*k, 3*k,... are considered.
ts10 = ts1.values.reshape(-1,1)
algo1 = rpt.Pelt(model=mod, min_size=mz).fit(ts10)
change_location1 = algo1.predict(pen=pn)

ts20 = ts2.values.reshape(-1,1)
algo2 = rpt.Pelt(model=mod,min_size=mz).fit(ts20)
change_location2 = algo2.predict(pen=pn)

ts30 = ts3.values.reshape(-1,1)
algo3 = rpt.Pelt(model=mod,min_size=mz).fit(ts30)
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

nav=15

nend=len(ts10)
ts100=np.reshape(ts10,nend)
for ist in change_location1:
    ystep=np.zeros(len(ts100))
    if ist < nend-10:
       step=cal_step(ts100,ist,nav)
       if abs(step) > 0.5:
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
    if ist < nend-10:
       step=cal_step(ts200,ist,nav)
       if abs(step) > 0.5:
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
    print('ist=',ist)
    if ist < nend-10:
       step=cal_step(ts300,ist,nav)
       if abs(step) > 0.8:
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
fig.subplots_adjust(hspace=0.4)
fig.suptitle('Detect and Remove Steps', size=15,  y=0.93);
  
# fig1.plot(ts1)
fig1.scatter(xx,ts1,color='black',marker='o',s=2,)
fig1.scatter(xx,ts100,color='blue',s=2)

for x in change_location1: 
    year = xx[x-1]
    fig1.axvline(year, lw=2, color='red')

fig2.scatter(xx,ts2,color='black', marker='o', s=2)
fig2.scatter(xx,ts200,color='blue', s=2)
for x in change_location2:
    year = xx[x-1]
    fig2.axvline(year, lw=2, color='red')
    print (year)
    
fig3.scatter(xx,ts3,color='black',marker='o', s=2)
fig3.scatter(xx,ts300,color='blue', s=2)
for x in change_location3:
    year = xx[x-1]
    fig3.axvline(year, lw=2, color='red')

fig1.set_ylabel('Dis. (cm)')
fig2.set_ylabel('Dis. (cm)')
fig3.set_ylabel('Dis. (cm)')

fig1.set_xlabel('Year')
fig2.set_xlabel('Year')
fig3.set_xlabel('Year')


fig1.set_title(gnss+": NS")
fig2.set_title(gnss+": EW")
fig3.set_title(gnss+": UD")

# fig1.set_xlim([0,2000])

# Save the full figure...

fig.savefig(gnss + '_step.png')
fig.savefig(gnss + '_step.pdf')

plt.close()
# plt.show()       
      

