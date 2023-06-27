# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:07:37 2022

@author: KNISSAN
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.optimize as spo
from pandas import DataFrame, Series 

import matplotlib.cm as cm
import os
from collections import OrderedDict
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('figure', figsize=(9, 11))

#Densities needed for our sets of particles in (kg/m^3)
al_density = 2790 
st_density = 7820 
cu_density = 8920 
pl_density = 1420 
ZrO2_density = 5680

#The five terminal velocities needed for our sets of particles in (m/s)
pl_term_vel = 0.00010080535530381048
al_term_vel = 0.00040838516992790924
st_term_vel = 0.0015376745623069
ZrO2_term_vel = 0.0010572214212152416
al_term_vel_3D = 0.000776

v_light_term = st_term_vel #Change this to your lightest particle

r = .001 #m
tau = r/v_light_term

p_heavy = cu_density #Change this to your heaviest particle
p_lighter = st_density #Change this to your lightest particle
p_fluid = 971 # density of the medium in (kg/m^3); in this case the silicone oil

K = (p_heavy - 2*p_fluid + p_lighter)/(-p_fluid + p_lighter)
K2 = (p_heavy -p_lighter)/(p_lighter - p_fluid)
scales = [12.9,11.9,13.5,13.3,12.9]
fps = 6
colormap = cm.get_cmap('tab10')

label_array=[27,28,29,30,31,32,33] # the number label to each experiment
ticksize = 22
point_spacing = 1 #every nth point
lgnd_font = 28
plot_symbols = ['o','d','s','v','*']
lw = 2.75
# (1/np.tan(theta_0/2))**2
# (1/np.sin())
def theta_t(t,theta_0,c3):
    return 2*(np.pi/2 - np.arctan((1/np.tan(theta_0/2))*np.exp(c3*K2*t)))

def x_theta(theta, x_0, a1, c1):
    return x_0 + (8/3) * (K/K2) * (a1/c1) * (np.sin(theta)-1)

def y_theta(theta, y_0, b0, b1, c1):
    return y_0 + (4/3) * (K/K2) * (2*b1*np.cos(theta) + (b0+b1)*np.log(np.tan(theta/2)))/c1

def x_sol(t,theta_0,x_0,a1,c1):
    numerator = 8*a1*K*np.sin(theta_0)*np.exp(3*c1*K2*t/4) + (3*c1*K2*x_0 - 8*a1*K*np.sin(theta_0))*((np.sin(theta_0/2))**2 + np.exp(3*c1*K2*t/2)*(np.cos(theta_0/2))**2)
    denominator = 3*c1*K2*(1-np.cos(theta_0)+np.exp(3*c1*K2*t/2)*(1+np.cos(theta_0)))
    return 2*numerator/denominator

def y_sol(t,theta_0,y_0,b0,b1,c1):
    numerator1 = (b1*K*(8 + 3*c1*K2*t) + 3*c1*K2*(b0*K*t + y_0))*(-1 + np.exp(3*c1*K2*t/2))
    numerator2 = 8*b1*K*np.cos(theta_0)*(-1+np.exp(3*c1*K2*t/2)) + (6*c1*K2*np.exp(3*c1*K2*t/2)*((b0+b1)*K*t + y_0))/(-1 + np.cos(theta_0))
    denominator = (3*c1*K2*(1+np.exp(3*c1*K2*t/2)*(1/np.tan(theta_0/2))**2))
    return (numerator1 + numerator2)/denominator

def x_sol_2(t,theta_0,x_0,c2,c3):
    F = np.exp(K2*c3*t)
    cot = np.cos(theta_0/2)/np.sin(theta_0/2)
    frac1 = (4*c2*F*K*cot)/(c3*K2+c3*(F**2)*K2*cot**2)
    frac2 = (2*c2*K*np.sin(theta_0))/(c3*K2)
    return x_0 + frac1 - frac2

def y_sol_2(t,theta_0,y_0,c1,c2,c3):
    F = np.exp(K2*c3*t)
    cot = np.cos(theta_0/2)/np.sin(theta_0/2)
    numerator = 2*K*(np.sin(theta_0/2)**2)*((c1+c2)*c3*K2*t - 2*c2 + 2*c2*(np.cos(theta_0) + 2/(1 + F**2 * cot**2)))
    denominator = c3*K2*(-1 + np.cos(theta_0))
    return y_0 + numerator/denominator
def x_sol_lim(t,theta_0,x_0,c2):
    return x_0 - c2*K*t*np.sin(2*theta_0)

def y_sol_lim(t,theta_0,y_0,c1,c2):
    return y_0 - K*t*(c1 + c2*np.cos(2*theta_0))

def x_theta_mac(theta,theta_0,x_0,b0_a1,c1):
    return x_0 + (16/3) * (K/K2) * ((b0_a1)/c1) * (np.sin(theta) - np.sin(theta_0))
                  
def y_theta_mac(theta,theta_0,y_0,b0,b0_a1,c1):
    return y_0 + (4/3) * (K/K2)*((b0_a1)*(np.cos(theta)-np.cos(theta_0)) + b0*np.log(np.tan(theta/2)/np.tan(theta_0/2)))/c1



theta_0_arr = []
x_0_arr = []
y_0_arr = []


a1_arr = []
a1_arr_2 = []
a2_arr = []
b0_arr= []
b1_arr= [] 

c1_arr = []
c1_errs = []
c2_arr = []
c2_errs = []
c3_arr = []
c3_errs = []
theta_sols = []
count = 0

initial_guesses = [.06,.067,.15,.001]

#at,bt,ar
initial_guesses_mac = [.23,.258,.267]

fig, ax = plt.subplots(3,1,sharex='all')

ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

ax1.tick_params(direction = 'in',length = 8, width = 1.75,labelsize=20)
ax2.tick_params(direction = 'in',length = 8, width = 1.75,labelsize=20)
ax3.tick_params(direction = 'in',length = 8, width = 1.75,labelsize=20)
ax5 = plt.axes([0,0,1,1])
t_half_arr = []

fig.subplots_adjust(hspace=0)

for data_loc in sorted(os.listdir(r'Z:\Mingxuan Liu\Particle Trajectories')):
    #Reading in data and setting up counter
    count+=1
    print(data_loc)
    color = colormap(count-1)
    file_loc = os.path.join(r'Z:\Mingxuan Liu\Particle Trajectories',data_loc)
    particle_data = pd.read_csv(file_loc)
    
    #scale = scales[count-1] #scales array determined through imagej
    scale = 7.12 #3D Scale
    
    #Retrieving positions, angles, and time arrays
    #p0 is heavier
    #p1 is lighter
    
    
    p0 = particle_data[particle_data['particle'] == 0]
    p1 = particle_data[particle_data['particle'] == 1]
    
    m0 = 4/3 * np.pi * (.001)**3 * p_heavy  # mass of the heavy object
    m1 = 4/3 * np.pi * (.001)**3 * p_lighter  # mass of the lighter object
    frames = np.copy(p0['frame'])
    
    r_x = p1['x'].to_numpy() - p0['x'].to_numpy()  # horizontal distances between two objects
    r_y = p1['y'].to_numpy() - p0['y'].to_numpy()  # vertical distances between two objects

    # center of mass between two obejcts
    COM = np.array([(m0*p0['x'].to_numpy() + m1*p1['x'].to_numpy())/(m1+m0), (m0*p0['y'].to_numpy() + m1*p1['y'].to_numpy())/(m1+m0)])
    COM = COM/scale
    # center of geometry between two objects
    center = np.array([p0['x'].to_numpy()/2 + p1['x'].to_numpy()/2, p0['y'].to_numpy()/2 + p1['y'].to_numpy()/2 ])

    cen_x = center[0]/scale
    cen_y = center[1]/scale
    COM_x = COM[0]  # horizontal component of COM
    COM_y = COM[1]  # vertical component of COM
    
    #We can choose to use either the COM frame or the center of geometry
#    x_data = COM_x - COM_x[0]
#    y_data = COM_y - COM_y[0]
    
    x_data = cen_x - cen_x[0]  # set the initial horizontal position of the geometric center as reference
    y_data = cen_y - cen_y[0]  # set the initial vertical position of the geometric center as reference
    
    angle = np.arctan2(r_y/scale,r_x/scale) + np.pi/2  # retrun the angle value between -pi and pi
#    angle = np.arctan(r_y/scale,r_x/scale) + np.pi/2
    time = ((frames - frames[0])/fps)/tau  # calculate the scaled time with tau as the time scale
    
    

    
    # Calculating the time where angle = pi/2
    # Finds the closest point to pi/2, creates a line from the points close by, then solve for t_half knowing we want pi/2
    idx_half_close = (np.abs(angle - np.pi/2)).argmin()
    
    line_fit_points_idx = idx_half_close + np.array([-2,-1,0,1,2])
    
    line_fit_points = angle[line_fit_points_idx]
    time_half_fit = time[line_fit_points_idx]

    def line(t,M,B):
        return M*t + B
    t_h_popt,t_h_pcov = spo.curve_fit(line,time_half_fit,line_fit_points)
    
    t_half = (np.pi/2 - t_h_popt[1])/t_h_popt[0]
    t_half_arr.append(t_half)
    
    #Fitting c3
#    c_init = initial_guesses[0]
    c3_init = initial_guesses_mac[2]
    theta_fit = spo.curve_fit(theta_t,time,angle,p0=[angle[0],c3_init])
    popt_ang = theta_fit[0]
    pcov_ang = theta_fit[1]
    c3_err = np.sqrt(np.diag(pcov_ang))[1]
    c3_errs.append(c3_err)
    theta_0 = popt_ang[0]  # best-fit initial angle
    theta_0_arr.append(popt_ang[0])
    c3 = popt_ang[1]
    c3_arr.append(c3)
    ang_sol = theta_t(time,popt_ang[0],popt_ang[1])
    
    #Fitting c1 and c2

    c1_init = initial_guesses_mac[0]/2 + initial_guesses[1]/2
    c2_init = -initial_guesses_mac[0]/2 + initial_guesses[1]/2
    

    

    
    x_fit = spo.curve_fit(lambda t,x_0,c2:x_sol_2(t,theta_0,x_0,c2,c3),time,x_data,p0=[x_data[0],c2_init])
#    x_fit = spo.curve_fit(lambda t,x_0,c2:x_sol_lim(t,theta_0,x_0,c2),time,x_data,p0=[x_data[0],c2_init])
    popt_x = x_fit[0]
    pcov_x = x_fit[1]
    c2_err = np.sqrt(np.diag(pcov_x))[1]
    c2_errs.append(c2_err)
    x_0 = popt_x[0]
    c2 = popt_x[1]
    c2_arr.append(c2)
    
    y_fit = spo.curve_fit(lambda t,y_0,c1:y_sol_2(t,theta_0,y_0,c1,c2,c3),time,-y_data,p0=[y_data[0],c1_init])
#    y_fit = spo.curve_fit(lambda t,y_0,c1:y_sol_lim(t,theta_0,y_0,c1,c2),time,-y_data,p0=[y_data[0],c1_init])
    popt_y = y_fit[0]
    pcov_y = y_fit[1]
    c1_err = np.sqrt(np.diag(pcov_y))[1]
    c1_errs.append(c1_err)
    y_0 = popt_y[0]
    c1 = popt_y[1]
    c1_arr.append(c1)
    
    x_0_arr.append(x_0)
    y_0_arr.append(y_0)

    
    
#    x_solution = x_sol_lim(time,theta_0,x_0,c2)
#    y_solution = y_sol_lim(time,theta_0,y_0,c1,c2)
    
    
    x_solution = x_sol_2(time,theta_0,x_0,c2,c3)
    y_solution = y_sol_2(time,theta_0,y_0,c1,c2,c3)
    
    marker = 'o'
    idx_t_half = (np.abs(time-t_half)).argmin()
    

    
    # Plotting y vs. t and inset
    ax2.scatter(time[::point_spacing]-t_half,-y_data[::point_spacing] + y_data[idx_t_half],s=200,fc = 'none',edgecolor=color,label = 'Data',marker=marker)
    ax2.plot(time-t_half,y_solution - y_solution[idx_t_half],linewidth=lw,color=color,label='Model')
    
    # ax2.scatter(time[::point_spacing],-y_data[::point_spacing],s=200,fc = 'none',edgecolor=color,label = 'Data',marker=marker)
    # ax2.plot(time,y_solution,linewidth=lw,color=color,label='Model')
    
    #plt.xlabel('time (scaled)',fontsize = 48)
    ax2.set_ylabel('y/R',fontsize = 30)
    ax5.tick_params(direction = 'in',length = 6, width = .825,labelsize=15)
    ip = InsetPosition(ax2, [.68,.6,.3,.35])
    ax5.set_axes_locator(ip)
    ax5.axhline(y=0,linestyle=":",color='k',linewidth=1)
    ax5.plot(time-t_half,y_data + y_solution,linewidth=lw-1.75,color=color) #Put back t-t_half
    ax5.set_ylabel('y - y$_{m}$',labelpad=3,fontsize=20)
    ax5.set_xlabel(r't/$\tau $',labelpad=-1.5,fontsize=20)
    ax5.set_ylim(-.5,.5)
#    ax5.set_xlim(time[0]-2-t_half,time[-1]+2-t_half)
    
    # Plotting x vs. t
    ax3.scatter(time[::point_spacing]-t_half,x_data[::point_spacing] - x_data[idx_t_half],s=200,fc = 'none',edgecolor=color,label = 'Data',marker=marker)
    ax3.plot(time-t_half,x_solution - x_solution[idx_t_half],linewidth=lw,color=color,label='Model')
    
    # ax3.scatter(time[::point_spacing],x_data[::point_spacing],s=200,fc = 'none',edgecolor=color,label = 'Data',marker=marker)
    # ax3.plot(time,x_solution,linewidth=lw,color=color,label='Model')
    
    ax3.set_xlabel(r't/$\tau $',fontsize = 30)
    ax3.set_ylabel('x/R',fontsize = 30,labelpad=-5)
    
    #Plotting theta vs time
    y_lbls = ['$\pi$','$\pi/2$','0']
    y_lbl_val = [np.pi,np.pi/2,0]
    
    ax1.set_yticks(y_lbl_val)
    ax1.set_yticklabels(y_lbls)
    marker = 'o'
    
    ax1.scatter(time[::point_spacing]-t_half,angle[::point_spacing],s=200,fc='none',edgecolor = color,label = 'Data',marker=marker) #data_loc[20]
    ax1.plot(time-t_half,ang_sol,linewidth=lw, color=color,label='Model')
    
    # ax1.scatter(time[::point_spacing],angle[::point_spacing],s=200,fc='none',edgecolor = color,label = 'Data',marker=marker) #data_loc[20]
    # ax1.plot(time,ang_sol,linewidth=lw, color=color,label='Model')
#    ax1.set_ylim(0,3*np.pi/2)
    theta_sols.append(ang_sol)
    ax1.set_ylabel(r'$ \theta $',fontsize = 30,rotation=0,labelpad=25)
    
    
    savepath = r'Z:\Mingxuan Liu\Tracking Analysis'
    
    data_dict = {'t':time,'x':x_data,'y':y_data,'theta':angle}
    df = pd.DataFrame(data_dict)
    print(savepath + '\St+Cu 3D '+str(label_array[count-1])+'.csv')
    df.to_csv(savepath + '\St+Cu 3D '+str(label_array[count-1])+'.csv')
    
plt.show()
param_table_lf = pd.DataFrame(columns = ['c1','c1_err','c2','c2_err','c3','c3_err'])
param_table_bf = pd.DataFrame(columns = ['at','bt','ar'])
for n in range(len(label_array)):
#    row_arr1 = np.array([a1_arr[n],b0_arr[n],b1_arr[n],c1_arr[n]])
#    row_arr_mac = np.array([a1_arr[n],b0_arr[n],c1_arr[n]])
    c1 = c1_arr[n]
    c1_err = c1_errs[n]
    c2 = c2_arr[n]
    c2_err = c2_errs[n]
    c3 = c3_arr[n]
    c3_err = c3_errs[n]
    row_arr_bf = np.array([c1-c2,c1+c2,4*c3/3])
    row_arr_lf = np.array([c1,c1_err,c2,c2_err,c3,c3_err])
    label = label_array[n]
    param_table_bf.loc['3D Cu+St '+str(label)] = row_arr_bf
    param_table_lf.loc['3D Cu+St '+str(label)] = row_arr_lf
print(param_table_bf)
print(param_table_lf)
param_table_bf.to_csv(r'Z:\Mingxuan Liu\Tracking Analysis\3D Cu+St Param Table bf.csv')
param_table_lf.to_csv(r'Z:\Mingxuan Liu\Tracking Analysis\3D Cu+St Param Table lf.csv')
#param_table_1st.to_csv(r'F:\Sedimentation\Analysis\Al+Pl Param Table 1st order new.csv')