import numpy as np
import os
from pathlib import Path
from mpi4py import MPI
# from scipy.interpolate import griddata, Rbf
import sys

def downsample_data(kinvis, resolution, numprocs, myid, dirname):
    # Run simulation with given Kinematic viscosity, resolution,
    # and numprocs and save data to npy files
    # ==> This is done in navier_2dfocs_generatedata.py

    # Load data from npy files split across processes and 
    # concatenate into single npy file
    # Each entry in one of the lists below represents data 
    # output from one processor
    x = []
    y = []
    x_ds = []
    y_ds = []
    halfpt = []
    halfpt_ds = []
    velsx = []
    velsy = []
    pressure = []
    
    for i in range(numprocs):
        smyid = '{:0>6d}'.format(i)
        nodevals = np.load(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        velvals = np.load(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        pressvals = np.load(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)

        nodevals_ds = np.load(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_ds_{smyid}.npy', allow_pickle=True)
        
        #mfem outputs single vector for nodes, 
        #where first half is x-coords and second half is y-coords
        halfpt = halfpt + [round(len(nodevals)/2)]
        x = x + [nodevals[0:halfpt[i]]]
        y = y + [nodevals[halfpt[i]:]]
        velsx = velsx + [velvals[0:halfpt[i]]]
        velsy = velsy + [velvals[halfpt[i]:]]
        pressure = pressure + [pressvals]

        halfpt_ds = halfpt_ds + [round(len(nodevals_ds)/2)]
        x_ds = x_ds + [nodevals_ds[0:halfpt_ds[i]]]
        y_ds = y_ds + [nodevals_ds[halfpt_ds[i]:]]
    
    xall = np.concatenate(x)    # x coordinates of velocity field
    yall = np.concatenate(y)    # y coordinates of velocity field
    velsxall = np.hstack(velsx) # x velocity values 
    velsyall = np.hstack(velsy) # y velocity values
    pressureall = np.hstack(pressure) # pressure

    xall_ds = np.concatenate(x_ds)    # x coordinates of velocity field
    yall_ds = np.concatenate(y_ds)    # y coordinates of velocity field

    #Interpolate velocities and pressures
    nodeslowres = np.vstack((xall_ds,yall_ds)).T
    nodeshires = np.vstack((xall,yall)).T

    #find where nodes match
    k = []
    for i in range(len(nodeslowres)):
        for j in range(len(nodeshires)):
            if abs(nodeslowres[i,0] - nodeshires[j,0]) == 0 and abs(nodeslowres[i,1] - nodeshires[j,1]) == 0:
                k.append(j)
                break

    velsx_ds = velsxall[k]
    velsy_ds = velsyall[k]
    pressure_ds = pressureall[k]

    nodenum = 0
    for i in range(myid):
        nodenum = nodenum + len(x_ds[i])
    
    vel_out = np.concatenate((velsx_ds[nodenum:(nodenum+len(x_ds[myid]))],velsy_ds[nodenum:(nodenum+len(x_ds[myid]))]))

    pressure_out  = pressure_ds[nodenum:(nodenum+len(x_ds[myid]))]

    return vel_out , pressure_out 
