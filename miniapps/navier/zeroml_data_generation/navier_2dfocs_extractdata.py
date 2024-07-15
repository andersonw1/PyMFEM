import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI

def extract_data(kinvis, resolution, numprocs, dirname):
    # Run simulation with given Kinematic viscosity, resolution,
    # and numprocs and save data to npy files
    # ==> This is done in navier_2dfocs_generatedata.py

    # Load data from npy files split across processes and 
    # concatenate into single npy file
    # Each entry in one of the lists below represents data 
    # output from one processor
    x = []
    y = []
    halfpt = []
    velsx = []
    velsy = []
    pressure = []
    
    for i in range(numprocs):
        smyid = '{:0>6d}'.format(i)
        nodevals = np.load(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        velvals = np.load(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        pressvals = np.load(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        
        #mfem outputs single vector for nodes, 
        #where first half is x-coords and second half is y-coords
        halfpt = halfpt + [round(len(nodevals)/2)]
        x = x + [nodevals[0:halfpt[i]]]
        y = y + [nodevals[halfpt[i]:]]
        velsx = velsx + [velvals[:,0:halfpt[i]]]
        velsy = velsy + [velvals[:,halfpt[i]:]]
        pressure = pressure + [pressvals]
    
    # Just to make the variables below clear:
    # A node on the mesh is given by the coordinate pair (xall[i], yall[i]).
    # If 2D velocity is given by the vector u = [u_1, u_2] 
    # then at a given node, u_1 = velsxall[i] and u_2 = velsyall[i]

    xall = np.concatenate(x)    # x coordinates of velocity field
    yall = np.concatenate(y)    # y coordinates of velocity field
    velsxall = np.hstack(velsx) # x velocity values 
    velsyall = np.hstack(velsy) # y velocity values
    pressureall = np.hstack(pressure) # y velocity values

    # TODO: Dinos could update to hdf5 here
    np.save(f'{dirname}/xCoord_Res_{resolution}_kinvis_{kinvis}.npy', xall)
    np.save(f'{dirname}/yCoord_Res_{resolution}_kinvis_{kinvis}.npy', yall)
    np.save(f'{dirname}/xVelocity_Res_{resolution}_kinvis_{kinvis}.npy', velsxall)
    np.save(f'{dirname}/yVelocity_Res_{resolution}_kinvis_{kinvis}.npy', velsyall)
    np.save(f'{dirname}/Pressure_Res_{resolution}_kinvis_{kinvis}.npy', pressureall)

    

if __name__ == '__main__':
  my_parser = argparse.ArgumentParser()
  my_parser.add_argument("--kinvis", 
                          type=float, 
                          default=1e2, 
                          help="Reynold's number for current simulation")
  my_parser.add_argument("-res", "--resolution", 
                          type=str, 
                          default='high', 
                          help="Resolution of simulation (either `high` or `low`)")
  my_parser.add_argument("--numprocs", 
                          type=int, 
                          default=8, 
                          help="Number of processes when running simulation")
  my_parser.add_argument("--dirname", 
                          type=str, 
                          default='/usr/workspace/zeroml/mfem-data', 
                          help="Directory in which to save results")
  args = my_parser.parse_args()

  # Update directory name based on Kinematic viscosity/Resolution
  my_dirname = f'{args.dirname}/resolution-{args.resolution}/kinvis-{args.kinvis}'
  print(f"Creating directory {my_dirname} (if is does not already exist)")
  Path(my_dirname).mkdir(parents=True, exist_ok=True)

  # Run single simulation with given Kinematic viscosity and resolution
  extract_data(kinvis=args.kinvis,
             resolution=args.resolution,
             numprocs=args.numprocs,
             dirname=my_dirname)
