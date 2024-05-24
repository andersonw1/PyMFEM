import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def extract_data(kinvis, resolution, numprocs, dirname):
    # TODO
    # Run simulation with given Kinematic viscosity, resolution,
    # and numprocs and save data to npy files

    # Load data from npy files split across processes and 
    # concatenate into single npy file
    # Each entry in one of the lists below represents data 
    # output from one processor
    x = []
    y = []
    halfpt = []
    velsx = []
    velsy = []
    
    for i in range(numprocs):
        smyid = '{:0>6d}'.format(i)
        # nodevals = np.load(f'{dirname}/Res_{resolution}_kinvis_{kinvis}_nodessmall2{smyid}.npy', allow_pickle=True)
        # velvals = np.load(f'{dirname}/Res_{resolution}_kinvis_{kinvis}_velssmall2{smyid}.npy', allow_pickle=True)
        nodevals = np.load(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        velvals = np.load(f'{dirname}/velocities_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        pressvals = np.load(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        
        #mfem outputs single vector for nodes, 
        #where first half is x-coords and second half is y-coords
        halfpt = halfpt + [round(len(nodevals )/2)]
        x = x + [nodevals[0:halfpt[i]]]
        y = y + [nodevals[halfpt[i]:]]
        velsx = velsx + [velvals[:,0:halfpt[i]]]
        velsy = velsy + [velvals[:,halfpt[i]:]]
    
    xall = np.concatenate(x)    # x coordinates of velocity field
    yall = np.concatenate(y)    # y coordinates of velocity field
    velsxall = np.hstack(velsx) # x velocity values 
    velsyall = np.hstack(velsy) # y velocity values

    np.save(f'{dirname}/xCoord_Res_{resolution}_kinvis_{kinvis}.npy', xall)
    np.save(f'{dirname}/yCoord_Res_{resolution}_kinvis_{kinvis}.npy', yall)
    np.save(f'{dirname}/xVelocity_Res_{resolution}_kinvis_{kinvis}.npy', velsxall)
    np.save(f'{dirname}/yVelocity_Res_{resolution}_kinvis_{kinvis}.npy', velsyall)
    

if __name__ == '__main__':
  my_parser = argparse.ArgumentParser()
  my_parser.add_argument("--kinvis", type=float, default=1e2, help="Reynold's number for current simulation")
  my_parser.add_argument("--resolution", type=str, default='high', help="Resolution of simulation (either `high` or `low`)")
  my_parser.add_argument("--numprocs", type=int, default=8, help="Number of processes when running simulation")
  my_parser.add_argument("--dirname", type=str, default='/usr/workspace/zeroml/mfem-data', help="Directory in which to save results")
  args = my_parser.parse_args()

  # Update directory name based on Kinematic viscosity/Resolution
  my_dirname = f'{args.dirname}/Resolution-{args.resolution}/kinvis-{args.kinvis}'
  Path(my_dirname).mkdir(parents=True, exist_ok=True)

  # Run single simulation with given Kinematic viscosity and resolution
  extract_data(kinvis=args.kinvis,
             resolution=args.resolution,
             numprocs=args.numprocs,
             dirname=my_dirname)
