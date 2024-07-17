import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI
from scipy.interpolate import griddata
from matplotlib.patches import Circle

def interpolate_data(kinvis, resolution, numprocs, dirname, interpx, interpy):

    # TODO: Dinos could update to hdf5 here
    xall = np.load(f'{dirname}/xCoord_Res_{resolution}_kinvis_{kinvis}.npy')
    yall = np.load(f'{dirname}/yCoord_Res_{resolution}_kinvis_{kinvis}.npy')
    velsxall = np.load(f'{dirname}/xVelocity_Res_{resolution}_kinvis_{kinvis}.npy')
    velsyall = np.load(f'{dirname}/yVelocity_Res_{resolution}_kinvis_{kinvis}.npy')
    pressureall = np.load(f'{dirname}/Pressure_Res_{resolution}_kinvis_{kinvis}.npy')
    

    xgrid = np.linspace(xall.min(), xall.max(), interpx)
    ygrid = np.linspace(yall.min(), yall.max(), interpy)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    
    velsxall_interp = griddata((xall,yall),velsxall, (xgrid, ygrid))
    velsyall_interp = griddata((xall,yall),velsyall, (xgrid, ygrid))
    pressureall_interp = griddata((xall,yall),pressureall, (xgrid, ygrid))
    
    np.save(f'{dirname}/xCoord_Res_{resolution}_kinvis_{kinvis}_interp.npy',xgrid)
    np.save(f'{dirname}/yCoord_Res_{resolution}_kinvis_{kinvis}_interp.npy',ygrid)
    np.save(f'{dirname}/xVelocity_Res_{resolution}_kinvis_{kinvis}_interp.npy',velsxall_interp)
    np.save(f'{dirname}/yVelocity_Res_{resolution}_kinvis_{kinvis}_interp.npy',velsyall_interp)
    np.save(f'{dirname}/Pressure_Res_{resolution}_kinvis_{kinvis}_interp.npy',pressureall_interp)
    
    #example for how to plot y velocities at the ith timestep
    # fig = plt.figure()
    # fig.set_size_inches(18,9)
    # ax = fig.add_subplot(111)
    # plottime = 0
    # ax.set_title(f'$t$ =  {plottime*0.001:.2f}',fontsize=30)
    # # zgrid = griddata((xall,yall),velsyall[i,:], (xgrid, ygrid))
    # # zgrid = griddata((xall,yall),velmag[i,:], (xgrid, ygrid))
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_aspect('equal', adjustable='box')
    # pc = ax.pcolormesh(xgrid,ygrid,velsyall_interp[plottime,:], shading = 'gouraud',cmap='RdBu_r')
    # fig.colorbar(pc)
    # # ax.pcolormesh(xgrid,ygrid,zgrid, shading = 'gouraud',cmap='RdBu_r',vmax = cmax, vmin = cmin)
    # circ = Circle((0.5,0.5),0.05,color='k')
    # ax.add_patch(circ)

    

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
  my_parser.add_argument("--interpx", 
                          type=int, 
                          default=100, 
                          help="Number of points in x-direction to interpolate at")
  my_parser.add_argument("--interpy", 
                          type=int, 
                          default=200, 
                          help="Number of points in y-direction to interpolate at")
  args = my_parser.parse_args()

  # Update directory name based on Kinematic viscosity/Resolution
  my_dirname = f'{args.dirname}/resolution-{args.resolution}/kinvis-{args.kinvis}'
  print(f"Creating directory {my_dirname} (if is does not already exist)")
  Path(my_dirname).mkdir(parents=True, exist_ok=True)

  # Run single simulation with given Kinematic viscosity and resolution
  interpolate_data(kinvis=args.kinvis,
             resolution=args.resolution,
             numprocs=args.numprocs,
             dirname=my_dirname,
             interpx=args.interpx,
             interpy=args.interpy)