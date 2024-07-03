'''
   navier_2dfocs_generatedata.py

   2D flow over a cylinder.

   Example run with kinematic viscosity of 0.001 and low res mesh:
   python navier_2dfocs_generatedata.py -tf 5  -res 1

'''

from mfem.par import intArray, doubleArray
import mfem.par as mfem
import os
import io
import sys
from os.path import expanduser, join
from numpy import sin, cos, exp, sqrt, zeros, abs, pi
import numpy as np
from pathlib import Path
from mpi4py import MPI
import navier_2dfocs_downsample
from scipy.interpolate import griddata
import ctypes

#set up simulation to run in parallel
num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank
smyid = '{:0>6d}'.format(myid)
   
def run(ser_ref_levels=1,
        order=4,
        kinvis=0.001,
        t_final = 5,
        dt = 1e-3,
        pa = True,
        ni = False,
        visualization = False,
        numba = True,
        dirname = "./",
        skipsteps = 0,
        resolution = 0):
    
    if resolution == 0:
        downsample = True
    else:
        downsample = False
    
    if downsample == True and ser_ref_levels == 0 and myid == 0:
         print('Exiting: ser_ref_levels = 0 so we cannot downsample. Must set -rs 1 or greater if we want to downsample.')
         sys.exit()

    vels = []
    
    #choose mesh
    mesh = mfem.Mesh("rect-cylinder.msh")

    # refine mesh in MFEM.
    # currently ser_ref_levels default is 0 (no refinement)
    for i in range(ser_ref_levels):
        mesh.UniformRefinement()

    if MPI.ROOT:
        print("Number of elements: " + str(mesh.GetNE()))
    
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh) #parallel mesh

    #Setup navier solver with partial assembly
    flowsolver = mfem.navier_solver.NavierSolver(pmesh,order,kinvis)
    flowsolver.EnablePA(pa)
    
    # I think the below lines are unclear in python. 
    # u_ic points to the current flowsolver velocity, which means
    # changing u_ic sets the velocity of the flowsolver.
    # E.g. u_ic.ProjectCoefficient(u_excoeff) set the initial condition

    u_ic = flowsolver.GetCurrentVelocity()  

    if numba:
        @mfem.jit.vector(vdim=pmesh.Dimension(),td = True, interface = 'c++')
        def u_excoeff(x,t,u):
            xi = x[0]
            yi = x[1]

            if yi <= 1e-8:
                 u[1] = 1
            else:
                 u[1] = 0

            u[0] = 0.0
            
    else:
            assert False, "numba required"

    u_ic.ProjectCoefficient(u_excoeff)


    # Set boundary conditions. Inlet and cylinder are Dirichlet 0,
    # other boundaries are natural
    attr = intArray(pmesh.bdr_attributes.Max())
    attr[0] = 1 #inlet
    attr[4] = 1 #cylinder
    flowsolver.AddVelDirichletBC(u_excoeff, attr)


    #Flowsolver setup
    time = 0.0
    last_step = False

    flowsolver.Setup(dt)


    u_gf = flowsolver.GetCurrentVelocity()
    p_gf = flowsolver.GetCurrentPressure()
    
    #save velocity and pressure grid functions
    sol_name_v = f'nav2dv_{resolution}_kinvis_{kinvis}.'+smyid
    sol_name_p = f'nav2dp_{resolution}_kinvis_{kinvis}.'+smyid
    
    u_gf.Save(sol_name_v, 16)
    p_gf.Save(sol_name_p, 16)

    #save numpy arrays is skipsteps is zero, otherwise IC is saved
    #at end of skipsteps
    if skipsteps == 0:
        np.save(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',u_gf.GetDataArray(), allow_pickle=True)
        np.save(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',p_gf.GetDataArray(), allow_pickle=True)

    #get/save the nodes on each processor
    velgf = mfem.ParGridFunction(pmesh,sol_name_v)
    nodes = mfem.ParGridFunction(velgf)
    pmesh.GetNodes(nodes)
    coords =  nodes.GetDataArray()
    np.save(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',coords, allow_pickle=True)
    os.remove(sol_name_v)
    os.remove(sol_name_p)
    
    step = 0

    if visualization:
         pvdc = mfem.ParaViewDataCollection("2dfoc", pmesh)
         pvdc.SetDataFormat(mfem.VTKFormat_BINARY32)
         pvdc.SetHighOrderOutput(True)
         pvdc.SetLevelsOfDetail(order)
         pvdc.SetCycle(0)
         pvdc.SetTime(time)
         pvdc.RegisterField("velocity", u_gf)
         pvdc.RegisterField("pressure", p_gf)
         pvdc.Save()
         

    while last_step == False and skipsteps > 0:
        if time + dt >= skipsteps*dt - dt/2:
            last_step = True

        
        time = flowsolver.Step(time, dt, step) #t should update in here

        #save pressure and velocity
        #there is probably a better way to do this that doesn't require 
        #reloading the array at every timestep. But, I don't know how to 
        #dereference pointers in PyMFEM so this will work for now

        #save last timestep as IC
        if last_step:
            np.save(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',np.array(u_gf.GetDataArray(), dtype=object), allow_pickle=True)
            np.save(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',np.array(p_gf.GetDataArray(), dtype=object), allow_pickle=True)
        
        if visualization and step % 10 == 0:
            pvdc.SetCycle(step)
            pvdc.SetTime(time)
            pvdc.Save()

        if MPI.ROOT:
             print(" "*7 + "Time" + " "*10 + "dt" )
             print(f'{time:.5e} {dt:.5e} \n')

        step = step + 1


    
    # flowsolver.PrintTimingData()

    # if downsampling, we define a new flowsolver_ds  on the low-res mesh
    # and reinitialize everything
    if downsample == True:
        mesh_ds = mfem.Mesh("rect-cylinder.msh")
        for i in range(ser_ref_levels-1):
            mesh_ds.UniformRefinement()
        pmesh_ds = mfem.ParMesh(MPI.COMM_WORLD, mesh_ds) #parallel mesh
        flowsolver_ds = mfem.navier_solver.NavierSolver(pmesh_ds,order,kinvis)
        flowsolver_ds.EnablePA(pa)

        u_ic_ds = flowsolver_ds.GetCurrentVelocity() 
        p_ic_ds = flowsolver_ds.GetCurrentPressure()
        
        u_ic_ds.Save(f'nav2dv_{resolution}_kinvis_{kinvis}_ds_{smyid}', 16)

        #get the nodes on each processor
        velgf_ds = mfem.ParGridFunction(pmesh_ds,f'nav2dv_{resolution}_kinvis_{kinvis}_ds_{smyid}')
        nodes_ds = mfem.ParGridFunction(velgf_ds)
        pmesh_ds.GetNodes(nodes_ds)
        coords_ds =  nodes_ds.GetDataArray()
        np.save(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_ds_{smyid}.npy',coords_ds, allow_pickle=True)
        os.remove(f'nav2dv_{resolution}_kinvis_{kinvis}_ds_{smyid}')

        MPI.COMM_WORLD.Barrier() #make sure all coordinate files are saved for each processor
        vel, pressure = navier_2dfocs_downsample.downsample_data(kinvis, resolution, num_procs, myid, dirname)
        MPI.COMM_WORLD.Barrier() #make sure we have vel/pressure defined
        
        #remove old coord file, save new one
        os.remove(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_ds_{smyid}.npy')
        np.save(f'{dirname}/nodes_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',coords_ds, allow_pickle=True)

        u_ic_ds.Assign(mfem.Vector(vel.astype(float)))
        p_ic_ds.Assign(mfem.Vector(pressure.astype(float)))

        u_gf.Destroy()
        p_gf.Destroy()

        u_gf = flowsolver_ds.GetCurrentVelocity()
        p_gf = flowsolver_ds.GetCurrentPressure()   

        np.save(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',u_ic_ds.GetDataArray(), allow_pickle=True)
        np.save(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',p_ic_ds.GetDataArray(), allow_pickle=True)


        attr_ds = intArray(pmesh_ds.bdr_attributes.Max())
        attr_ds[0] = 1 #inlet
        attr_ds[4] = 1 #cylinder
        flowsolver_ds.AddVelDirichletBC(u_excoeff, attr_ds)
        flowsolver_ds.Setup(dt)
        time_ds = 0
        step_ds = 0

    
    last_step = False
    
    vel = np.load(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
    pressure = np.load(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
 

    while last_step == False:
        if downsample == True:
            if time_ds + dt >= t_final - dt/2:
                last_step = True
            time_ds = flowsolver_ds.Step(time_ds, dt, step_ds) #t should update in here
        else:
            if time + dt >= t_final + skipsteps*dt - dt/2:
                last_step = True
            time = flowsolver.Step(time, dt, step) #t should update in here

        vel = np.vstack((vel,np.array((ctypes.c_double * u_gf.Size()).from_address(int(u_gf.GetData())), copy=True)))
        pressure = np.vstack((pressure,np.array((ctypes.c_double * p_gf.Size()).from_address(int(p_gf.GetData())), copy=True)))


        if visualization and step % 10 == 0:
            pvdc.SetCycle(step)
            pvdc.SetTime(time)
            pvdc.Save()

        if MPI.ROOT:
             print(" "*7 + "Time" + " "*10 + "dt" )
             if downsample:
                 print(f'{time_ds + skipsteps*dt:.5e} {dt:.5e} \n')
                 step_ds = step_ds + 1
             else:
                 print(f'{time:.5e} {dt:.5e} \n')
                 step = step + 1

    np.save(f'{dirname}/velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',np.array(vel, dtype=object), allow_pickle=True)
    np.save(f'{dirname}/pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',np.array(pressure, dtype=object), allow_pickle=True)


    print('Finished Sim')



if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='navier_mms (translated from miniapps/navier/navier_mms.cpp)')

    parser.add_argument('-rs', '--refine-serial',
                        action='store', default=1, type=int,
                        help="Number of times to refine the mesh uniformly in serial.")
    parser.add_argument('-o', '--order',
                        action='store', default=4, type=int,
                        help="Order (degree) of the finite elements.")
    parser.add_argument('-kinvis', '--kinvis',
                        action='store', default=0.001, type=float,
                        help="Kinematic viscosity.")                    
    parser.add_argument('-dt', '--time-step',
                        action='store', default=1e-3, type=float,
                        help="Time step.")
    parser.add_argument('-tf', '--final-time',
                        action='store', default=5, type=float,
                        help="Final time.")
    parser.add_argument('-no-pa', '--disable-pa',
                        action='store_false',
                        help="Disable partial assembly.")
    parser.add_argument('-ni', '--enable-pa',
                        action='store_true',
                        help="Enable numerical integration rules.")
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization')
    parser.add_argument("-n", "--numba",
                        default=1, action='store', type=int,
                        help="Use Number compiled coefficient")
    parser.add_argument("--dirname", 
                        type=str,
                        default='/usr/workspace/zeroml/mfem-data', 
                        help="Directory in which to save results")
    parser.add_argument("--skipsteps",
                        default=0,action='store', type=int,
                        help='Number of timesteps to skip before downsampling')
    parser.add_argument("-res", "--resolution",
                        default=0,action='store', type=int,
                        help='Mesh file to use. Zero for low res, otherwise hi-res')
 
   
    args = parser.parse_args()
    parser.print_options(args)

    # Update directory name based on Kinematic viscosity/Resolution
    my_dirname = f'{args.dirname}/resolution-{args.resolution}/kinvis-{args.kinvis}'
    print(f"Creating directory {my_dirname} (if it does not already exist)")
    Path(my_dirname).mkdir(parents=True, exist_ok=True)

    numba = (args.numba == 1)

    run(ser_ref_levels = args.refine_serial,
        order=args.order,
        kinvis=args.kinvis,
        t_final=args.final_time,
        dt=args.time_step,
        pa=True,
        ni=False,
        visualization=args.visualization,
        numba=numba,
        dirname=my_dirname,
        skipsteps = args.skipsteps,
        resolution = args.resolution)
