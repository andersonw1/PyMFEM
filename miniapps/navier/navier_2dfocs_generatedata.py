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
from mpi4py import MPI

#set up simulation to run in parallel
num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank
smyid = '{:0>6d}'.format(myid)
   
def run(ser_ref_levels=0,
        order=4,
        kinvis=0.001,
        t_final = 5,
        dt = 1e-3,
        pa = True,
        ni = False,
        visualization = False,
        numba = True,
        resolution = 0):
    
    vels = []
    
    #choose mesh
    if resolution == 0: #low res
         mesh = mfem.Mesh("rect-cylinder_lowres.msh")
    else: #hi res
         mesh = mfem.Mesh("rect-cylinder_hires.msh")

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
    p_gf.Save(sol_name_v, 16)

    #save numpy arrays
    np.save(f'velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',u_gf.GetDataArray(), allow_pickle=True)
    np.save(f'pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',p_gf.GetDataArray(), allow_pickle=True)

    #get/save the nodes on each processor
    velgf = mfem.ParGridFunction(pmesh,sol_name_v)
    nodes = mfem.ParGridFunction(velgf)
    pmesh.GetNodes(nodes)
    coords =  nodes.GetDataArray()

    np.save(f'nodes_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',coords)
    
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

    while last_step == False:
        if time + dt >= t_final - dt/2:
            last_step = True

        
        time = flowsolver.Step(time, dt, step) #t should update in here

        #save pressure and velocity
        #there is probably a better way to do this that doesn't require 
        #reloading the array at every timestep. But, I don't know how to 
        #dereference pointers in PyMFEM so this will work for now

        vel = np.load(f'velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        vel = np.vstack((vel,u_gf.GetDataArray()))
        np.save(f'velocity_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',np.array(vel, dtype=object), allow_pickle=True)
            
        pressure = np.load(f'pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy', allow_pickle=True)
        pressure = np.vstack((pressure,p_gf.GetDataArray()))
        np.save(f'pressure_resolution_{resolution}_kinvis_{kinvis}_{smyid}.npy',np.array(pressure, dtype=object), allow_pickle=True)

        if visualization and step % 10 == 0:
            pvdc.SetCycle(step)
            pvdc.SetTime(time)
            pvdc.Save()

        if MPI.ROOT:
             print(" "*7 + "Time" + " "*10 + "dt" )
             print(f'{time:.5e} {dt:.5e} \n')


        step = step + 1
    
    flowsolver.PrintTimingData()
    
    # mesh_name = "nav2d-mesh."+smyid
    # pmesh.Print(mesh_name, 16)
    # u_gf.Save(sol_name_v, 16)
    # p_gf.Save(sol_name_p, 16)


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='navier_mms (translated from miniapps/navier/navier_mms.cpp)')

    parser.add_argument('-rs', '--refine-serial',
                        action='store', default=0, type=int,
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
    parser.add_argument("-res", "--resolution",
                        default=0,action='store', type=int,
                        help='Mesh file to use. Zero for low res, otherwise hi-res')
   
    args = parser.parse_args()
    parser.print_options(args)

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
        resolution=args.resolution)

