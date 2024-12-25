import numpy as np
from tqdm import tqdm
import stoked 
import miepy

temperature = 300.
nm = 1e-9
ns = 1e-9
us = 1e-6
ms = 1e-3

class electrodynamics(stoked.interactions):
    def __init__(self, cluster):
        self.cluster = cluster

    def update(self):
        self.cluster.update_position(self.position)

    def force(self):
        F = self.cluster.force()
        return F

natom = 8
length = 1000    # number of the structures in the input data obtained in experiment
wvl = 800*nm    # vacuum wavelength of laser
radius = 75*nm
density = 10490
inertia = stoked.inertia_sphere(radius,density)
water = miepy.materials.water()
Ag = miepy.materials.Ag()
wid = 2000*nm    # beam width
z = 0.         # defocus
dt = 1000*ns     # simulation time step
viscosity = 8e-4  # viscosity of water at 300 K
drag = stoked.drag_sphere(radius, viscosity)
Nsteps = int(1/dt)
potential = -77e-3 # surface potential of Ag NP

# set beam power and polarization (in Jones vector)
source = miepy.sources.gaussian_beam(polarization=[1,1j], width=wid, power=0.05, center=[0,0,z])

dat = np.loadtxt('experiment.in').reshape(length,natom,2)  # read the input data obtained in experiment
dthreshold = 3e-7  # set and tune the d1-d2 threshold for the criterion of reaching the state d1<d2 or the state d1>d2
mem = 2000 # (tune) time steps taken to reach either the state d1<d2 or the state d1>d2
count = [0,0] # record both the number of states that reach the state d1<d2 first and the number of states that reach the state d1>d2 first
committor = [] # initialize the output committor

for k in range(0,length):
    cluster = miepy.sphere_cluster(position=np.append(dat[k,:,:],np.full((natom,1),0.),axis=1).copy(),
                                   radius=radius,
                                   material=Ag,
                                   source=source,
                                   wavelength=wvl,
                                   medium=water,
                                   lmax=2)
    bd = stoked.brownian_dynamics(position=initial.copy(), drag=drag, temperature=temperature, dt=dt, inertia=inertia,
                           interactions=[electrodynamics(cluster),
                                         stoked.double_layer_sphere(radius, potential, debye=27.6*nm), # set Debye screening length
                                         stoked.collisions_sphere(radius,1)], constraint=stoked.constrain_position(z=0))
    trj = bd.run(mem+1).position
    trj = trj.reshape(natom*(mem+1),3)
    fl = 0
    for i in range(0,mem+1):
        if (fl==0):
            struct = trj[i].copy()
            # d1 = ... Calculate d1 of the ith structure.
            # d2 = ... Calculate d2 of the ith structure.
            if (d1-d2 > dthreshold):
                count[0]=count[0]+1
                l = 1
            if (d2-d1 > dthreshold):
                count[1]=count[1]+1
                l = 1
    committor.append(1.*count[0]/count[1])
np.savetxt('committor.out',np.asarray(committor))
