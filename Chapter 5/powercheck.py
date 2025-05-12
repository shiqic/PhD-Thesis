import numpy as np
from tqdm import tqdm
from scipy import constants
import miepy
import stoked
import math
from subprocess import call

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
        '''
        for i in range(0,np.size(self.position,axis=0)):
            x = self.position[i,0]
            y = self.position[i,1]
            F[i,2] = F[i,2] + U(x,y) # F[i,2] is the z component of the force on the i-th particle. The U(x,y) is the extra electrostatic force profile that depends on x and y.
        '''
        return F

tmp=np.loadtxt('../p/para.in')
natom=6
maxtime=1024000
wvl=800*nm
radius = 75*nm
density = 10490
inertia = stoked.inertia_sphere(radius,density)
water = miepy.materials.water()
Ag = miepy.materials.Ag()
wid=1750*nm
z=math.pi*(wid**2)/wvl
pow=tmp[1]/1000.0#0.1
source = miepy.sources.gaussian_beam(polarization=[1,1j], width=wid, power=pow,center=[0,0,z])
temperature = tmp[0]*1.0#100
dt =500*ns
viscosity = tmp[2]*(1e-6)
drag = stoked.drag_sphere(radius, viscosity)
Nsteps = int(1/dt)
potential = -77e-3
debye = 100.0*nm
mobility = (1.0e6)/(6*math.pi*tmp[2]*radius)

mem=maxtime

initial = np.loadtxt("../p/avgpos.in")*nm
ind=np.arange(11,-1,-1)
drc = np.loadtxt("../p/direc.out")[:,ind]
direc = drc[:,0:11]

ki = (1.38e-23)*temperature*(np.loadtxt("../p/stdev.out")[ind]**(-1))
ki[11] = 0.

r2=initial[:,0]+1j*initial[:,1]
rr2=np.matmul(np.full((mem,1),1.),r2.reshape(1,natom))

trj=nm*(np.loadtxt('trajectory.out'))
trj0=np.append(trj,np.full((mem*natom,1),0.),axis=1).reshape(mem,natom,3)

trj1=(trj[:,0]+1j*trj[:,1]).reshape(mem,natom)
th=np.sum((trj1.conj())*rr2,axis=1)
rot=np.matmul((th*(abs(th)**(-1))).reshape(mem,1),np.full((1,natom),1.))
trj1=rot*trj1

trj=trj1.reshape(mem*natom,1)
trj1=np.append(trj.real,trj.imag,axis=1)
trj=np.append(trj1,np.full((mem*natom,1),0.),axis=1).reshape(mem,natom,3)
trj1=trj1.reshape(mem,2*natom)
trj1=trj1-np.matmul(np.full((maxtime,1),1.),initial.reshape(1,2*natom))
trj1=np.matmul(trj1,drc)
f0=-np.matmul(np.full((maxtime,1),1.),ki.reshape(1,2*natom))*trj1
f0=np.matmul(f0,drc.transpose())

pwcheck=np.full((mem-1,3),0.)
for k in tqdm(range(0,mem-1)):
    cl=miepy.sphere_cluster(position=trj[k],radius=radius,material=Ag,lmax=2,source=source,wavelength=wvl, medium=water)
    bd = stoked.brownian_dynamics(position=trj[k].copy(), drag=drag, temperature=temperature, dt=dt, inertia=inertia,
                                  interactions=[electrodynamics(cl),
                                                stoked.double_layer_sphere(radius, potential, debye=debye),
                                                stoked.collisions_sphere(radius,1)], constraint=stoked.constrain_position(z=0))
    bd.interactions[1].position=trj[k].copy()
    fp=(bd.interactions[0].force()[:,0:2] + bd.interactions[1].force()[:,0:2]).reshape(natom*2)
    fpr=fp-f0[k,:]
    pwcheck[k,0]=np.dot(fp,fpr)*mobility
    pwcheck[k,1]=np.dot(fpr,fpr)*mobility
    
    trjtmp=0.5*(trj0[k]+trj0[k+1])
    cl=miepy.sphere_cluster(position=trjtmp,radius=radius,material=Ag,lmax=2,source=source,wavelength=wvl, medium=water)
    bd = stoked.brownian_dynamics(position=trjtmp.copy(), drag=drag, temperature=temperature, dt=dt, inertia=inertia,
                                  interactions=[electrodynamics(cl),
                                                stoked.double_layer_sphere(radius, potential, debye=debye),
                                                stoked.collisions_sphere(radius,1)], constraint=stoked.constrain_position(z=0))
    # In interactions = [], more interactions can be added, including an extra z-direction electrostatic force.
    bd.interactions[1].position=trjtmp.copy() # Store the position for calculation of double layer interaction. This step should be done before running "bd.interactions.force()".
    # The variable fp is the sum of the electrodynamic interaction (stored in bd.interactions[0]) and the double layer interaction (stored in bd.interactions[1]).
    fp=(bd.interactions[0].force()[:,0:2] + bd.interactions[1].force()[:,0:2]).reshape(natom*2)
    # Likewise, collision can be found in bd.interaction[2].force(). 
    pwcheck[k,2]=(1/dt)*np.dot(fp,(trj0[k+1,:,0:2]-trj0[k,:,0:2]).reshape(natom*2))

pwch=np.mean(pwcheck,axis=0).reshape(3,1)
pwch=np.append(pwch,(maxtime**(-0.5))*np.std(pwcheck,axis=0).reshape(3,1))
np.savetxt('powercheck_wtksi_DL.out',pwch)
