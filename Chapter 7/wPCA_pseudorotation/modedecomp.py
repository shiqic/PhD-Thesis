import numpy as np
from tqdm import tqdm

nm = 1e-9

natom=8
stp=2
maxtime=400000//stp
wvl=800*nm

initial = np.loadtxt("initpos.in")*nm

r2=initial[:,0]+1j*initial[:,1]
rr2=np.matmul(np.full((maxtime,1),1.),r2.reshape(1,natom))
errthr=nm
npro=100

l=[]; lw=[]
avg=np.full(natom,0.)
for i in tqdm(range(0,npro)):
    lw.append([])
    trj=nm*np.loadtxt('../pro{}/trajectory.in'.format(i))
    trj=((trj[:,0]+1j*trj[:,1]).reshape(stp*maxtime,natom))[0:(stp*maxtime):stp,:]
    th=np.sum((trj.conj())*rr2,axis=1); ath=abs(th)**(-1)
    lw[i].append(ath)
    trj=np.matmul((th*ath).reshape(maxtime,1),np.full((1,natom),1.))*trj
    avg=avg+np.sum(trj*np.matmul(ath.reshape(maxtime,1),np.full((1,natom),1.)),axis=0)
    l.append((trj.reshape(maxtime,natom)[0:maxtime,:]))

lw=np.asarray(lw); sumw=np.sum(lw)
avg=avg/sumw

np.savetxt('avgpos.in',np.append((avg.real).reshape(natom,1),(avg.imag).reshape(natom,1),axis=1)/nm)
avg=np.matmul(np.full((maxtime,1),1.),avg.reshape(1,natom))
covr=np.full((2*natom,2*natom),0.)

for i in range(0,npro):
    l[i]=l[i]-avg
    l[i]=np.append(l[i].real,l[i].imag,axis=1)
    wtmp=np.matmul(lw[i,:].reshape(maxtime,1),np.full((1,2*natom),1.))
    covr=covr+np.matmul(l[i].transpose(),(wtmp*l[i]))

cov=np.linalg.eig(covr/sumw)
ind=np.argsort(cov[0])
np.savetxt('stdev.out',cov[0][ind])
ind1=[]
for i in range(0,natom):
    ind1.append(i)
    ind1.append(i+natom)
np.savetxt('direc.out',cov[1][ind1,:][:,ind])
