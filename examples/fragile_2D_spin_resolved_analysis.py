# Example of analyzing spin-resolved topology using the module "spin_resolved_analysis" for a 2D model of fragile topology
# The model can be found in Eqs. (D15)-(D16) of arXiv:1908.00016v2, or Eqs. (C55) and (C60) of arXiv:2207.10099v1.
# The parameter choices are in the caption of Fig. 7 in arXiv:2207.10099v1.
# The fragile bands are coupled to a set of trivial bands to gap out its Wilson loop spectrum (but not the spin-resolved Wilson loop spectrum, as we will see in this example code).
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The calculations in this python file include
# 1. 2D bulk spin-sz band structure calculation, We will consider the P_2 sz P_2 and P_6 sz P_6 spectrum, where
#    P_2 and P_6 are the occupied band projectors indicated in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1
# 2. ky-directed spin-sz-resolved Wilson loop spectrum for P_2 and P_6 mentioned in 1.
#    The results can be compared with Figs. 8 (c), (d), and Figs. 10 (c), (d) in the Supplementary Appendices of arXiv:2207.10099v1.

# Notice that the analyses of spin-resolved topology in this example code are all based on spin sz.

# In this example code we use interchangably [kx,ky] and [k1,k2].

## Generically, user should check that the grid number choice in 2. leads to convergent results for
## the spin-resolved Wilson loop, by gradually increasing the grid number. 

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib_v4 import * # import the nested Wilson loop library
from spin_resolved_analysis import * # in order to facilitate the analyses of spin-resolved topology
import timeit
import os

path_data="./fragile_2D_spin_resolved_analysis"
if not os.path.exists(path_data):
    os.mkdir(path_data)

########### Construct the model ###########

# Pauli matrices
sx = np.array([0,1,0,0],dtype=complex)
sy = np.array([0,0,1,0],dtype=complex)
sz = np.array([0,0,0,1],dtype=complex)
s0 = np.array([1,0,0,0],dtype=complex)

# parameters
pi = np.pi
I = +1j
t1 = 5.0
t2 = 1.5
vm = -1.5
tPH = 0.1
vs = 1.3
vMz = 0.4

# paremeters when we add additional orbitals
vmu = 8.25*vm
vC = 4.0
vCS = 0.45*vC

# lattice
lat=[[1,0],[0,1]]
x1 = [0,0]
x2 = [0,0]
x2cx = [0.5,0]
x2cy = [0,0.5]
orbs=[x1,x2,x2cx,x2cy]

def fragile_2D_model():
    
    model = tb_model(2,2,lat,orbs,nspin=2)
    
    # t1
    model.set_hop(t1/2,0,0,[1,0],mode="add") 
    model.set_hop(-t1/2,1,1,[1,0],mode="add") 
    model.set_hop(t1/2,0,0,[0,1],mode="add") 
    model.set_hop(-t1/2,1,1,[0,1],mode="add") 
    
    # t2
    model.set_hop(t2/2,0,1,[1,0],mode="add") 
    model.set_hop(t2/2,1,0,[1,0],mode="add") 
    model.set_hop(-t2/2,0,1,[0,1],mode="add") 
    model.set_hop(-t2/2,1,0,[0,1],mode="add") 
    
    # vm 
    model.set_onsite([vm,-vm,vmu,vmu],mode="add") 
    
    # tPH
    model.set_hop(tPH/2,0,0,[1,0],mode="add") 
    model.set_hop(tPH/2,1,1,[1,0],mode="add") 
    model.set_hop(tPH/2,0,0,[0,1],mode="add") 
    model.set_hop(tPH/2,1,1,[0,1],mode="add") 
    
    # vs
    model.set_hop(-I*vs/((2*I)**2)*sz,0,1,[1,1],mode="add") 
    model.set_hop(I*vs/((2*I)**2)*sz,1,0,[1,1],mode="add") 
    model.set_hop(I*vs/((2*I)**2)*sz,0,1,[1,-1],mode="add") 
    model.set_hop(-I*vs/((2*I)**2)*sz,1,0,[1,-1],mode="add") 
    
    # vMz
    model.set_hop(sy*vMz/(2*I),0,0,[1,0],mode="add") 
    model.set_hop(-sy*vMz/(2*I),1,1,[1,0],mode="add") 
    model.set_hop(-sx*vMz/(2*I),0,0,[0,1],mode="add") 
    model.set_hop(sx*vMz/(2*I),1,1,[0,1],mode="add") 
    
    # vC 
    model.set_hop(vC/2,1,2,[0,0],mode="add") 
    model.set_hop(vC/2,1,2,[-1,0],mode="add") 
    
    model.set_hop(vC/2,1,3,[0,0],mode="add") 
    model.set_hop(vC/2,1,3,[0,-1],mode="add") 
    
    # vCS 
    model.set_hop(0.5*sy*I*vCS,1,2,[0,0], mode="add") 
    model.set_hop(-0.5*sy*I*vCS,1,2,[-1,0], mode="add") 
    
    model.set_hop(-0.5*sx*I*vCS,1,3,[0,0], mode="add") 
    model.set_hop(+0.5*sx*I*vCS,1,3,[0,-1], mode="add") 
    
    return model

print("Construct the model\n")
my_model=fragile_2D_model()

########### 2D bulk spin-sz bands ###########

print("Compute the 2D bulk spin-sz bands using get_PsP_evals from spin_resolved_analysis\n")

# high symmetry k point
Gamma = [0.0,0.0]
X = [0.5,0.0]
Y = [0.0,0.5]
M = [0.5,0.5]

# get the path
path = [Gamma,X,M,Gamma]
label = (r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$')
numkpts = 1001
(k_vec,k_dist,k_node) = my_model.k_path(path,numkpts,report=False)

print("First consider P_2 sz P_2 where P_2 is indicated in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1")

spin_sz_evals=[]
start_time = timeit.default_timer()
for k in k_vec:
    spin_sz_evals.append(get_PsP_evals(model=my_model,spin_dir=[0,0,1],occ=[4,5],k=k))
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

spin_sz_evals=np.array(spin_sz_evals)
spin_sz_evals=np.transpose(spin_sz_evals)

plt.figure()
for n in range(spin_sz_evals.shape[0]):
    plt.plot(k_dist,spin_sz_evals[n])
plt.xlim([np.min(k_dist),np.max(k_dist)])
plt.xlabel(r"$\mathbf{k}$",fontsize=20)
plt.ylabel(r"$P_2 s_z P_2$ eigenvalue",fontsize=20)
plt.xticks(k_node,label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_2_bulk_spin_sz_bands.pdf")
plt.close()

print("Next consider P_6 sz P_6 where P_6 is indicated in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1")

spin_sz_evals=[]
start_time = timeit.default_timer()
for k in k_vec:
    spin_sz_evals.append(get_PsP_evals(model=my_model,spin_dir=[0,0,1],occ=[0,1,2,3,4,5],k=k))
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

spin_sz_evals=np.array(spin_sz_evals)
spin_sz_evals=np.transpose(spin_sz_evals)

plt.figure()
for n in range(spin_sz_evals.shape[0]):
    plt.plot(k_dist,spin_sz_evals[n])
plt.xlim([np.min(k_dist),np.max(k_dist)])
plt.xlabel(r"$\mathbf{k}$",fontsize=20)
plt.ylabel(r"$P_6 s_z P_6$ eigenvalue",fontsize=20)
plt.xticks(k_node,label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_6_bulk_spin_sz_bands.pdf")
plt.close()

########### ky-directed spin-sz-resolved Wilson Loop calculation ###########

print("Now start ky-directed spin-sz-resolved Wilson loop calculation\n")

print("First consider P_2 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1")

print("Collect the eigenvectors of the projected spin operator using get_PsP_evecs from spin_resolved_analysis")

start_time = timeit.default_timer()
N1=501
N2=51
spin_dir=[0.0,0.0,1.0]
occ=[4,5]
occ_ind=len(occ)
Nt=my_model._nsta
k1_list=np.linspace(-0.5,0.5,num=N1,endpoint=True)
k2_list=np.linspace(-0.5,0.5,num=N2,endpoint=True)
PsP_evecs=np.zeros((N1,N2,Nt,Nt//2,2),dtype=complex)
for k1_ind in range(len(k1_list)):
    for k2_ind in range(len(k2_list)):
        k1=k1_list[k1_ind]
        k2=k2_list[k2_ind]
        PsP_evecs[k1_ind,k2_ind]=get_PsP_evecs(my_model,spin_dir,occ,[k1,k2])
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

### Initiate the pythtb wave function array
PsP_wfs=wf_array(my_model,[N1,N2])

### Fill the eigenstates of the projected spin operator to the pythtb wave function array
for k1_ind in range(N1):
    for k2_ind in range(N2):
        PsP_wfs[k1_ind,k2_ind]=PsP_evecs[k1_ind,k2_ind]
PsP_wfs.impose_pbc(0,0)
PsP_wfs.impose_pbc(1,1)

### Compute the spin-resolved wannier band basis and the spin-resolved wannier band energy
### Note the band indices as list(np.arange(occ_ind//2)) and list(np.arange(Nt-occ_ind//2,Nt))
### for the P_minus and P_plus Wannier band basis array
### The general idea is that the user can fill in any types of wave functions into the wave function array,
### but the users need to be careful about selecting the band indices when they start to initiate the wannier_band_basis_array

print("Now compute the ky-directed spin-resolved Wilson loop eigenphase spectrum")

start_time = timeit.default_timer()
dir1=1 # we have two directions, kx and ky, to choose, and here dir1=1 means the second direction, namely ky
minus_wbbarr=wannier_band_basis_array(PsP_wfs,list(np.arange(occ_ind//2)),dir1) 
plus_wbbarr =wannier_band_basis_array(PsP_wfs,list(np.arange(Nt-occ_ind//2,Nt)),dir1) 

minus_wbbarr.solve_wannier_band_basis()
plus_wbbarr.solve_wannier_band_basis()
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

# get the P_minus and P_plus Wannier band energy
minus_wbe=minus_wbbarr._wannier_band_energy
plus_wbe=plus_wbbarr._wannier_band_energy

### Plot out the results

plot_grid=minus_wbe.shape[0]

plt.figure(figsize=(5.2,5))
for n in range(minus_wbe.shape[1]):
    plt.plot(minus_wbe[:,n],'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $\gamma_1^-$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_2^-$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_2_minus_spin_resolved_Wilson_loop_eigenphases.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(minus_wbe,axis=1)+np.pi,2.0*np.pi)-np.pi,'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_1^-]))$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_2^-$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_2_minus_spin_resolved_Wilson_loop_summed_eigenphases.pdf")
plt.close()

plot_grid=plus_wbe.shape[0]

plt.figure(figsize=(5.2,5))
for n in range(plus_wbe.shape[1]):
    plt.plot(plus_wbe[:,n],'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $\gamma_1^+$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_2^+$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_2_plus_spin_resolved_Wilson_loop_eigenphases.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(plus_wbe,axis=1)+np.pi,2.0*np.pi)-np.pi,'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_1^+]))$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_2^+$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_2_plus_spin_resolved_Wilson_loop_summed_eigenphases.pdf")
plt.close()

print("Next consider P_6 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1")

print("Collect the eigenvectors of the projected spin operator using get_PsP_evecs from spin_resolved_analysis")

start_time = timeit.default_timer()
N1=501
N2=51
spin_dir=[0.0,0.0,1.0]
occ=[0,1,2,3,4,5]
occ_ind=len(occ)
Nt=my_model._nsta
k1_list=np.linspace(-0.5,0.5,num=N1,endpoint=True)
k2_list=np.linspace(-0.5,0.5,num=N2,endpoint=True)
PsP_evecs=np.zeros((N1,N2,Nt,Nt//2,2),dtype=complex)
for k1_ind in range(len(k1_list)):
    for k2_ind in range(len(k2_list)):
        k1=k1_list[k1_ind]
        k2=k2_list[k2_ind]
        PsP_evecs[k1_ind,k2_ind]=get_PsP_evecs(my_model,spin_dir,occ,[k1,k2])
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

### Initiate the pythtb wave function array
PsP_wfs=wf_array(my_model,[N1,N2])

### Fill the eigenstates of the projected spin operator to the pythtb wave function array
for k1_ind in range(N1):
    for k2_ind in range(N2):
        PsP_wfs[k1_ind,k2_ind]=PsP_evecs[k1_ind,k2_ind]
PsP_wfs.impose_pbc(0,0)
PsP_wfs.impose_pbc(1,1)

### Compute the spin-resolved wannier band basis and the spin-resolved wannier band energy
### Note the band indices as list(np.arange(occ_ind//2)) and list(np.arange(Nt-occ_ind//2,Nt))
### for the P_minus and P_plus Wannier band basis array
### The general idea is that the user can fill in any types of wave functions into the wave function array,
### but the users need to be careful about selecting the band indices when they start to initiate the wannier_band_basis_array

print("Now compute the ky-directed spin-resolved Wilson loop eigenphase spectrum")

start_time = timeit.default_timer()
dir1=1 # we have two directions, kx and ky, to choose, and here dir1=1 means the second direction, namely ky
minus_wbbarr=wannier_band_basis_array(PsP_wfs,list(np.arange(occ_ind//2)),dir1) 
plus_wbbarr =wannier_band_basis_array(PsP_wfs,list(np.arange(Nt-occ_ind//2,Nt)),dir1) 

minus_wbbarr.solve_wannier_band_basis()
plus_wbbarr.solve_wannier_band_basis()
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

# get the P_minus and P_plus Wannier band energy
minus_wbe=minus_wbbarr._wannier_band_energy
plus_wbe=plus_wbbarr._wannier_band_energy

### Plot out the results

plot_grid=minus_wbe.shape[0]

plt.figure(figsize=(5.2,5))
for n in range(minus_wbe.shape[1]):
    plt.plot(minus_wbe[:,n],'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $\gamma_1^-$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_6^-$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_6_minus_spin_resolved_Wilson_loop_eigenphases.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(minus_wbe,axis=1)+np.pi,2.0*np.pi)-np.pi,'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_1^-]))$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_6^-$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_6_minus_spin_resolved_Wilson_loop_summed_eigenphases.pdf")
plt.close()

plot_grid=plus_wbe.shape[0]

plt.figure(figsize=(5.2,5))
for n in range(plus_wbe.shape[1]):
    plt.plot(plus_wbe[:,n],'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $\gamma_1^+$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_6^+$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_6_plus_spin_resolved_Wilson_loop_eigenphases.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(plus_wbe,axis=1)+np.pi,2.0*np.pi)-np.pi,'.')
plt.xlim([0,plot_grid-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_1^+]))$",fontsize=20)
plt.xticks([0,(plot_grid-1)/2.0,plot_grid-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_6^+$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_6_plus_spin_resolved_Wilson_loop_summed_eigenphases.pdf")
plt.close()

# end of the code
print("Done")

