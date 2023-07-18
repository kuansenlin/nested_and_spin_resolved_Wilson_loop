# Example of (spin-resolved) band topology analysis using (spin-resolved) Wilson loop for a model of 2D topological insulator (TI)
# The model can be found in Eq. (C37) of arXiv:2207.10099v1.
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The calculations in this python file include
# 1. 2D bulk band structure calculation corresponding to Fig. 2 (d) in the main text of arXiv:2207.10099v1
# 2. 2D bulk spin-sz band structure calculation corresponding to Fig. 2 (d) in the main text of arXiv:2207.10099v1 (eigenvalues of P(k)szP(k) where P(k) is the occupied band projector)
# 3. 1D ribbon band structure calculation for a ribbon finite along x demonstrating helical edge state dispersion
# 4. Wilson loop calculation demonstrating a helical winding (corresponding to Fig. 3 (b) in the Supplementary Appendices of arXiv:2207.10099v1)
# 5. Spin-resolved Wilson loop calculation demonstrating nonzero winding in the spin-resolved subspace of the occupied space
#    (corresponding to Figs. 3 (c) and (d) in the Supplementary Appendices of arXiv:2207.10099v1)
#    The spin operator used in the spin-resolved Wilson loop calculation is sz

# The spin-resolved analysis considered in this example code is based on spin "sz" operator

# (k1,k2) and (kx,ky) are used interchangeably in this example code
# The first (a1) and second (a2) lattice directions are used interchangeably as x and y, respectively, in this example code

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib import * # import the nested Wilson loop library
from spin_resolved_analysis import * # in order to facilitate the analyses of spin-resolved topology
import timeit
import os

# directory to save the data
path_data="./TI_2D_with_spin_resolved_analysis"
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
epsilon = 1.0
t1x = 0.8
t1y = 1.2
t2x = 1.3
t2y = 0.9
tR = 0.8
tPHx = 0.3
tPHy = 0.4
tMx = 0.3
tMy = 0.2

# try other inversion breaking terms
# f011, f012, f021, f022, f031, and f032 are not included in Eq. (C37) of arXiv:2207.10099v1
f011 = 0.0
f012 = 0.0
f021 = 0.0
f022 = 0.0
f031 = 0.0
f032 = 0.0
fcx10 = 1.0 # t_I in Eq. (C37) of arXiv:2207.10099v1

# lattice
lat=[[1,0],[0,1]]
x1 = [0,0]
x2 = [0,0]
orbs=[x1,x2]

def BHZ():
    
    model = tb_model(2,2,lat,orbs,nspin=2)
    
    # epsilon
    model.set_onsite([epsilon,-epsilon],mode="add")
    
    # t1x and t1y
    model.set_hop(-t1x/2,0,0,[1,0],mode="add")
    model.set_hop(t1x/2,1,1,[1,0],mode="add")
    model.set_hop(-t1y/2,0,0,[0,1],mode="add")
    model.set_hop(t1y/2,1,1,[0,1],mode="add")
    
    # t2x and t2y
    model.set_hop((t2x/(2*I))*(-I),0,1,[1,0],mode="add")
    model.set_hop((t2x/(2*I))*(+I),1,0,[1,0],mode="add")
    model.set_hop((t2y/(2*I))*sz,0,1,[0,1],mode="add")
    model.set_hop((t2y/(2*I))*sz,1,0,[0,1],mode="add")
    
    # tR
    model.set_hop((tR/(2*I))*sx,0,0,[0,1],mode="add")
    model.set_hop((tR/(2*I))*sx,1,1,[0,1],mode="add")
    model.set_hop((-tR/(2*I))*sy,0,0,[1,0],mode="add")
    model.set_hop((-tR/(2*I))*sy,1,1,[1,0],mode="add")
    
    # tPHx and tPHy
    model.set_hop(tPHx/2,0,0,[1,0],mode="add")
    model.set_hop(tPHx/2,1,1,[1,0],mode="add")
    model.set_hop(tPHy/2,0,0,[0,1],mode="add")
    model.set_hop(tPHy/2,1,1,[0,1],mode="add")
    
    # tMx and tMy
    model.set_hop((tMx/2)*(-I)*sx,0,1,[1,0],mode="add")
    model.set_hop((tMx/2)*(+I)*sx,1,0,[1,0],mode="add")
    model.set_hop((tMy/2)*(-I)*sy,0,1,[0,1],mode="add")
    model.set_hop((tMy/2)*(+I)*sy,1,0,[0,1],mode="add")
    
    # try other inversion breaking terms
    model.set_hop((f011/(2*I))*sx,0,0,[1,0],mode="add")
    model.set_hop((f011/(2*I))*sx,1,1,[1,0],mode="add")

    model.set_hop((f012/(2*I))*sx,0,0,[0,1],mode="add")
    model.set_hop((f012/(2*I))*sx,1,1,[0,1],mode="add")
    
    model.set_hop((f021/(2*I))*sy,0,0,[1,0],mode="add")
    model.set_hop((f021/(2*I))*sy,1,1,[1,0],mode="add")

    model.set_hop((f022/(2*I))*sy,0,0,[0,1],mode="add")
    model.set_hop((f022/(2*I))*sy,1,1,[0,1],mode="add")
    
    model.set_hop((f031/(2*I))*sz,0,0,[1,0],mode="add")
    model.set_hop((f031/(2*I))*sz,1,1,[1,0],mode="add")

    model.set_hop((f032/(2*I))*sz,0,0,[0,1],mode="add")
    model.set_hop((f032/(2*I))*sz,1,1,[0,1],mode="add")
    
    model.set_hop(fcx10/2,0,1,[1,0],mode="add")
    model.set_hop(fcx10/2,1,0,[1,0],mode="add")
    
    return model

print("Construct the model")
my_model=BHZ()

########### 2D bulk energy bands ###########

print("Compute the 2D bulk energy bands")

numkpts = 1001

Gamma = [0.0,0.0]
X = [0.5,0.0]
Y = [0.0,0.5]
M = [0.5,0.5]

path = [Gamma,X,M,Y,Gamma,M,X,Y]
label = (r'$\Gamma$',r'$X$',r'$M$',r'$Y$',r'$\Gamma$',r'$M$',r'$X$',r'$Y$')

(k_vec,k_dist,k_node) = my_model.k_path(path,numkpts,report=False)
start_time = timeit.default_timer()
evals = my_model.solve_all(k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure()
for n in range(evals.shape[0]):
    plt.plot(k_dist,evals[n])
plt.xlim([np.min(k_dist),np.max(k_dist)])
plt.xlabel(r"$\mathbf{k}$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.xticks(k_node,label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/bulk_energy_bands.pdf")
plt.close()

########### 2D bulk spin-sz bands ###########

print("Compute the 2D bulk spin-sz bands using get_PsP_evals from spin_resolved_analysis")

spin_sz_evals=[]
start_time = timeit.default_timer()
for k in k_vec:
    spin_sz_evals.append(get_PsP_evals(model=my_model,spin_dir=[0,0,1],occ=[0,1],k=k))
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

spin_sz_evals=np.array(spin_sz_evals)
spin_sz_evals=np.transpose(spin_sz_evals)

plt.figure()
for n in range(spin_sz_evals.shape[0]):
    plt.plot(k_dist,spin_sz_evals[n])
plt.xlim([np.min(k_dist),np.max(k_dist)])
plt.xlabel(r"$\mathbf{k}$",fontsize=20)
plt.ylabel(r"$P s_z P$ eigenvalue",fontsize=20)
plt.xticks(k_node,label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/bulk_spin_sz_bands.pdf")
plt.close()

########### 1D ribbon (finite along x) energy bands ###########

print("Compute the 1D ribbon (finite along x) energy bands")

xribbon=my_model.cut_piece(21,0,glue_edgs=False)

numkpts = 201
ribbon_path = [[-0.5],[0.0],[0.5]]
(xribbon_k_vec,xribbon_k_dist,xribbon_k_node) = xribbon.k_path(ribbon_path,numkpts,report=False)
start_time = timeit.default_timer()
xribbon_evals = xribbon.solve_all(xribbon_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure()
for n in range(len(xribbon_evals[:,0])):
    plt.plot(xribbon_k_dist,xribbon_evals[n,:])
plt.xticks(xribbon_k_node,[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$\overline{k}_y$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.title(r"ribbon finite along $x$",fontsize=20)
plt.ylim([-1,1])
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/xribbon_energy_bands.pdf")
plt.close()

########### kx-directed Wilson Loop calculation ###########

print("kx-directed Wilson loop calculation")

start_time = timeit.default_timer()
dir1=0 # we have two directions, kx and ky, to choose, and here dir1=0 means the first direction, namely kx
energy_band_ind=[0,1]
array1=wf_array(my_model,[51,201]) # form the pythtb wave function array object
array1.solve_on_grid([-0.5,-0.5]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) # form the wannier band basis array object
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
for n in range(pha.shape[1]):
    plt.plot(pha[:,n],'o',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1 (k_y)$",fontsize=20)
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_directed_Wilson_loop.pdf")
plt.close()

########### kx-directed spin-sz-resolved Wilson Loop calculation ###########

print("Now start kx-directed spin-sz-resolved Wilson loop calculation")

print("First, collect the eigenvectors of the projected spin operator using get_PsP_evecs from spin_resolved_analysis")

start_time = timeit.default_timer()

# set up the grids and energy band index
N1=51
N2=201
Nt=my_model._nsta # number of tight-binding basis functions per unit cell
k1_list=np.linspace(-0.5,0.5,num=N1,endpoint=True)
k2_list=np.linspace(-0.5,0.5,num=N2,endpoint=True)
occ_ind=[0,1]
N_occ=len(occ_ind)

# initiate the pythtb wave function array to store the PsP eigenvectors
PsP_wfs=wf_array(my_model,[N1,N2])

# collect the PsP eigenvectors
for k1_ind in range(len(k1_list)):
    for k2_ind in range(len(k2_list)):
        k1=k1_list[k1_ind]
        k2=k2_list[k2_ind]
        PsP_wfs[k1_ind,k2_ind]=get_PsP_evecs(model=my_model,spin_dir=[0,0,1],occ=occ_ind,k=[k1,k2])

print("--- %s seconds ---" % (timeit.default_timer() - start_time))

# impose periodic boundary conditions through pythtb
PsP_wfs.impose_pbc(0,0)
PsP_wfs.impose_pbc(1,1)

print("Next, computate the kx-directed spin-resolved Wilson loop using solve_wannier_band_basis from the nested Wilson loop library")

start_time = timeit.default_timer()

dir1=0 # we have two directions, kx and ky, to choose, and here dir1=0 means the first direction, namely kx
minus_wbbarr=wannier_band_basis_array(PsP_wfs,list(np.arange(N_occ//2)),dir1) 
plus_wbbarr =wannier_band_basis_array(PsP_wfs,list(np.arange(Nt-N_occ//2,Nt)),dir1) 

minus_wbbarr.solve_wannier_band_basis()
plus_wbbarr.solve_wannier_band_basis()

print("--- %s seconds ---" % (timeit.default_timer() - start_time))

# get the P_minus and P_plus Wannier band energy
minus_wbe=minus_wbbarr._wannier_band_energy
plus_wbe=plus_wbbarr._wannier_band_energy

plt.figure(figsize=(5.2,5))
for n in range(minus_wbe.shape[1]):
    plt.plot(minus_wbe[:,n],'o',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1^- (k_y)$, $s=s_z$",fontsize=20)
plt.xticks([0,(minus_wbe.shape[0]-1)/2.0,minus_wbe.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,minus_wbe.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_directed_spin_sz_resolved_Wilson_loop_over_lower_spin_band.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
for n in range(plus_wbe.shape[1]):
    plt.plot(plus_wbe[:,n],'o',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1^+ (k_y)$, $s=s_z$",fontsize=20)
plt.xticks([0,(plus_wbe.shape[0]-1)/2.0,plus_wbe.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,plus_wbe.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_directed_spin_sz_resolved_Wilson_loop_over_upper_spin_band.pdf")
plt.close()

# end of the code
print("Done")

