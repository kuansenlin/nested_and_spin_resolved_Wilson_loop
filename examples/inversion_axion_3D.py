# Example of nested Wilson loop calculation using a model of 3D inversion-symmetric axion insulator
# The model can be found in Appendices A 1 and 2 of arXiv:1810.02373v1.
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The calculations in this python file include
# 1. 3D bulk band structure calculation
# 2. 2D slab (finite along x) band structure calculation 
# 3. 2D slab (finite along x) Wilson loop with half-filling
# 4. 1D rod band structure calculation for a rod finite along x and y
# 5. kx-directed Wilson loop at kz = 0, 0.5pi, pi planes for P_2 and P_3 in Fig. 10 (a) of arXiv:1810.02373v1
# 6. demonstration of a choice of Wannier windows to divide the Wannier band energies into inner and outer parts as in Fig. 10 (e) of arXiv:1810.02373v1
# 7. nested Wilson loop as a function of kz for P_in and P_out in Fig. 10 (e) of arXiv:1810.02373v1, and the results correspond to Figs. 10 (h)-(k) of arXiv:1810.02373v1

# Notice that in this example code, the nested Wilson loop eigenphases are obtained by doing Wilson loop first along kx and then along ky.
# Such nested Wilson loop eigenphases depend on both kx and kz. 
# And if the base point of kx is kx0 in the calculation of the first Wilson loop, then the nested Wilson loop eigenphases thus obtained depend on [kx0,kz].
# In this example code we have chosen kx0 = 0.

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib_v4 import *
import timeit
import os

path_data="./inversion_axion_3D"
if not os.path.exists(path_data):
    os.mkdir(path_data)

########### Construct the model ###########

I=1j
sqrt3 = np.sqrt(3.0)
sx=np.array([0,1,0,0],dtype='complex')
sy=np.array([0,0,1,0],dtype='complex')
sz=np.array([0,0,0,1],dtype='complex')


#This is the I-symmetric TB model of an AXI from arXiv:1810.02373v1

#basic BHZ terms
tx = 2.3
ty = 2.5
tz= 3
m=-5

#p-orbital-like hopping terms to break mirrors and C4 symmetry
txBCx = 0.9
txBCy = 0

#SOC terms
vz=2.4
vxy=3.2
#Break Mx and My
vxy2 = 1.5
#Break Mz
vz2 = 0.4


#This breaks Particle-Hole (PH) Symmetry
#This is tPHX=tPHY (and tPHZ=0) in the notation of arXiv:1810.02373v1
tPH = 0.3


#We will work in the limit of large SOC to have large Wilson gaps

#MAGNETIC TERMS
mA = 1.2
mB = 0.3


#Terms to trivialize the fragile Wilson winding at kz=0

#FRAGILE SPIN SPLITTING
mszF = 2.2 #this is usZ in arXiv:1810.02373v1

#USE TOPOLOGICAL QUANTUM CHEMISTRY TO MAKE S ORBITALS AT GENERAL POSTION LIKE S AND P (BY (ANTI)BONDING)
vFsp = 20 #this is t_sp in arXiv:1810.02373v1
#spin chemical potential
muF = -0.5 #t1 in arXiv:1810.02373v1


#"COUPLED PARAMETERS"
#####################################
#term to couple fragile and trivial bands
vFc1 = 4 #vC in arXiv:1810.02373v1

#"UNCOUPLED PARAMETERS"
#####################################
#term to couple bands with fragile and trivial winding 
#vFc1 = 0

ux1 = 0.35
uy1 = 0.15 
uz1 = 0.12

#Use inversion about origin (0,0,0) to set coordiantes of fourth orbital pair
ux2 = -ux1
uy2 = -uy1
uz2 = -uz1

lat=[[1,0,0],[0,1,0],[0,0,1]]
x1 = [0,0,0]
x2 = [0,0,0]

#Create s- and p- like (anti)bonding pairs about 1a in Space Group 2
x3 = [ux1,uy1,uz1]
x4 = [ux2,uy2,uz2]
orbs=[x1,x2,x3,x4]

def axionmodel3D():
    model=tb_model(3,3,lat,orbs,nspin=2)
    model.set_onsite([m,-m,0,0],mode="add")

    #Tbreaking zeeman term
    model.set_onsite([sz*mA,sz*mA,0,0],mode="add")

    #mBsxtauZ
    model.set_onsite([sz*mB,-sz*mB,0,0],mode="add")


    #FRAGILE TERMS
    #############################
    #mszF*sz in the subspace of new orbitals
    model.set_onsite([0,0,mszF*sz,mszF*sz],mode="add")

    model.set_onsite([0,0,muF,muF],mode="add")

    #Exploit that two S at the general position in SG 2
    #are equivalent to an s and a p at 1a
    model.set_hop(vFsp,2,3,[0,0,0],mode="add")

    #Couple the effective new s (bonding) orbitals at 1a to the ones already there
    model.set_hop(vFc1,0,2,[0,0,0],mode="add")
    model.set_hop(vFc1,0,3,[0,0,0],mode="add")

    ###############################################################
    #HOPPING WITHIN THE ORIGINAL TI/FRAGILE BANDS
   
    #x hopping no SOC :  tx*tauZ*(cos kx)
    model.set_hop(tx/2,0,0,[1,0,0],mode="add")
    model.set_hop(-tx/2,1,1,[1,0,0],mode="add")

    #y hopping no SOC :  ty*tauZ*(cos ky) 
    model.set_hop(ty/2,0,0,[0,1,0],mode="add")
    model.set_hop(-ty/2,1,1,[0,1,0],mode="add")

    #z hopping no SOC tz*tauZ*cos(kz)
    model.set_hop(tz*0.5,0,0,[0,0,1],mode="add")
    model.set_hop(-tz*0.5,1,1,[0,0,1],mode="add")

    #x hopping noSOC (break C4z) txBCx*tauY*sin(kx)
    model.set_hop(-txBCx/2,0,1,[1,0,0],mode="add")
    model.set_hop(txBCx/2,1,0,[1,0,0],mode="add")

    #y hopping noSOC (helps open bulk and Wilson gaps) txBCy*tauY*sin(ky)
    model.set_hop(-txBCy/2,0,1,[0,1,0],mode="add")
    model.set_hop(txBCy/2,1,0,[0,1,0],mode="add")

    #xy hopping break PH  txy*Identity*(cos kx + cos ky)
    model.set_hop(tPH/2,0,0,[1,0,0],mode="add")
    model.set_hop(tPH/2,1,1,[1,0,0],mode="add")
    model.set_hop(tPH/2,0,0,[0,1,0],mode="add")
    model.set_hop(tPH/2,1,1,[0,1,0],mode="add")

    #XY SOC  vxy*(sin(kx)sx + sin(ky)sy)
    model.set_hop(-vxy/2*I*sx,0,1,[1,0,0],mode="add")
    model.set_hop(-vxy/2*I*sx,1,0,[1,0,0],mode="add")
    model.set_hop(-vxy/2*I*sy,0,1,[0,1,0],mode="add")
    model.set_hop(-vxy/2*I*sy,1,0,[0,1,0],mode="add")

    #XY SOC that breaks mirrors  vxy*(sin(kx)sz + sin(ky)sx)
    model.set_hop(-vxy2/2*I*sz,0,1,[1,0,0],mode="add")
    model.set_hop(-vxy2/2*I*sz,1,0,[1,0,0],mode="add")
    model.set_hop(-vxy2/2*I*sz,0,1,[0,1,0],mode="add")
    model.set_hop(-vxy2/2*I*sz,1,0,[0,1,0],mode="add")

    #z SOC vz*tauXsZ * sin(kz)
    model.set_hop(-vz/2*I*sz,0,1,[0,0,1],mode="add")
    model.set_hop(-vz/2*I*sz,1,0,[0,0,1],mode="add")

    #z SOC that breaks Mz vz2*tauXsx * sin(kz)
    model.set_hop(-vz2/2*I*sx,0,1,[0,0,1],mode="add")
    model.set_hop(-vz2/2*I*sx,1,0,[0,0,1],mode="add")
    
    return model

print("Construct the model")
my_model=axionmodel3D()

########### 3D bulk energy bands ###########

print("Compute the 3D bulk energy bands")

numkpts = 201

GM = [0,0,0]
X = [0.5,0,0]
Y = [0,0.5,0]
S = [0.5,0.5,0]
Z = [0,0,0.5]
U = [0.5,0,0.5]
T = [0,0.5,0.5]
R = [0.5,0.5,0.5]

path=[GM,X,S,Y,GM,Z,U,R,T,Z,Y,T,U,X,S,R,GM]
label=(r'$\Gamma $',r'$X$',r'$S$',r'$Y$',r'$\Gamma$',r'$Z$',r'$U$',r'$R$',r'$T$',r'$Z$',r'$Y$',r'$T$',r'$U$',r'$X$',r'$S$',r'$R$',r'$\Gamma$')

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

########### 2D slab energy bands ###########

print("Compute the 2D slab (finite along x) energy bands")

xslab=my_model.cut_piece(15,0,glue_edgs=False)

numkpts=201
GM_bar=[0,0]
X_bar=[0.5,0]
Y_bar=[0,0.5]
M_bar=[0.5,0.5]
slab_path=[GM_bar,X_bar,M_bar,Y_bar,GM_bar,M_bar,X_bar,Y_bar]
slab_label=(r'$(0,0)$',r'$(\pi,0)$',r'$(\pi,\pi)$',r'$(0,\pi)$',r'$(0,0)$',r'$(\pi,\pi)$',r'$(\pi,0)$',r'$(0,\pi)$')

(xslab_k_vec,xslab_k_dist,xslab_k_node) = xslab.k_path(slab_path,numkpts,report=False)
start_time = timeit.default_timer()
xslab_evals = xslab.solve_all(xslab_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(8,4))
for n in range(xslab_evals.shape[0]):
    plt.plot(xslab_k_dist,xslab_evals[n])
plt.xlim([np.min(xslab_k_dist),np.max(xslab_k_dist)])
plt.ylim([-5,5])
plt.xlabel(r"$\overline{\mathbf{k}}=(\overline{k}_y,\overline{k}_z)$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.xticks(xslab_k_node,slab_label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/xslab_energy_bands.pdf")
plt.close()

########### slab Wilson loop ###########

print("Compute the 2D slab (finite along x) Wilson loop with half-filling")

start_time = timeit.default_timer()
xslab_array=wf_array(xslab,[401,31])
xslab_array.solve_on_grid([0,0])
xslab_phs=xslab_array.berry_phase(np.arange((xslab._nsta)//2),dir=1,contin=False,berry_evals=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
for n in range(xslab_phs.shape[1]):
    plt.plot(xslab_phs[:,n],'.',color='black')
plt.xlim([0,xslab_phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(xslab_phs.shape[0]-1)/2.0,xslab_phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed slab $\gamma_1$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/xslab_wilson_loop_eigenphases.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(xslab_phs,axis=1)+np.pi,2.0*np.pi)-np.pi,'o',color='black')
plt.xlim([0,xslab_phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(xslab_phs.shape[0]-1)/2.0,xslab_phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed slab $\mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_1]))$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/xslab_summed_wilson_loop_eigenphases.pdf")
plt.close()

########### 1D rod energy bands ###########

print("Compute the 1D rod (finite along x and y) energy bands")

start_time = timeit.default_timer()
temp=my_model.cut_piece(20,0,glue_edgs=False)
zrod=temp.cut_piece(20,1,glue_edgs=False)

numkpts=101
rod_path=[[-0.5],[0],[0.5]]
rod_label=(r'$-\pi$',r'$0$',r'$\pi$')

(zrod_k_vec,zrod_k_dist,zrod_k_node) = zrod.k_path(rod_path,numkpts,report=False)
zrod_evals = zrod.solve_all(zrod_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure()
for n in range(zrod_evals.shape[0]):
    plt.plot(zrod_k_dist,zrod_evals[n])
plt.xlim([np.min(zrod_k_dist),np.max(zrod_k_dist)])
plt.ylim([-2,2])
plt.xlabel(r"$\overline{k}_z$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.xticks(zrod_k_node,rod_label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/zrod_energy_bands.pdf")
plt.close()

########### Wilson loop calculation ###########

print("kx-directed Wilson loop calculation")

print("First consider P_2 in Fig. 10 (a) of arXiv:1810.02373v1")

start_time = timeit.default_timer()
my_array=wf_array(my_model,[31,101,5])
my_array.solve_on_grid([0,0,0])
phs=my_array.berry_phase([2,3],dir=0,contin=False,berry_evals=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
k_ind=0
for n in range(phs.shape[2]):
    plt.plot(phs[:,k_ind,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[0]-1)/2.0,phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_2$, $k_z = {}\pi$".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P2_kx_directed_Wannier_bands_kz_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=1
for n in range(phs.shape[2]):
    plt.plot(phs[:,k_ind,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[0]-1)/2.0,phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_2$, $k_z = {}\pi$".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P2_kx_directed_Wannier_bands_kz_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=2
for n in range(phs.shape[2]):
    plt.plot(phs[:,k_ind,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[0]-1)/2.0,phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_2$, $k_z = {}\pi$".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P2_kx_directed_Wannier_bands_kz_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)))
plt.close()

print("Next consider P_3 in Fig. 10 (a) of arXiv:1810.02373v1")

start_time = timeit.default_timer()
my_array=wf_array(my_model,[31,101,5])
my_array.solve_on_grid([0,0,0])
phs=my_array.berry_phase([1,2,3],dir=0,contin=False,berry_evals=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
k_ind=0
for n in range(phs.shape[2]):
    plt.plot(phs[:,k_ind,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[0]-1)/2.0,phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_3$, $k_z = {}\pi$".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_kx_directed_Wannier_bands_kz_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=1
for n in range(phs.shape[2]):
    plt.plot(phs[:,k_ind,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[0]-1)/2.0,phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_3$, $k_z = {}\pi$".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_kx_directed_Wannier_bands_kz_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=2
for n in range(phs.shape[2]):
    plt.plot(phs[:,k_ind,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[0]-1)/2.0,phs.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_3$, $k_z = {}\pi$".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_kx_directed_Wannier_bands_kz_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[1]-1)),4)))
plt.close()

########### Nested Wilson loop calculation ###########

print("Now start nested Wilson loop calculation: first along kx and then along ky")

print("First compute the kx-directed Wannier bands of P_3 in Fig. 10 (a) of arXiv:1810.02373v1")

k1_grid = 31
k2_grid = 31
k3_grid = 51
k1_base=0.0
k2_base=0.0
k3_base=0.0
dir1=0 # we have three directions, kx, ky and kz, to choose, and here dir1=0 means the first direction, namely kx
energy_band_ind=[1,2,3]

start_time = timeit.default_timer()
array1=wf_array(my_model,[k1_grid,k2_grid,k3_grid]) #mesh is [kxres,kyres,kzres]
array1.solve_on_grid([k1_base,k2_base,k3_base]) #solve wavefunction
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

wannier_band_energy=array2._wannier_band_energy

window_1=-0.065*np.pi
window_2=0.065*np.pi

plt.figure(figsize=(6,8))
plt.plot(np.sort(wannier_band_energy.flatten()),'.',color='black')
plt.axhline(y=window_2,color='C1',ls='--',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,color='C0',ls='--',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.xlabel(r"sorted Wannier band basis index",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,len(np.sort(wannier_band_energy.flatten()))-1])
plt.ylim([-np.pi,np.pi])
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_directed_Wannier_band_energies_with_windows.pdf")
plt.close()

print("Then do nested Wilson loop along ky")

start_time = timeit.default_timer()
dir2=0 # we have two remaining directions, ky and kz, to choose, and here dir2=0 means the first remaining direction, which is ky
window_list=[window_1,window_2]
in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1) # can remove the part wnum=1 without affecting the results
window_list=[window_2,window_1]
out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2) # can remove the part wnum=2 without affecting the results
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
for n in range(in_data.shape[1]):
    plt.plot(in_data[:,n],'.',color='black')
plt.xlim([0,in_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_z$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_x=0,k_z)$",fontsize=20)
plt.xticks([0,(in_data.shape[0]-1)/2.0,in_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{in}$ of $P_3$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_inner_nested_Wilson_loop_eigenphase.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(in_data,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='black')
plt.xlim([0,in_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_z$",fontsize=20)
plt.ylabel(r"$ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_2])) (k_x=0,k_z)$",fontsize=20)
plt.xticks([0,(in_data.shape[0]-1)/2.0,in_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{in}$ of $P_3$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_inner_nested_Wilson_loop_summed_eigenphase.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
for n in range(out_data.shape[1]):
    plt.plot(out_data[:,n],'.',color='black')
plt.xlim([0,out_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_z$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_x=0,k_z)$",fontsize=20)
plt.xticks([0,(out_data.shape[0]-1)/2.0,out_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{out}$ of $P_3$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_outer_nested_Wilson_loop_eigenphase.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(out_data,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='black')
plt.xlim([0,out_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_z$",fontsize=20)
plt.ylabel(r"$ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_2])) (k_x=0,k_z)$",fontsize=20)
plt.xticks([0,(out_data.shape[0]-1)/2.0,out_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{out}$ of $P_3$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P3_outer_nested_Wilson_loop_summed_eigenphase.pdf")
plt.close()

print("The results of nested Wilson loop correspond to Figs. 10 (h)-(k) of arXiv:1810.02373v1")

# end of the code
print("Done")

