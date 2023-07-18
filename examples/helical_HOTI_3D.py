# Example of nested Wilson loop calculation using a model of 3D helical higher-order topological insulator with inversion and time-reversal symmetries
# The model can be found in Appendix D 5 of arXiv:2207.10099v1.
# The code is run using python3
# The code is run using pythtb version 1.7.2

# Note that in Appendix D 5 of arXiv:2207.10099v1 the coupling between the eight-band helical HOTI model and the eight
# trivial bands are chosen to be td = 1.0, while in this example code td is chosen to be 0.5 to avoid surface energy gap closing

# The calculations in this python file include
# 1. 3D bulk band structure calculation
# 2. 2D slab (finite along z) band structure calculation 
# 3. 2D slab (finite along z) Wilson loop
# 4. 1D rod band structure calculation for a rod finite along y and z
# 5. kz-directed Wilson loop at kx = 0, 0.5pi, pi planes for P_4 and P_6 in Fig. 24 (a) of arXiv:2207.10099v1
# 6. demonstration of a choice of Wannier windows to divide the Wannier band energy into inner and outer parts as in Fig. 25 of arXiv:2207.10099v1
# 7. nested Wilson loop as a function of kx for P_in and P_out in Fig. 25 of arXiv:2207.10099v1

# Notice that in this example code, the nested Wilson loop eigenphases are obtained by doing Wilson loop first along kz and then along ky.
# Such nested Wilson loop eigenphases depend on both kx and kz. 
# And if the base point of kz is kz0 in the calculation of the first Wilson loop, then the nested Wilson loop eigenphases thus obtained depend on [kx,kz0].
# In this example code we have chosen kz0 = -0.5 in the reduced coordinate.

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib import *
import timeit
import os

path_data="./helical_HOTI_3D"
if not os.path.exists(path_data):
    os.mkdir(path_data)

########### Construct the model ###########

# Pauli matrices
sx = np.array([0,1,0,0],dtype=complex)
sy = np.array([0,0,1,0],dtype=complex)
sz = np.array([0,0,0,1],dtype=complex)
s0 = np.array([1,0,0,0],dtype=complex)

# parameters and the function to generate helical-HOTI model
cutoff = 10**(-10)
pi = np.pi
I = +1j

m1 = -3.0 
vx = 1.0 
vz = 1.0 
ux = 1.0 
uz = 1.0 
m2 = 0.3 
m3 = 0.2 
vy = 2.0 
mv1 = -0.4 
mv2 = 0.2 
vH = 1.2 
A_spin_mix = 0.5 # this is for the spin mixing term

f323 = 0.25 # this is for the term f323*tau_z mu_y sigma_z

# additional hopping parameters
t_a = 11.0
t_b = -0.5
t_c = 1.5
t_d = 0.5 # can tune t_d smaller, like 0.5, to avoid potential surface energy gap closing
# can tune t_d to 1.0 to facilitate the gapping of bulk Wilson loop spectrum

# lattice
u3x = 0.35
u3y = 0.15
u3z = 0.31
lat=[[1,0,0],[0,1,0],[0,0,1]]
x1 = [0,0,0]
x2 = [0,0,0]
x3 = [0,0,0]
x4 = [0,0,0]
x5 = [u3x,u3y,u3z]
x6 = [u3x,u3y,u3z]
x7 = [-u3x,-u3y,-u3z]
x8 = [-u3x,-u3y,-u3z]
orbs = [x1,x2,x3,x4,x5,x6,x7,x8] # this will be crucial when we form the V_G later on !!!!

# function to generate a pythtb model of helical HOTI coupled to trivial bands
def helicalHOTI():
    
    model=tb_model(3,3,lat,orbs,nspin=2)
    
    # onsite for m1, m3 and mv1
    model.set_onsite([m1,m1,-m1,-m1,0.0,0.0,0.0,0.0],mode="add")
    model.set_onsite([m3,-m3,-m3,m3,0.0,0.0,0.0,0.0],mode="add")
    model.set_onsite([mv1,-mv1,mv1,-mv1,0.0,0.0,0.0,0.0],mode="add")
    
    # add hopping with non-zero lattice vector displacement
    
    # for vx
    model.set_hop(0.5*vx,0,0,[-1,0,0],mode="add")
    model.set_hop(0.5*vx,1,1,[-1,0,0],mode="add")
    model.set_hop(-0.5*vx,2,2,[-1,0,0],mode="add")
    model.set_hop(-0.5*vx,3,3,[-1,0,0],mode="add")
    
    # for vy
    model.set_hop(0.5*vy,0,0,[0,-1,0],mode="add")
    model.set_hop(0.5*vy,1,1,[0,-1,0],mode="add")
    model.set_hop(-0.5*vy,2,2,[0,-1,0],mode="add")
    model.set_hop(-0.5*vy,3,3,[0,-1,0],mode="add")
    
    # for vz
    model.set_hop(0.5*vz,0,0,[0,0,-1],mode="add")
    model.set_hop(0.5*vz,1,1,[0,0,-1],mode="add")
    model.set_hop(-0.5*vz,2,2,[0,0,-1],mode="add")
    model.set_hop(-0.5*vz,3,3,[0,0,-1],mode="add")
    
    # for ux
    model.set_hop(1*ux/(2*I),0,3,[-1,0,0],mode="add")
    model.set_hop(-1*ux/(2*I),1,2,[-1,0,0],mode="add")
    model.set_hop(-1*ux/(2*I),2,1,[-1,0,0],mode="add")
    model.set_hop(1*ux/(2*I),3,0,[-1,0,0],mode="add")
    
    # for uz
    model.set_hop(-1*uz/(2*I),0,2,[0,0,-1],mode="add")
    model.set_hop(-1*uz/(2*I),1,3,[0,0,-1],mode="add")
    model.set_hop(-1*uz/(2*I),2,0,[0,0,-1],mode="add")
    model.set_hop(-1*uz/(2*I),3,1,[0,0,-1],mode="add")
    
    # for m2, be careful of this since this is the hopping within unit cell
    model.set_hop(m2,0,1,[0,0,0],mode="add")
    model.set_hop(-m2,2,3,[0,0,0],mode="add")
    
    # for mv2, be careful of this since this is the hopping within unit cell
    model.set_hop(mv2,0,1,[0,0,0],mode="add")
    model.set_hop(mv2,2,3,[0,0,0],mode="add")
    
    # for vH, be careful of this since this hopping is spin-dependent
    model.set_hop((-vH/(2*I))*(-I)*sz,0,2,[0,-1,0],mode="add")
    model.set_hop((-vH/(2*I))*(+I)*sz,1,3,[0,-1,0],mode="add")
    model.set_hop((-vH/(2*I))*(+I)*sz,2,0,[0,-1,0],mode="add")
    model.set_hop((-vH/(2*I))*(-I)*sz,3,1,[0,-1,0],mode="add")
    
    # add the spin mixing term
    model.set_hop((-A_spin_mix/(2*I))*(-I)*sx,0,2,[0,0,-1],mode="add")
    model.set_hop((-A_spin_mix/(2*I))*(-I)*sx,1,3,[0,0,-1],mode="add")
    model.set_hop((-A_spin_mix/(2*I))*(+I)*sx,2,0,[0,0,-1],mode="add")
    model.set_hop((-A_spin_mix/(2*I))*(+I)*sx,3,1,[0,0,-1],mode="add")
    
    # f323*tau_z mu_y sigma_z
    model.set_hop(f323*(-I)*sz,0,1,[0,0,0],mode="add")
    model.set_hop(f323*(+I)*sz,2,3,[0,0,0],mode="add")
    
    ############################# now add the hopping for the additional bands #############################
    
    # first do t_a, this will induce bonding and anti-bonding and is hopping within cell, so be careful
    model.set_hop(t_a,4,6,[0,0,0],mode="add")
    model.set_hop(t_a,5,7,[0,0,0],mode="add")
    
    # next do t_b, this is onsite energy for the additional orbitals
    model.set_onsite([0.0,0.0,0.0,0.0,t_b,t_b,t_b,t_b],mode="add")
    
    # next do t_c, this will cause energy splitting between x5 & x6 and x7 & x8
    # this is also on-site energy
    model.set_onsite([0.0,0.0,0.0,0.0,t_c,-t_c,t_c,-t_c],mode="add")
    
    # next do t_d, this will couple the additional orbitals to the original model
    model.set_hop(t_d,0,4,[0,0,0],mode="add")
    model.set_hop(t_d,0,5,[0,0,0],mode="add")
    model.set_hop(t_d,0,6,[0,0,0],mode="add")
    model.set_hop(t_d,0,7,[0,0,0],mode="add")
    
    model.set_hop(t_d,1,4,[0,0,0],mode="add")
    model.set_hop(t_d,1,5,[0,0,0],mode="add")
    model.set_hop(t_d,1,6,[0,0,0],mode="add")
    model.set_hop(t_d,1,7,[0,0,0],mode="add")
    
    return model

print("Construct the model")
my_model=helicalHOTI()

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

print("Compute the 2D slab (finite along z) energy bands")

zslab=my_model.cut_piece(15,2,glue_edgs=False)

numkpts=201
GM_bar=[0,0]
X_bar=[0.5,0]
Y_bar=[0,0.5]
M_bar=[0.5,0.5]
slab_path=[GM_bar,X_bar,M_bar,Y_bar,GM_bar,M_bar,X_bar,Y_bar]
slab_label=(r'$(0,0)$',r'$(\pi,0)$',r'$(\pi,\pi)$',r'$(0,\pi)$',r'$(0,0)$',r'$(\pi,\pi)$',r'$(\pi,0)$',r'$(0,\pi)$')

(zslab_k_vec,zslab_k_dist,zslab_k_node) = zslab.k_path(slab_path,numkpts,report=False)
start_time = timeit.default_timer()
zslab_evals = zslab.solve_all(zslab_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(8,4))
for n in range(zslab_evals.shape[0]):
    plt.plot(zslab_k_dist,zslab_evals[n])
plt.xlim([np.min(zslab_k_dist),np.max(zslab_k_dist)])
plt.ylim([-2,2])
plt.xlabel(r"$\overline{\mathbf{k}}=(\overline{k}_x,\overline{k}_y)$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.xticks(zslab_k_node,slab_label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/zslab_energy_bands.pdf")
plt.close()

########### slab Wilson loop ###########

print("Compute the 2D slab (finite along z) Wilson loop with half-filling")

start_time = timeit.default_timer()
zslab_array=wf_array(zslab,[101,31])
zslab_array.solve_on_grid([-0.5,-0.5])
zslab_phs=zslab_array.berry_phase(np.arange((zslab._nsta)//2),dir=1,contin=False,berry_evals=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
for n in range(zslab_phs.shape[1]):
    plt.plot(zslab_phs[:,n],'.')
plt.xlim([0,zslab_phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(zslab_phs.shape[0]-1)/2.0,zslab_phs.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed slab $\gamma_1$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/zslab_wilson_loop_eigenphases.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(zslab_phs,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='black')
plt.xlim([0,zslab_phs.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(zslab_phs.shape[0]-1)/2.0,zslab_phs.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed slab $\mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_1]))$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/zslab_summed_wilson_loop_eigenphases.pdf")
plt.close()

########### 1D rod energy bands ###########

print("Compute the 1D rod (finite along y and z) energy bands")

start_time = timeit.default_timer()
temp=my_model.cut_piece(15,1,glue_edgs=False)
xrod=temp.cut_piece(15,2,glue_edgs=False)

numkpts=101
rod_path=[[-0.5],[0],[0.5]]
rod_label=(r'$-\pi$',r'$0$',r'$\pi$')

(xrod_k_vec,xrod_k_dist,xrod_k_node) = xrod.k_path(rod_path,numkpts,report=False)
xrod_evals = xrod.solve_all(xrod_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure()
for n in range(xrod_evals.shape[0]):
    plt.plot(xrod_k_dist,xrod_evals[n])
plt.xlim([np.min(xrod_k_dist),np.max(xrod_k_dist)])
plt.ylim([-1,1])
plt.xlabel(r"$\overline{k}_x$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.xticks(xrod_k_node,rod_label,fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/xrod_energy_bands.pdf")
plt.close()

########### Wilson loop calculation ###########

print("kz-directed Wilson loop calculation")

print("First consider P_4 in Fig. 24 (a) of arXiv:2207.10099v1")

start_time = timeit.default_timer()
my_array=wf_array(my_model,[5,101,31])
my_array.solve_on_grid([-0.5,-0.5,-0.5])
phs=my_array.berry_phase([4,5,6,7],dir=2,contin=False,berry_evals=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
k_ind=0
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_4$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P4_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=1
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_4$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P4_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=2
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_4$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P4_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=3
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_4$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P4_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=4
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_4$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P4_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

print("Next consider P_6 in Fig. 24 (a) of arXiv:2207.10099v1")

start_time = timeit.default_timer()
my_array=wf_array(my_model,[5,101,31])
my_array.solve_on_grid([-0.5,-0.5,-0.5])
phs=my_array.berry_phase([2,3,4,5,6,7],dir=2,contin=False,berry_evals=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
k_ind=0
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_6$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=1
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_6$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=2
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_6$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=3
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_6$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

plt.figure(figsize=(5.2,5))
k_ind=4
for n in range(phs.shape[2]):
    plt.plot(phs[k_ind,:,n],'.',color='black')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks([0,(phs.shape[1]-1)/2.0,phs.shape[1]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,phs.shape[1]-1])
plt.ylim([-np.pi,np.pi])
plt.title("$P_6$, $k_x = {}\pi$".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_kz_directed_Wannier_bands_kx_{}pi.pdf".format(round(2.0*(k_ind/(phs.shape[0]-1)-0.5),4)))
plt.close()

########### Nested Wilson loop calculation ###########

print("Now start nested Wilson loop calculation: first along kz and then along ky")

print("First compute the kz-directed Wannier bands of P_6 in Fig. 24 (a) of arXiv:2207.10099v1")

k1_grid = 201
k2_grid = 31
k3_grid = 31
k1_base=-0.5
k2_base=-0.5
k3_base=-0.5
dir1=2 # we have three directions, kx, ky and kz, to choose, and here dir1=2 means the third direction, namely kz
energy_band_ind=[2,3,4,5,6,7]

start_time = timeit.default_timer()
array1=wf_array(my_model,[k1_grid,k2_grid,k3_grid]) #mesh is [kxres,kyres,kzres]
array1.solve_on_grid([k1_base,k2_base,k3_base]) #solve wavefunction
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

wannier_band_energy=array2._wannier_band_energy

window_1=-0.02*np.pi
window_2=0.02*np.pi

plt.figure(figsize=(6,10.5))
plt.plot(np.sort(wannier_band_energy.flatten()),'.',color='black')
plt.axhline(y=window_2,color='C1',ls='--',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,color='C0',ls='--',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.xlabel(r"sorted Wannier band basis index",fontsize=20)
plt.ylabel(r"$k_z$-directed $\gamma_1$",fontsize=20)
plt.xticks(fontsize=12.5)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,len(np.sort(wannier_band_energy.flatten()))-1])
plt.ylim([-np.pi,np.pi])
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kz_directed_Wannier_band_energies_with_windows.pdf")
plt.close()

print("Then do nested Wilson loop along ky")

start_time = timeit.default_timer()
dir2=1 # we have two remaining directions, kx and ky, to choose, and here dir2=1 means the second remaining direction, which is ky
window_list=[window_1,window_2]
in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
window_list=[window_2,window_1]
out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=4)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,5))
for n in range(in_data.shape[1]):
    plt.plot(in_data[:,n],'.')
plt.xlim([0,in_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_x,k_z=-\pi)$",fontsize=20)
plt.xticks([0,(in_data.shape[0]-1)/2.0,in_data.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{in}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_inner_nested_Wilson_loop_eigenphase.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(in_data,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='black')
plt.xlim([0,in_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_2])) (k_x,k_z=-\pi)$",fontsize=20)
plt.xticks([0,(in_data.shape[0]-1)/2.0,in_data.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{in}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_inner_nested_Wilson_loop_summed_eigenphase.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
for n in range(out_data.shape[1]):
    plt.plot(out_data[:,n],'.')
plt.xlim([0,out_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_x,k_z=-\pi)$",fontsize=20)
plt.xticks([0,(out_data.shape[0]-1)/2.0,out_data.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{out}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_outer_nested_Wilson_loop_eigenphase.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
plt.plot(np.mod(np.sum(out_data,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='black')
plt.xlim([0,out_data.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_2])) (k_x,k_z=-\pi)$",fontsize=20)
plt.xticks([0,(out_data.shape[0]-1)/2.0,out_data.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"$P_{out}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P6_outer_nested_Wilson_loop_summed_eigenphase.pdf")
plt.close()

print("The results of nested Wilson loop correspond to Figs. 26 (a), (d), and (g) of arXiv:2207.10099v1 in which t_d is chosen as 1.0")

# end of the code
print("Done")

