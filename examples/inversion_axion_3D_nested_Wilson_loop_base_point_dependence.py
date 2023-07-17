# Demonstration of the base point (in momentum space) dependence of the nested Wilson loop sepctrum using a model of 3D inversion-symmetric axion insulator
# The model can be found in Appendices A 1 and 2 of arXiv:1810.02373v1.
# The code is run using python3
# The code is run using pythtb version 1.7.2

# In arXiv:1810.02373v1, the nested Wilson loop spectrum is obtained by computing a two-step Wilson loop depicted in its Fig. 4,
# namely, we do a two-step Wilson loop calculation first along kx and then along ky.

# In Figs. 3 (i)-(k) of arXiv:1810.02373v1, the nested Wilson loop spectrum is then plotted as a function of kz.
# In principle, the resulting nested Wilson loop spectrum will also depend on kx. And this is deduced from the
# fact that the last step of nested Wilson loop is taken along ky, rendering the resulting nested Wilson loop spectrum
# independent of ky -- namely dependent of kx and kz. 
# Same logic applies in the momentum dependence of the nested Wilson loop spectrum in Eq. (6.13) of PRB 96, 245115 (2017).

# Going back to our 3D inversion-symmetric axion insulator model, if we adopt the procedure of nested Wilson loop calculation in
# Fig. 4 of arXiv:1810.02373v1, the nested Wilson loop eigenphases then depend on both kx and kz.
# In particular, their dependence on kx is through the choice of the base point along kx, when we do the first step of the two-step nested Wilson loop calculation.
# To practically see this, we compute and plot the nested Wilson loop spectrum as a function of kz, following the procedure in Fig. 4 of arXiv:1810.02373v1,
# and then we repeat the same computation, but use a different base point along kx when do the first step of the two-step nested Wilson loop.

# The occupied electronic bands in this calculation correspond to P_3 in Fig. 10 (a) of arXiv:1810.02373v1.

# Specifically, in this example code, we will demonstrate that for different base point choices along kx in the first step of the nested Wilson loop calculation,
# the resulting nested Wilson loop spectra all have winding number -1 (or +1) when kz advances 2pi. 
# And in particular, such a net winding number is independent of what values kx take. 
# Such a result that the nested Wilson loop spectrum can only have non-zero winding number along at most one momentum direction has been proved in Appendix D 3 b of arXiv:2207.10099v1.

# We will use interchangeably [kx,ky,kz] and [k1,k2,k3].

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib_v4 import *
import timeit
import os
import pickle

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

print("Construct the model\n")
my_model=axionmodel3D()

# define a function to extract out the winding number from an 1D array of nested Wilson loop summed eigenphase
def get_wind(arr):
    accu=0.0
    for i in range(len(arr)-1):
        accu=accu+(np.mod(arr[i+1]-arr[i]+np.pi,2.0*np.pi)-np.pi) # mod the increment of the adjacent summed eigenphases by 2pi
    return accu/(2.0*np.pi)

# grid for nested Wilson loop
k1_grid = 31 # this is used in the first  step of the two-step nested Wilson loop calculation
k2_grid = 31 # this is used in the second step of the two-step nested Wilson loop calculation
k3_grid = 51 # this is the grid along which we will plot the nested Wilson loop eigenphases
k2_base=0.0 
k3_base=0.0
# we do not specify the k1_base here since we will be going over a for loop below for different values of k1_base.
energy_band_ind=[1,2,3] # this corresponds to P_3 in Fig. 3 (a) or Fig. 10 (a) of arXiv:1810.02373v1

# the window_1 and window_2 are determined by a pre-calculation that involves just computing the 
# kx-directed Wannier bands. One can do this pre-calculation by choosing just one value of k1_base,
# (without going over the for loop "for k1_base in np.linspace(0.0,1.0,num=N1,endpoint=True)" below)
# and then use this example code up to the line with "array2.solve_wannier_band_basis()", in which one 
# can extract out the wannier band energy by using "array2._wannier_band_energy" after the calculation 
# of "array2.solve_wannier_band_basis()" is done. And then one can determine what values of the window_1
# and window_2 should be chosen such that one can separate the Wannier band energies into an inner part,
# and an outer part, just as Fig. 3 (f) of arXiv:1810.02373v1.
window_1=-0.065*np.pi 
window_2=0.065*np.pi

N1=11 # number of different kx values
print("The k grid used for nested Wilson loop calculation is {}*{}*{}\n".format(k1_grid,k2_grid,k3_grid))

# create a directory to save the figures and data
path_data="./inversion_axion_3D_nested_Wilson_loop_base_point_dependence"
# comment out the above line and use instead the line below for path_data if the users would like to save the relevant information about this calculation in the created directory
# path_data="./inversion_axion_3D_nested_Wilson_loop_base_point_dependence_grid_{}x{}x{}_energy_band_ind_{}_window_1_{}pi_window_2_{}pi_N1_{}".format(k1_grid,k2_grid,k3_grid,energy_band_ind,round(window_1/np.pi,6),round(window_2/np.pi,6),N1)
if not os.path.exists(path_data):
    os.mkdir(path_data)

# empty list to store the inner and outer nested Wilson loop eigenphases for different k1_base
inner_nested_data=[]
outer_nested_data=[]

# empty list to store the winding numbers of the inner and outer nested Wilson loop for different k1_base
inner_winding_num_data=[]
outer_winding_num_data=[]

print("Compute nested Wilson loop for different base point along kx from 0.0 to 1.0 with grid number = {}".format(N1))
# note that the N1 here is different from the k1_grid.
# N1 is the number of values we will be using as the kx base point.
# k1_grid is the grid number we will be using to perform the first step of the two-step nested Wilson loop, going from the "kx base point" to "kx base point + 2pi".

start_time = timeit.default_timer()

for k1_base in np.linspace(0.0,1.0,num=N1,endpoint=True):

    dir1=0 # we have three directions, kx, ky and kz, to choose, and here dir1=0 means the first direction, namely kx
    array1=wf_array(my_model,[k1_grid,k2_grid,k3_grid]) #mesh is [kxres,kyres,kzres]
    array1.solve_on_grid([k1_base,k2_base,k3_base]) #solve wavefunction
    array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
    array2.solve_wannier_band_basis()

    dir2=0 # we have two remaining directions, ky and kz, to choose, and here dir2=0 means the first remaining direction, which is ky
    window_list=[window_1,window_2]
    in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True)#,wnum=1) # can remove the part wnum=1 without affecting the results
    window_list=[window_2,window_1]
    out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True)#,wnum=2) # can remove the part wnum=2 without affecting the results

    # store the inner and outer Wilson loop eigenphases into empty lists
    inner_nested_data.append(in_data)
    outer_nested_data.append(out_data)

    # evaluate and store the winding numbers of the inner and outer nested Wilson loop into empty lists
    # the combination of mod and sum is to turn the array obtained from using "nested_berry_phase" into 
    # a 1D array whose elements are the nested Wilson loop eigenphases summed over all nested Wannier bands.
    inner_winding_num_data.append(get_wind(np.mod(np.sum(in_data,axis=1)+np.pi,2.0*np.pi)-np.pi))
    outer_winding_num_data.append(get_wind(np.mod(np.sum(out_data,axis=1)+np.pi,2.0*np.pi)-np.pi))

    print("calculation for k1_base = {} finished".format(k1_base))

print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

# plot the results of the winding numbers for the inner and outer nested Wilson loop
plt.figure()
plt.plot(inner_winding_num_data,'o',color='C0')
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"inner nested Wilson loop winding number",fontsize=15)
plt.xticks([0.0,(len(inner_winding_num_data)-1)/2.0,len(inner_winding_num_data)-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-2.0,-1.0,0.0,1.0,2.0],[r"$-2$",r"$-1$",r"$0$",r"$1$",r"$2$"],fontsize=20)
plt.title(r"winding number as a function of base point $k_x$",fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_3_inner_nested_Wilson_loop_winding_number_as_a_fn_of_kx.pdf")
plt.close()

plt.figure()
plt.plot(outer_winding_num_data,'o',color='C1')
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"outer nested Wilson loop winding number",fontsize=15)
plt.xticks([0.0,(len(outer_winding_num_data)-1)/2.0,len(outer_winding_num_data)-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-2.0,-1.0,0.0,1.0,2.0],[r"$-2$",r"$-1$",r"$0$",r"$1$",r"$2$"],fontsize=20)
plt.title(r"winding number as a function of base point $k_x$",fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/P_3_outer_nested_Wilson_loop_winding_number_as_a_fn_of_kx.pdf")
plt.close()

# save the data for the winding numbers of the inner and outer nested Wilson loop
with open(path_data+"/inner_winding_num_data.pk", 'wb') as f:
    pickle.dump(inner_winding_num_data, f, pickle.HIGHEST_PROTOCOL)

with open(path_data+"/outer_winding_num_data.pk", 'wb') as f:
    pickle.dump(outer_winding_num_data, f, pickle.HIGHEST_PROTOCOL)

# plot the results of the inner and outer nested Wilson loop eigenphase
path_inner_eigenphase_figs=path_data+"/inner_eigenphase_figs"
if not os.path.exists(path_inner_eigenphase_figs):
    os.mkdir(path_inner_eigenphase_figs)

path_outer_eigenphase_figs=path_data+"/outer_eigenphase_figs"
if not os.path.exists(path_outer_eigenphase_figs):
    os.mkdir(path_outer_eigenphase_figs)

for i in range(N1):

    # get the kx value
    kx=(0.0+i/(N1-1))*(2.0*np.pi)

    # get the corresponding inner and outer nested Wilson loop eigenphase data
    in_data=inner_nested_data[i]
    out_data=outer_nested_data[i]

    plt.figure(figsize=(5.2,5))
    for n in range(in_data.shape[1]):
        plt.plot(in_data[:,n],'.',color='C0')
    plt.xlim([0,in_data.shape[0]-1])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel(r"$k_z$",fontsize=20)
    plt.ylabel(r"$\gamma_2 (k_x={}\pi,k_z)$".format(round(kx/np.pi,8)),fontsize=20)
    plt.xticks([0,(in_data.shape[0]-1)/2.0,in_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
    plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
    plt.title(r"$P_{in}$ of $P_3$",fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_inner_eigenphase_figs+"/P_3_inner_nested_Wilson_loop_eigenphase_kx_ind_{}_kx_{}pi.pdf".format(i,round(kx/np.pi,8)))
    plt.close()

    plt.figure(figsize=(5.2,5))
    plt.plot(np.mod(np.sum(in_data,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='C0')
    plt.xlim([0,in_data.shape[0]-1])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel(r"$k_z$",fontsize=20)
    plt.ylabel(r"$ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_2]))$"+r"$(k_x={}\pi,k_z)$".format(round(kx/np.pi,8)),fontsize=17.5)
    plt.xticks([0,(in_data.shape[0]-1)/2.0,in_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
    plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
    plt.title(r"$P_{in}$ of $P_3$",fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_inner_eigenphase_figs+"/P_3_inner_nested_Wilson_loop_summed_eigenphase_kx_ind_{}_kx_{}pi.pdf".format(i,round(kx/np.pi,8)))
    plt.close()

    plt.figure(figsize=(5.2,5))
    for n in range(out_data.shape[1]):
        plt.plot(out_data[:,n],'.',color='C1')
    plt.xlim([0,out_data.shape[0]-1])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel(r"$k_z$",fontsize=20)
    plt.ylabel(r"$\gamma_2 (k_x={}\pi,k_z)$".format(round(kx/np.pi,8)),fontsize=20)
    plt.xticks([0,(out_data.shape[0]-1)/2.0,out_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
    plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
    plt.title(r"$P_{out}$ of $P_3$",fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_outer_eigenphase_figs+"/P_3_outer_nested_Wilson_loop_eigenphase_kx_ind_{}_kx_{}pi.pdf".format(i,round(kx/np.pi,8)))
    plt.close()

    plt.figure(figsize=(5.2,5))
    plt.plot(np.mod(np.sum(out_data,axis=1)+np.pi,2.0*np.pi)-np.pi,'.',color='C1')
    plt.xlim([0,out_data.shape[0]-1])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel(r"$k_z$",fontsize=20)
    plt.ylabel(r"$ \mathrm{Im}(\mathrm{ln}(\mathrm{det}[\mathcal{W}_2]))$"+r"$(k_x={}\pi,k_z)$".format(round(kx/np.pi,8)),fontsize=17.5)
    plt.xticks([0,(out_data.shape[0]-1)/2.0,out_data.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
    plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
    plt.title(r"$P_{out}$ of $P_3$",fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_outer_eigenphase_figs+"/P_3_outer_nested_Wilson_loop_summed_eigenphase_kx_ind_{}_kx_{}pi.pdf".format(i,round(kx/np.pi,8)))
    plt.close()

# save also the entire data for inner and outer nested Wilson loop individual eigenphase
with open(path_data+"/inner_nested_data.pk", 'wb') as f:
    pickle.dump(inner_nested_data, f, pickle.HIGHEST_PROTOCOL)

with open(path_data+"/outer_nested_data.pk", 'wb') as f:
    pickle.dump(outer_nested_data, f, pickle.HIGHEST_PROTOCOL)

# end of the code
print("Done.")

