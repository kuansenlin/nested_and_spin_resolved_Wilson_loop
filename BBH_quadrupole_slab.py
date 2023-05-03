# Example of nested Wilson loop calculation for a slab cut from a 3D system
# The main focus of this example is how to properly choose the directions used in the nested Wilson loop calculation,
# as there are subtleties between the "index of the entries of the wave function array and Wannier band basis array",
# and "the index of the (reciprocal) lattice vectors".

################################################################################################################################
# The rule is to always refer to the entries of the wave function array and Wannier band basis array, even though it is a slab model 
# cut from a 3D model, when choosing the directions used in the nested Wilson loop calculation.
# There are two directions the users need to choose when doing nested Wilson loop calculation.
# The first direction is the index of the entries of the wave function array (PythTB wf_array class).
# And the second direction is the index of the entries of the Wannier band basis array (wannier_band_basis_array class)
################################################################################################################################

# The model we consider is a slab model cut from a 3D model obtained by stacking 2D Benalcazar-Bernevig-Hughes (BBH) model of a 2D quadrupole insulator
# The 2D BBH model can be found in Eqs. (6.28)--(6.29) of PRB 96, 245115 (2017)
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The model parameters of the 2D BBH model chosen in this example file lie in the middle blue region of Fig. 25 in PRB 96, 245115 (2017).
# This means that the nested Wilson loop eigenphases take value \pi for the two ways depicted in Fig. 20 of PRB 96, 245115 (2017).

# We will place the 2D BBH model in the "Cartesian xy plane", and then stack it along the Cartesian z-axis.
# To make the results comparable with the 2D BBH model, we do not turn on any hopping terms along Cartesian z-axis.
# Therefore, the model we consider consists 2D layers of BBH models that are decoupled with each other.

# In order to demonstrate the proper choice of the directions used in the nested Wilson loop calculation, 
# the lattice vectors, however, are chosen to be:
# a1 = \hat{z}, a2 = \hat{x}, a3 = \hat{y}
# where \hat{x}, \hat{y}, and \hat{z} are the unit vectors in the Cartesian coordinate.
# This means that the reciprocal lattice vectors are
# b1 = 2\pi \hat{z}, b2 = 2\pi \hat{x}, b3 = 2\pi \hat{y}.
# In the following calculations, we will use (k1,k2,k3) as the reduced coordinate in k-space,
# such that a generic k vector can be written as k = (k1*b1 + k2*b2 + k3*b3)/(2*\pi)

# On the other hand, we will also shift the positions of the tight-binding basis states uniformly, 
# in order to demonstrate that our calculation can indeed obtain the actual centers of the (hybrid) Wannier functions.
# To be specific, we will make all the tight-binding basis states locate at position 0.1*\hat{x} + 0.2*\hat{y} = 0.1*a2 + 0.2*a3.

# We will then cut this stacked 3D model such that it is finite along a1 = \hat{z} with unit cell number N1.
# Importantly, we will denote (k2,k3) as the reduced coordinate in k-space of the resulting slab model.
# Note that k2 and k3 correspond to Cartesian x and y in this example.

# In order to obtain results that can be compared with the 2D BBH model, we should perform nested Wilson loop calculations either:
# 1. First along b2 = 2\pi \hat{x} and then along b3 = 2\pi \hat{y},
# or 
# 2. First along b3 = 2\pi \hat{y} and then along b2 = 2\pi \hat{x}.

# We then deduce that if we use the first way to perform the nested Wilson loop calculation, we will obtain
# N1 \pi+0.2*(2*\pi) = -0.6*\pi mod 2*\pi eigenphases for either the inner or outer nested Wilson loop.
# We also deduce that if we use the second way to perform the nested Wilson loop calculation, we will obtain
# N1 \pi+0.1*(2*\pi) = -0.8*\pi mod 2*\pi eigenphases for either the inner or outer nested Wilson loop.
# And this example code will verify these deductions.
# Therefore, our nested Wilson loop library indeed can be applied to slab models cut from 3D systems.

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib_v4 import *
import timeit
import os

# directory to save the data
path_data="./BBH_quadrupole_slab"
if not os.path.exists(path_data):
    os.mkdir(path_data)

pi = np.pi
I = +1j

########### Construct the model ###########

# model parameters
gamma_x = 0.75
lambda_x = 1.0
gamma_y = 0.75
lambda_y = 1.0

lat = [[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]
orbs = [[0.0,0.1,0.2],[0.0,0.1,0.2],[0.0,0.1,0.2],[0.0,0.1,0.2]]
def quadrupole2Dto3Drotaterotateshift():
    
    # initiate the tb model
    model = tb_model(3,3,lat,orbs,nspin=1)
    
    # add gamma_x
    model.set_hop(gamma_x,0,2,[0,0,0],mode="add")
    model.set_hop(gamma_x,1,3,[0,0,0],mode="add")
    
    # add gamma_y
    model.set_hop(gamma_y,0,3,[0,0,0],mode="add")
    model.set_hop(-gamma_y,1,2,[0,0,0],mode="add")
    
    # add lambda_x
    model.set_hop(lambda_x,0,2,[0,1,0],mode="add")
    model.set_hop(lambda_x,3,1,[0,1,0],mode="add")
    
    # add lambda_y
    model.set_hop(lambda_y,0,3,[0,0,1],mode="add")
    model.set_hop(-lambda_y,2,1,[0,0,1],mode="add")
    
    return model

print("Construct the 3D model\n")
my_model = quadrupole2Dto3Drotaterotateshift()

# cut the model
print("Cut the 3D model into slab\n")
finite_model=my_model.cut_piece(5,0,glue_edgs=False)

########### nested Wilson loop calculation for the slab ###########

print("Nested Wilson loop calculation for the slab: first along k2 and then k3")

print("First do k2-directed Wilson loop to obtain k2-directed Wannier band basis states")

start_time = timeit.default_timer()
dir1=0 # Note that we use (k2,k3) as the reduced coordinate in the k-space for the slab, and so dir1 = 0 means the first of (k2,k3), which is k2.
energy_band_ind=list(np.arange(finite_model._nsta//2))
array1=wf_array(finite_model,[51,51]) 
array1.solve_on_grid([-0.5,-0.5]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

window_1=(-0.75*np.pi)+0.1*2.0*np.pi
window_2=(+0.0*np.pi)+0.1*2.0*np.pi

plt.figure()
for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'o')
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlabel(r"$k_3$",fontsize=20)
plt.ylabel(r"$k_2$-directed $\gamma_1$",fontsize=20)
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/k2_directed_Wannier_bands.pdf")
plt.close()

print("Then perform k3-directed nested Wilson loop")

start_time = timeit.default_timer()
dir2=0 # Since we have perform Wilson loop along k2, the resulting Wannier band basis states depend on (only one) reduced coordinate (k3), and so dir2=0 means k3.
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=finite_model._nsta//4)
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=finite_model._nsta//4)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

plt.figure()
plt.axhline(y=nested_inner[0],ls='--',color='C2',label=r"${}\pi$".format(round(nested_inner[0]/np.pi,6)))
plt.plot(nested_inner,"o",color="C0")
plt.title("Nested Wilson loop first along $k_2$ and then $k_3$",fontsize=15)
plt.xlabel("inner nested Wilson loop eigenphase index",fontsize=15)
plt.ylabel(r"$\gamma_2$",fontsize=20)
plt.xticks(np.arange(len(nested_inner)),np.arange(len(nested_inner)),fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/inner_nested_Wilson_loop_first_along_k2_and_then_k3.pdf")
plt.close()

plt.figure()
plt.axhline(y=nested_outer[0],ls='--',color='C2',label=r"${}\pi$".format(round(nested_outer[0]/np.pi,6)))
plt.plot(nested_outer,"o",color="C1")
plt.title("Nested Wilson loop first along $k_2$ and then $k_3$",fontsize=15)
plt.xlabel("outer nested Wilson loop eigenphase index",fontsize=15)
plt.ylabel(r"$\gamma_2$",fontsize=20)
plt.xticks(np.arange(len(nested_outer)),np.arange(len(nested_outer)),fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/outer_nested_Wilson_loop_first_along_k2_and_then_k3.pdf")
plt.close()

print("Nested Wilson loop calculation for the slab: first along k3 and then k2")

print("First do k3-directed Wilson loop to obtain k3-directed Wannier band basis states")

start_time = timeit.default_timer()
dir1=1 # Note that we use (k2,k3) as the reduced coordinate in the k-space for the slab, and so dir1 = 1 means the second of (k2,k3), which is k3.
energy_band_ind=list(np.arange(finite_model._nsta//2))
array1=wf_array(finite_model,[51,51]) 
array1.solve_on_grid([-0.5,-0.5]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

window_1=(-0.75*np.pi)+0.2*2.0*np.pi
window_2=(+0.0*np.pi)+0.2*2.0*np.pi

plt.figure()
for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'o')
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlabel(r"$k_2$",fontsize=20)
plt.ylabel(r"$k_3$-directed $\gamma_1$",fontsize=20)
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/k3_directed_Wannier_bands.pdf")
plt.close()

print("Then perform k2-directed nested Wilson loop")

start_time = timeit.default_timer()
dir2=0 # Since we have performed Wilson loop along k3, the resulting Wannier band basis states depend on (only one) reduced coordinate (k2), and so dir2=0 means k2.
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=finite_model._nsta//4)
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=finite_model._nsta//4)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

plt.figure()
plt.axhline(y=nested_inner[0],ls='--',color='C2',label=r"${}\pi$".format(round(nested_inner[0]/np.pi,6)))
plt.plot(nested_inner,"o",color="C0")
plt.title("Nested Wilson loop first along $k_3$ and then $k_2$",fontsize=15)
plt.xlabel("inner nested Wilson loop eigenphase index",fontsize=15)
plt.ylabel(r"$\gamma_2$",fontsize=20)
plt.xticks(np.arange(len(nested_inner)),np.arange(len(nested_inner)),fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/inner_nested_Wilson_loop_first_along_k3_and_then_k2.pdf")
plt.close()

plt.figure()
plt.axhline(y=nested_outer[0],ls='--',color='C2',label=r"${}\pi$".format(round(nested_outer[0]/np.pi,6)))
plt.plot(nested_outer,"o",color="C1")
plt.title("Nested Wilson loop first along $k_3$ and then $k_2$",fontsize=15)
plt.xlabel("outer nested Wilson loop eigenphase index",fontsize=15)
plt.ylabel(r"$\gamma_2$",fontsize=20)
plt.xticks(np.arange(len(nested_outer)),np.arange(len(nested_outer)),fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/outer_nested_Wilson_loop_first_along_k3_and_then_k2.pdf")
plt.close()

# end of the code
print("Done")

