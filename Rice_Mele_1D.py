# Example calculation of Wilson loop eigenphases of 1D Rice-Mele chain using the 
# wannier_band_basis_array class in the nested Wilson loop library.
# The model can be found in Eq. (127) of arXiv:2011.12426v2.
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The calculations in this python file include
# 1. 1D energy bands
# 2. 0D finite size spectrum
# 3. Waniner band basis calculation
# 4. spectral flow in Rice-Mele chain

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
from pythtb import *
from nestedWilsonLib_v4 import *
import timeit
import os

# directory to save the data
path_data="./Rice_Mele_1D"
if not os.path.exists(path_data):
    os.mkdir(path_data)

pi = np.pi
I = +1j

# model parameters
t=1.0
delta_t=-0.1
Delta=0.5
phi=1.0*pi 

print("First part of the calculation: fix t = {}, delta_t = {}, Delta = {}, phi = {}pi".format(round(t,4),round(delta_t,4),round(Delta,4),round(phi/pi,4)))

# construct the pythtb model
lat = [[1.0]]
orbs = [[-0.25],[0.25]]
my_model = tb_model(1,1,lat,orbs,nspin=1)
my_model.set_hop(t-delta_t*np.cos(phi),1,0,[0],mode="add")
my_model.set_hop(t+delta_t*np.cos(phi),0,1,[-1],mode="add")
my_model.set_onsite([+Delta*np.sin(phi),-Delta*np.sin(phi)],mode="add")

# 1D energy band calculation
print("Compute 1D energy bands")
path = [[-0.5],[0.0],[0.5]]
numkpts = 201
(k_vec,k_dist,k_node) = my_model.k_path(path,numkpts,report=False)
evals = my_model.solve_all(k_vec)
plt.figure()
for n in range(len(evals[:,0])):
    plt.plot(k_dist,evals[n,:],linewidth=5)
plt.xticks(k_node,[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$k$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.title(r"$t={}$, $\delta t = {}$, $\Delta = {}$, $\phi = {}\pi$".format(round(t,4),round(delta_t,4),round(Delta,4),round(phi/pi,4)),fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/Rice_Mele_chain_energy_bands_t_{}_delta_t_{}_Delta_{}_phi_{}pi.pdf".format(round(t,4),round(delta_t,4),round(Delta,4),round(phi/pi,4)))

# 0D finite size spectrum with open boundary condition
print("Compute 0D finite size spectrum with open boundary condition")
finite_model=my_model.cut_piece(50,0,glue_edgs=False)
(finite_evals,finite_evecs)=finite_model.solve_all(eig_vectors=True)
plt.figure()
plt.plot(finite_evals,'.')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(r"OBC, ${}$ sites, $t={}$, $\delta t = {}$, $\Delta = {}$, $\phi = {}\pi$".format(len(finite_evals),round(t,4),round(delta_t,4),round(Delta,4),round(phi/pi,4)),fontsize=15)
plt.xlabel(r"eigenstate index",fontsize=15)
plt.ylabel(r"$E$",fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/Rice_Mele_chain_OBC_spectrum_{}_sites_t_{}_delta_t_{}_Delta_{}_phi_{}pi.pdf".format(len(finite_evals),round(t,4),round(delta_t,4),round(Delta,4),round(phi/pi,4)))

# compute the Wilson loop eigenphase and Wannier band basis using wannier_band_basis_array
print("Compute the Wilson loop eigenphase and Wannier band basis using wannier_band_basis_array")
array1=wf_array(my_model,[51]) # wf_array in pythtb
array1.solve_on_grid([-0.5]) 
dir1=0 # this line can be commented out without affecting the 1D result
array2=wannier_band_basis_array(array1,[0],dir1) # wannier_band_basis_array in nestedWilsonLib
array2.solve_wannier_band_basis() # get the wannier band basis and wannier band energy (wilson loop eigenphases)
print("Wilson loop eigenphase (Berry phase) = {}*pi".format(array2._wannier_band_energy/pi))
print("Wannier band basis wave function is {}".format(array2._wfs[0]))
print("Wannier band basis probability distribution is {}".format(np.abs(array2._wfs[0])**2))

with open(path_data+'/Wannier_band_basis_result.txt','w') as f:
    f.write("The 1D Rice Mele model used in this calculation has parameters t = {}, delta t = {}, Delta = {}, phi = {}*pi\n".format(t,delta_t,Delta,phi/np.pi))
    f.write("Wilson loop eigenphase (Berry phase) for the lower energy band is (unit: pi):\n")
    np.savetxt(f,array2._wannier_band_energy/pi)
    f.write("The (two-component) corresponding Wannier band basis wave function is:\n")
    np.savetxt(f,array2._wfs[0])
    f.write("whose probability distribution is:\n")
    np.savetxt(f,np.abs(array2._wfs[0])**2)

########################################################################################################

t=1.0
delta_t=-0.1
Delta=0.5
lat = [[1.0]]
orbs = [[-0.25],[0.25]]

print("Second part of the calculation: fix t = {}, delta_t = {}, Delta = {}, vary phi from 0 to 2*pi and study the spectral flow".format(round(t,4),round(delta_t,4),round(Delta,4)))

phi_list=np.linspace(0.0,2.0*pi,num=101,endpoint=True)
gamma_1_data=[]
for phi in phi_list:
    my_model = tb_model(1,1,lat,orbs,nspin=1)
    my_model.set_hop(t-delta_t*np.cos(phi),1,0,[0],mode="add")
    my_model.set_hop(t+delta_t*np.cos(phi),0,1,[-1],mode="add")
    my_model.set_onsite([+Delta*np.sin(phi),-Delta*np.sin(phi)],mode="add")
    
    array1=wf_array(my_model,[51]) # wf_array in pythtb
    array1.solve_on_grid([-0.5]) 
    dir1=0 # this line can be commented out without affecting the 1D result
    array2=wannier_band_basis_array(array1,[0],dir1) # wannier_band_basis_array in nestedWilsonLib
    array2.solve_wannier_band_basis() # get the wannier band basis and wannier band energy
    
    gamma_1_data.append(array2._wannier_band_energy)

gamma_1_data=np.array(gamma_1_data)

# plot the spectral flow
plt.figure()
for n in range(len(gamma_1_data[0,:])):
    plt.plot(gamma_1_data[:,n],phi_list,'o')
plt.ylabel(r"$\phi$",fontsize=20)
plt.xlabel(r"$\gamma_1$ (Berry phase)",fontsize=20)
plt.yticks([0.0,np.pi,2.0*np.pi],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.xticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.title(r"1D Rice-Mele chain spectral flow",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/Rice_Mele_chain_spectral_flow_t_{}_delta_t_{}_Delta_{}.pdf".format(round(t,4),round(delta_t,4),round(Delta,4)))

# end of the code
print("Done")

