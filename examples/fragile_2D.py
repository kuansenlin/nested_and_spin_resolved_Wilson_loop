# Example of nested Wilson loop calculation using a 2D model of fragile topology
# The model can be found in Eqs. (D15)-(D16) of arXiv:1908.00016v2, or Eqs. (C55) and (C60) of arXiv:2207.10099v1.
# The parameter choices are in the caption of Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1.
# The fragile bands are coupled to a set of trivial bands to gap out its Wilson loop spectrum.
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The calculations in this python file include
# 1. 2D bulk energy bands along high-symmetry lines in k-space.
# 2. Symmetry checking for C4 and mirror My.
#    The verification of C4 and mirror My symmetry is by checking the energy spectrum at a set of random k points.
# 3. 1D ribbon energy bands, ribbon is finite along x or y.
# 4. kx-directed or ky-directed Wannier bands for the two fragile bands (not including the trivial bands) 
#    The two fragile bands correspond to P_2 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1
#    This demonstrates the fragile winding of the Wilson loop spectrum if we consider only the two fragile bands.
# 5. kx-directed or ky-directed Wannier bands for the lower 6 bands (2 fragile + 4 trivial) with "properly 
#    chosen window" in the Wannier band energies separating inner (4) and outer (2) 
#    Wannier bands. Inner Wannier bands center around 0 while outer Wannier bands center around pi (or -pi).
#    The projector onto the lower 6 bands corresponds to P_6 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1.
# 6. Eigenphases of the nested Wilson loop for both the inner (4) and outer (2) Wannier bands.
#    The occupied space is the same as 5., namely P_6 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1.
#    We will consider two ways depicted in Fig. 20 of PRB 96, 245115 (2017).
#    Note that nested Wilson loop eigenphases for a 2D system can have dependence on the momentum along which we perform the calculation to obtain the Wannier band basis.
#    The base points in the calculation 4., 5., and 6. are chosen as [0.0,0.0] in this calculation.
# 7. Demonstration the momentum-dependence of the nested Wilson loop eigenphases.
#    The occupied space is the same as 5., namely P_6 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1.
#    We will again consider two ways depicted in Fig. 20 of PRB 96, 245115 (2017).

# In this example code we use interchangably [kx,ky] and [k1,k2].

# In this example code, whenever we do a nested Wilson loop calculation, the occupied electronic energy bands
# are always the lower 6 bands (2 fragile + 4 trivial), namely P_6 in Fig. 7 in the Supplementary Appendices of arXiv:2207.10099v1.

## Generically, user should check that the grid number choice in 4., 5., 6., and 7. leads to convergent results for
## both the Wilson loop and nested Wilson loop, by gradually increasing the grid number. Here, we 
## choose [501,51], where 501 as a large grid number along kx is to accurately sample the k points in 
## the ky-directed Wannier bands, such that the nested Wilson loop computed along kx is accurate enough.
## On the other hand, we also have a calculation with grid number choice [51,501] where 501 as a large 
## grid number along ky is to accurately sample the k points in the kx-directed Wannier bands, such that 
## the nested Wilson loop computed along ky is accurate enough.

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy.linalg as lin
from pythtb import *
from nestedWilsonLib import * # import nested Wilson loop library
import timeit
import os

path_data="./fragile_2D"
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

########### 2D bulk energy bands ###########

print("Compute the 2D bulk energy bands")

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
start_time = timeit.default_timer()
evals = my_model.solve_all(k_vec)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

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

########### check C4 and My symmetries ###########

print("Check C4 and My symmetries using the energy spectrum at a set of random k points")

# function to return C4 acting on k
def C4_k(k):
    assert len(k)==2
    return np.array([-k[1],k[0]])
# function to return My acting on k
def My_k(k):
    assert len(k)==2
    return np.array([k[0],-k[1]]) 

random_k_list=np.random.random((100,2))
print("Will consider the energy eigenvalues at {} random k points and their corresponding C4 k and My k".format(random_k_list.shape[0]))
E_list=[]
start_time = timeit.default_timer()
for random_k in random_k_list:
    E_k=my_model.solve_one(random_k)
    E_C4k=my_model.solve_one(C4_k(random_k))
    E_Myk=my_model.solve_one(My_k(random_k))
    E_list.append([E_k,E_C4k,E_Myk])
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

# check whether the energies at k, C4 k, and My K are close enough
T_F_collection=[]
for E in E_list:
    E_k=E[0]
    E_C4k=E[1]
    E_Myk=E[2]
    T_F_collection.append(np.isclose(E_k,E_C4k))
    T_F_collection.append(np.isclose(E_k,E_Myk))
    T_F_collection.append(np.isclose(E_C4k,E_Myk))
T_F_collection=np.array(T_F_collection)
T_F_collection=T_F_collection.flatten()
assert np.all(T_F_collection)

print("The energy eigenvalues at k, C4 k, My k are checked to be numerically close enough in terms of np.isclose() for a set of random k points\n")

########### 1D ribbon energy bands ###########

print("Compute 1D ribbon energy bands (finite along x)")

start_time = timeit.default_timer()
xribbon=my_model.cut_piece(20,0,glue_edgs=False)
(xribbon_k_vec,xribbon_k_dist,xribbon_k_node) = xribbon.k_path([[-0.5],[0.0],[0.5]],201,report=False)
xribbon_evals = xribbon.solve_all(xribbon_k_vec)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

for n in range(len(xribbon_evals[:,0])):
    plt.plot(xribbon_k_dist,xribbon_evals[n,:])
plt.xlim([np.min(xribbon_k_dist),np.max(xribbon_k_dist)])
plt.xticks(xribbon_k_node,[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$\overline{k}_y$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.title(r"ribbon finite along $x$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/xribbon_energy_bands.pdf")
plt.close()

print("Compute 1D ribbon energy bands (finite along y)")

start_time = timeit.default_timer()
yribbon=my_model.cut_piece(20,1,glue_edgs=False)
(yribbon_k_vec,yribbon_k_dist,yribbon_k_node) = yribbon.k_path([[-0.5],[0.0],[0.5]],201,report=False)
yribbon_evals = yribbon.solve_all(yribbon_k_vec)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

for n in range(len(yribbon_evals[:,0])):
    plt.plot(yribbon_k_dist,yribbon_evals[n,:])
plt.xlim([np.min(yribbon_k_dist),np.max(yribbon_k_dist)])
plt.xticks(yribbon_k_node,[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$\overline{k}_x$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.title(r"ribbon finite along $y$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/yribbon_energy_bands.pdf")
plt.close()

########### Wilson loop calculation for the two fragile bands ###########

print("Wilson loop calculation for the two fragile bands")

print("The base point in the calculation is chosen to be [0.0,0.0]")

print("First do kx-directed Wilson loop to obtain the kx-directed Wannier bands")

start_time = timeit.default_timer()
dir1=0 # we have two directions, kx and ky, to choose, and here dir1=0 means the first direction, namely kx
energy_band_ind=[4,5]
array1=wf_array(my_model,[51,501]) 
array1.solve_on_grid([0.0,0.0]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,6.5))
for n in range(pha.shape[1]):
    plt.plot(pha[:,n],'.')
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.title(r"$P_2$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_directed_Wannier_bands_P_2_fragile_winding.pdf")
plt.close()

print("Next do ky-directed Wilson loop to obtain the ky-directed Wannier bands")

start_time = timeit.default_timer()
dir1=1 # we have two directions, kx and ky, to choose, and here dir1=1 means the second direction, namely ky
energy_band_ind=[4,5]
array1=wf_array(my_model,[501,51]) 
array1.solve_on_grid([0.0,0.0]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

plt.figure(figsize=(5.2,6.5))
for n in range(pha.shape[1]):
    plt.plot(pha[:,n],'.')
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $\gamma_1$",fontsize=20)
plt.title(r"$P_2$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/ky_directed_Wannier_bands_P_2_fragile_winding.pdf")
plt.close()

########### nested Wilson loop calculation ###########

print("Now begin nested Wilson loop calculation: first along kx and then ky")

print("The occupied bands are now chosen to be the lower 6 bands (2 fragile bands + 4 trivial bands)")

print("The base point in the calculation is chosen to be [0.0,0.0]")

print("First do Wilson loop along kx to obtain the kx-directed Wannier bands")

start_time = timeit.default_timer()
dir1=0 # we have two directions, kx and ky, to choose, and here dir1=0 means the first direction, namely kx
energy_band_ind=[0,1,2,3,4,5]
array1=wf_array(my_model,[51,501]) 
array1.solve_on_grid([0.0,0.0]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

window_1=-0.96*np.pi
window_2=+0.96*np.pi

plt.figure(figsize=(5.2,6.5))
for n in range(pha.shape[1]):
    plt.plot(pha[:,n],'.')
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$k_x$-directed $\gamma_1$",fontsize=20)
plt.title(r"$P_6$",fontsize=20)
plt.legend(loc='center right',fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_directed_gapped_Wannier_bands_P_6.pdf")
plt.close()

print("Then do nested Wilson loop along ky")

start_time = timeit.default_timer()
dir2=0 # since ky is the only remaining direction, choosing dir2=0 here means integrating along ky, this line can be commented out without affecting the results
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=4)
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

with open(path_data+'/nested_Wilson_loop_result.txt','w') as f:
    f.write("If we do nested Wilson loop calculation first along kx and then ky, then the nested Wilson loop eigenphase for the Wannier band within the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_inner/np.pi)
    f.write(", and the nested Wilson loop eigenphase for the Wannier band outside the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_outer/np.pi)
    f.write("Since the base point of [kx,ky] is [0.0,0.0] in the reduced coordinate, in this calculation we obtain the nested Wilson loop eigenphase at kx = 0.0 in the reduced coordinate.\n")
    f.write("\n\n")

print("Nested Wilson loop calculation: first along ky and then kx")

print("The occupied bands are now chosen to be the lower 6 bands (2 fragile bands + 4 trivial bands)")

print("The base point in the calculation is chosen to be [0.0,0.0]")

print("First do Wilson loop along ky to obtain the ky-directed Wannier bands")

start_time = timeit.default_timer()
dir1=1 # we have two directions, kx and ky, to choose, and here dir1=1 means the second direction, namely ky
energy_band_ind=[0,1,2,3,4,5]
array1=wf_array(my_model,[501,51]) 
array1.solve_on_grid([0.0,0.0]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

window_1=-0.96*np.pi
window_2=+0.96*np.pi

plt.figure(figsize=(5.2,6.5))
for n in range(pha.shape[1]):
    plt.plot(pha[:,n],'.')
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$",r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$k_y$-directed $\gamma_1$",fontsize=20)
plt.title(r"$P_6$",fontsize=20)
plt.legend(loc='center right',fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/ky_directed_gapped_Wannier_bands_P_6.pdf")
plt.close()

print("Then do nested Wilson loop along kx")

start_time = timeit.default_timer()
dir2=0 # since kx is the only remaining direction, choosing dir2=0 here means integrating along kx, this line can be commented out without affecting the results
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=4)
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

with open(path_data+'/nested_Wilson_loop_result.txt','a') as f:
    f.write("If we do nested Wilson loop calculation first along ky and then kx, then the nested Wilson loop eigenphase for the Wannier band within the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_inner/np.pi)
    f.write(", and the nested Wilson loop eigenphase for the Wannier band outside the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_outer/np.pi)
    f.write("Since the base point of [kx,ky] is [0.0,0.0] in the reduced coordinate, in this calculation we obtain the nested Wilson loop eigenphase at ky = 0.0 in the reduced coordinate.\n")

########### dependence of the nested Wilson loop eigenphases on base point momentum ###########

print("Momentum-dependence of the nested Wilson loop eigenphases")

print("First consider the case where we integrate along kx and then ky, such that the nested Wilson loop eigenphases depend on kx")

print("The occupied bands are now chosen to be the lower 6 bands (2 fragile bands + 4 trivial bands)")

print("Method: compute the nested Wilson loop using different base points in kx")

nested_inner_collection=[]
nested_outer_collection=[]

start_time = timeit.default_timer()
for k in np.linspace(-0.5,0.5,num=101,endpoint=True):
    dir1=0 # we have two directions, kx and ky, to choose, and here dir1=0 means the first direction, namely kx
    energy_band_ind=[0,1,2,3,4,5]
    array1=wf_array(my_model,[51,501]) 
    array1.solve_on_grid([k,0.0]) 
    array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
    array2.solve_wannier_band_basis()
    pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
    dir2=0 # since ky is the only remaining direction, choosing dir2=0 here means integrating along ky, this line can be commented out without affecting the results
    window_1=-0.96*np.pi
    window_2=+0.96*np.pi
    window_list=[window_1,window_2]
    nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=4)
    window_list=[window_2,window_1]
    nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
    nested_inner_collection.append(nested_inner)
    nested_outer_collection.append(nested_outer)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

nested_inner_collection=np.array(nested_inner_collection)
nested_outer_collection=np.array(nested_outer_collection)

plt.figure(figsize=(5.2,5))
for n in range(nested_inner_collection.shape[1]):
    plt.plot(nested_inner_collection[:,n],'o')
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_x)$",fontsize=20)
plt.xticks([0,(nested_inner_collection.shape[0]-1)/2.0,nested_inner_collection.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,nested_inner_collection.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title(r"$P_{in}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_dependence_of_inner_nested_Wilson_loop_eigenphases_first_integrate_along_kx_and_then_ky.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
for n in range(nested_outer_collection.shape[1]):
    plt.plot(nested_outer_collection[:,n],'o')
plt.xlabel(r"$k_x$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_x)$",fontsize=20)
plt.xticks([0,(nested_outer_collection.shape[0]-1)/2.0,nested_outer_collection.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,nested_outer_collection.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title(r"$P_{out}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/kx_dependence_of_outer_nested_Wilson_loop_eigenphases_first_integrate_along_kx_and_then_ky.pdf")
plt.close()

print("Next consider the case where we integrate along ky and then kx, such that the nested Wilson loop eigenphases depend on ky")

print("The occupied bands are now chosen to be the lower 6 bands (2 fragile bands + 4 trivial bands)")

print("Method: compute the nested Wilson loop using different base points in ky")

nested_inner_collection=[]
nested_outer_collection=[]

start_time = timeit.default_timer()
for k in np.linspace(-0.5,0.5,num=101,endpoint=True):
    dir1=1 # we have two directions, kx and ky, to choose, and here dir1=1 means the second direction, namely ky
    energy_band_ind=[0,1,2,3,4,5]
    array1=wf_array(my_model,[501,51]) 
    array1.solve_on_grid([0.0,k]) 
    array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
    array2.solve_wannier_band_basis()
    pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
    dir2=0 # since kx is the only remaining direction, choosing dir2=0 here means integrating along kx, this line can be commented out without affecting the results
    window_1=-0.96*np.pi
    window_2=+0.96*np.pi
    window_list=[window_1,window_2]
    nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=4)
    window_list=[window_2,window_1]
    nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
    nested_inner_collection.append(nested_inner)
    nested_outer_collection.append(nested_outer)
print("--- %s seconds ---\n" % (timeit.default_timer() - start_time))

nested_inner_collection=np.array(nested_inner_collection)
nested_outer_collection=np.array(nested_outer_collection)

plt.figure(figsize=(5.2,5))
for n in range(nested_inner_collection.shape[1]):
    plt.plot(nested_inner_collection[:,n],'o')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_y)$",fontsize=20)
plt.xticks([0,(nested_inner_collection.shape[0]-1)/2.0,nested_inner_collection.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,nested_inner_collection.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title(r"$P_{in}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/ky_dependence_of_inner_nested_Wilson_loop_eigenphases_first_integrate_along_ky_and_then_kx.pdf")
plt.close()

plt.figure(figsize=(5.2,5))
for n in range(nested_outer_collection.shape[1]):
    plt.plot(nested_outer_collection[:,n],'o')
plt.xlabel(r"$k_y$",fontsize=20)
plt.ylabel(r"$\gamma_2 (k_y)$",fontsize=20)
plt.xticks([0,(nested_outer_collection.shape[0]-1)/2.0,nested_outer_collection.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,nested_outer_collection.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.title(r"$P_{out}$ of $P_6$",fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/ky_dependence_of_outer_nested_Wilson_loop_eigenphases_first_integrate_along_ky_and_then_kx.pdf")
plt.close()

# end of the code
print("Done")

