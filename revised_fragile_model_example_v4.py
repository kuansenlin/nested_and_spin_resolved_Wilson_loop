#Run this demo script for the revised version of 2D fragile model in Eqs.(D15)-(D16) of arXiv:1908.00016 to return its:

#1.  Bulk energy bands along high symmetry lines in k-space.
#2.  Printed-out symmetry checking for C4 and mirror My.
#3.  Ribbon energy bands, ribbon is finite along y.
#4.  y-directed Wannier bands (and the zoom-in around pi and -pi) for the lower 6 bands with "properly 
#    chosen window" in the Wannier band energy separating inner (4) and outer (2) Wannier bands. Inner Wannier 
#    bands center around 0 while outer Wannier bands center around pi (or -pi).
#5.  Eigen-phases of the nested Wilson loop for both the inner (4) and outer (2) Wannier bands.

## the execution time for each calculation will also be printed out.

## the base point in the calculation 4. and 5. is chosen as [0.0,0.0]

## generically, user should check the grid number choice in 4. and 5. leads to convergent results for
## both the Wilson loop and nested Wilson loop, by gradually increasing the grid number. Here, we 
## choose [501,51], where 501 as a large grid number along kx is to accurately sample the k points in 
## the y-directed Wannier bands, such that the nested Wilson loop computed along kx is accurate enough.

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib_v4 import *
import timeit

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

def revised_fragile_model():
    
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

my_model = revised_fragile_model()

#############################################
####Plot the band structure##################
#############################################
print("Compute the energy band structure")

# high symmetry k point
Gamma = [0.0,0.0]
X = [0.5,0.0]
Y = [0.0,0.5]
M = [0.5,0.5]

# get the path
path = [Gamma,X,M,Gamma]
label = (r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$')
numkpts = 201
(k_vec,k_dist,k_node) = my_model.k_path(path,numkpts,report=False)

# get the eigenvalues
start_time = timeit.default_timer()
evals = my_model.solve_all(k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

# now do the plotting
fig, ax = plt.subplots(figsize=(10,5))

# specify horizontal axis details
ax.set_xlim(k_node[0],k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(label,fontsize=30)
ax.set_yticks([-15,-10,-5,0,5,10])
ax.set_yticklabels([-15,-10,-5,0,5,10],fontsize=30)
for n in range(len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color='k')

# plot bands
assert my_model._norb*my_model._nspin == len(evals[:,0])
for n in range(my_model._norb*my_model._nspin):
    ax.plot(k_dist,evals[n])

# put title
ax.set_xlabel(r"$\mathbf{k}$",fontsize=30)
ax.set_ylabel(r"$E$",fontsize=30)

# make an PDF figure of a plot
fig.tight_layout()
fig.savefig("revised_fragile_model_energy_band.pdf")
plt.close()

print("\n")

######################################
####Check symmetries##################
######################################

## check C4 and mirror My symmetry
print("Check the symmetries of the model")
Ea=my_model.solve_one([0.212,0.367])
Eb=my_model.solve_one([-0.367,0.212]) # C4
Ec=my_model.solve_one([0.212,-0.367]) # mirror My
assert np.all(np.isclose(Ea,Eb))
print("The model has C4 symmetry")
assert np.all(np.isclose(Ea,Ec))
print("The model has mirror My symmetry")

print("\n")

######################################
####Plot ribbon band structure########
######################################
print("Compute the ribbon energy band structure")
ribbon_model=my_model.cut_piece(21,1,glue_edgs=False)

# get the path
ribbon_path = [[-0.5],[0.0],[0.5]]
ribbon_label = (r'$-\pi$',r'$0$',r'$\pi$')
numkpts = 201
(ribbon_k_vec,ribbon_k_dist,ribbon_k_node) = ribbon_model.k_path(ribbon_path,numkpts,report=False)

# get the eigenvalues
start_time = timeit.default_timer()
ribbon_evals = ribbon_model.solve_all(ribbon_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

# now do the plotting
fig, ax = plt.subplots(figsize=(10,5)) #figsize=(1.25*figure_size,0.75*figure_size))

# specify horizontal axis details
ax.set_xlim(ribbon_k_node[0],ribbon_k_node[-1])
ax.set_xticks(ribbon_k_node)
ax.set_xticklabels(ribbon_label,fontsize=30)
ax.set_yticks([-15,-10,-5,0,5,10])
ax.set_yticklabels([-15,-10,-5,0,5,10],fontsize=30)
for n in range(len(ribbon_k_node)):
    ax.axvline(x=ribbon_k_node[n], linewidth=0.5, color='k')

# plot bands
assert ribbon_model._norb*ribbon_model._nspin == len(ribbon_evals[:,0])
for n in range(ribbon_model._norb*ribbon_model._nspin):
    ax.plot(ribbon_k_dist,ribbon_evals[n])

# put title
ax.set_xlabel(r"$k_1$",fontsize=30)
ax.set_ylabel(r"$E$",fontsize=30)

# make an PDF figure of a plot
fig.tight_layout()
fig.savefig("revised_fragile_model_ribbon_energy_band.pdf")
plt.close()

print("\n")

############################################
####Plot the Wannier bands##################
############################################

### use the nested Wilson loop library to compute the Wannier bands
print("Compute Wilson loop")
start_time = timeit.default_timer()
array1=wf_array(my_model,[501,51]) # wf_array in pythtb
array1.solve_on_grid([0.0,0.0]) 
dir1=1
array2=wannier_band_basis_array(array1,[0,1,2,3,4,5],dir1) # wannier_band_basis_array in nestedWilsonLib
array2.solve_wannier_band_basis() # get the wannier band basis and wannier band energy
print("--- %s seconds ---" % (timeit.default_timer() - start_time))
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("The shape of _wannier_band_energy array is {}".format(pha.shape))

## plot the Wannier band
window_1=-0.96*np.pi
window_2=+0.96*np.pi

plt.figure()
for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'.',markersize=4.0)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$", r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
plt.xlabel(r'$k_1$',fontsize=30)
plt.ylabel(r'$k_2$-directed $\gamma_1$',fontsize=30)
plt.title('Wannier band',fontsize=20)
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig("revised_fragile_model_Wannier_band.pdf")
plt.close()

## zoom in around pi
plt.figure()
for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'.',markersize=12.0)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([window_2-0.1*np.pi,np.pi])
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$", r"$2\pi$"],fontsize=20)
plt.yticks([window_2-0.1*np.pi,np.pi],[r"${}\pi$".format(round((window_2-0.1*np.pi)/np.pi,6)),r"$\pi$"],fontsize=20)
plt.xlabel(r'$k_1$',fontsize=30)
plt.ylabel(r'$k_2$-directed $\gamma_1$',fontsize=30)
plt.title('Wannier band',fontsize=20)
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig("revised_fragile_model_Wannier_band_around_pi.pdf")
plt.close()

## zoom in around -pi
plt.figure()
for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'.',markersize=12.0)
plt.xlim([0,pha.shape[0]-1])
plt.ylim([-np.pi,window_1+0.1*np.pi])
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$0$",r"$\pi$", r"$2\pi$"],fontsize=20)
plt.yticks([-np.pi,window_1+0.1*np.pi],[r"$-\pi$",r"${}\pi$".format(round((window_1+0.1*np.pi)/np.pi,6))],fontsize=20)
plt.xlabel(r'$k_1$',fontsize=30)
plt.ylabel(r'$k_2$-directed $\gamma_1$',fontsize=30)
plt.title('Wannier band',fontsize=20)
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig("revised_fragile_model_Wannier_band_around_minus_pi.pdf")
plt.close()

print("\n")

###########################################################
####Computation of the nested Wilson loop##################
###########################################################

### computation of nested Wilson loop
print("Compute nested Wilson loop")
start_time = timeit.default_timer()
dir2=0
## nested Wilson loop over the inner Wannier bands
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=4)
## nested Wilson loop over the outer Wannier bands
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

print("\n")

## print out the results of the nested Wilson loop
print("eigen-phases of nested Wilson loop over inner Wannier bands (in unit of pi):")
for xin in nested_inner:
    print("{}".format(xin/np.pi))

print("summed eigen-phases of nested Wilson loop over inner Wannier bands (in unit of pi after mod 2pi): {}".format((np.mod(np.sum(nested_inner)+np.pi,2.0*np.pi)-np.pi)/np.pi))

print("\n")

print("eigen-phases of nested Wilson loop over outer Wannier bands (in unit of pi):")
for xout in nested_outer:
    print("{}".format(xout/np.pi))

print("summed eigen-phases of nested Wilson loop over outer Wannier bands (in unit of pi after mod 2pi): {}".format((np.mod(np.sum(nested_outer)+np.pi,2.0*np.pi)-np.pi)/np.pi))

print("\n")

## end of the code
print("Done")

