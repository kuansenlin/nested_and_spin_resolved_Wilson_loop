# Example of nested Wilson loop calculation using the 2D Benalcazar-Bernevig-Hughes (BBH) model of a 2D quadrupole insulator
# The model can be found in Eqs. (6.28)--(6.29) of PRB 96, 245115 (2017)
# The code is run using python3
# The code is run using pythtb version 1.7.2

# The calculations in this python file include
# 1. verification that for the chosen parameters the 2D model is gapped in bulk
# 2. a surface plot of the 2D energy bands
# 3. 1D ribbon energy bands (finite along x or y)
# 4. 0D finite size energy spectrum and the probability distribution of the corner modes
# 5. nested Wilson loop calculation (two ways depicted in Fig. 20 of PRB 96, 245115 (2017))

# Notice that the nested Wilson loop eigenphases can have momentum-dependence. 
# For instance, if we obtain the nested Wilson loop by doing Wilson loop first along k1 and then along k2,
# then the nested Wilson loop eigenphases depend on the base point along k1.
# In this example code, the wave function array has base point [-0.5,-0.5] in the reduced coordinate.
# Therefore, the nested Wilson loop eigenphases obtained in this example code correspond to the those at 
# either k1 = -0.5 or k2 = -0.5 in the reduced coordinate, depending on whether the first direction for the Wilson loop calculation is k1 or k2.

# (k1,k2) and (kx,ky) are used interchangeably in this example code
# The first (a1) and second (a2) lattice directions are used interchangeably as x and y, respectively, in this example code

# The model parameters chosen in this example file lie in the middle blue region of Fig. 25 in PRB 96, 245115 (2017)

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plt.switch_backend('agg')
from pythtb import *
from nestedWilsonLib import *
import timeit
import os

# directory to save the data
path_data="./BBH_quadrupole_2D"
if not os.path.exists(path_data):
    os.mkdir(path_data)

########### Construct the model ###########

# model parameters
gamma_x = 0.75
lambda_x = 1.0
gamma_y = 0.75
lambda_y = 1.0

lat = [[1.0,0.0],[0.0,1.0]]
orbs = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
def quadrupole2D():
    
    # initiate the tb model
    model = tb_model(2,2,lat,orbs,nspin=1)
    
    # add gamma_x
    model.set_hop(gamma_x,0,2,[0,0],mode="add")
    model.set_hop(gamma_x,1,3,[0,0],mode="add")
    
    # add gamma_y
    model.set_hop(gamma_y,0,3,[0,0],mode="add")
    model.set_hop(-gamma_y,1,2,[0,0],mode="add")
    
    # add lambda_x
    model.set_hop(lambda_x,0,2,[1,0],mode="add")
    model.set_hop(lambda_x,3,1,[1,0],mode="add")
    
    # add lambda_y
    model.set_hop(lambda_y,0,3,[0,1],mode="add")
    model.set_hop(-lambda_y,2,1,[0,1],mode="add")
    
    return model

print("Construct the model")
my_model = quadrupole2D()

########### verification of the 2D bulk gap ###########

print("Compute energy spectrum and bulk gap over uniformly sampled grids in 2D BZ")

k_list=np.linspace(-0.5,0.5,num=51,endpoint=True)
kvec_list=[]
evals_list=[]
gap_list=[]
start_time = timeit.default_timer()
for kx in k_list:
    for ky in k_list:
        kvec_list.append([kx,ky])
        evals=my_model.solve_one([kx,ky])
        evals_list.append(evals)
        gap_list.append(evals[len(evals)//2]-evals[len(evals)//2-1])
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

kvec_list=np.array(kvec_list)
evals_list=np.array(evals_list)
gap_list=np.array(gap_list)

plt.figure()
plt.plot(np.sort(gap_list),'.',color='black')
plt.axhline(y=0.0,ls='--',color='C0',label=r"$\Delta = 0.0$")
plt.xlabel(r"sorted $\mathbf{k}$ index",fontsize=20)
plt.ylabel(r"$\Delta = E[{}]-E[{}]$".format(len(evals)//2,len(evals)//2-1),fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.3*len(gap_list),1.0,r"min$(\Delta)\approx{}$".format(round(np.min(gap_list),6)),bbox=props,fontsize=20)
plt.title(r"grid = ${} \times {}$".format(len(k_list),len(k_list)),fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(path_data+"/bulk_gap_sampled_over_2D_BZ.pdf")
plt.close()

########### 2D bulk energy bands (surface plot) ###########

print("Plot the 2D bulk energy bands")

x = kvec_list[:,0]
y = kvec_list[:,1]

fig = plt.figure(figsize=(6,4.5))
ax = fig.add_subplot(111, projection="3d")
for n in range(len(evals_list[0,:])):
    z = evals_list[:,n]
    ax.plot_trisurf(x,y,z)
ax.set_xlabel(r"$k_1$",fontsize=20,labelpad=10)
ax.set_ylabel(r"$k_2$",fontsize=20,labelpad=10)
ax.set_zlabel(r"$E$",fontsize=20)
ax.zaxis.set_tick_params(labelsize=20)
ax.set_xticks([-0.5,-0.25,0.0,0.25,0.5])
ax.set_xticklabels([r"$-\pi$",r"$-0.5\pi$",r"$0$",r"$0.5\pi$",r"$\pi$"],fontsize=20)
ax.set_yticks([-0.5,-0.25,0.0,0.25,0.5])
ax.set_yticklabels([r"$-\pi$",r"$-0.5\pi$",r"$0$",r"$0.5\pi$",r"$\pi$"],fontsize=20)
fig.tight_layout()
plt.savefig(path_data+"/bulk_bands_2D.pdf")
plt.close()

########### 1D ribbon energy bands ###########

print("Compute 1D ribbon energy bands (finite along x)")

start_time = timeit.default_timer()
xribbon=my_model.cut_piece(20,0,glue_edgs=False)
(xribbon_k_vec,xribbon_k_dist,xribbon_k_node) = xribbon.k_path([[-0.5],[0.0],[0.5]],201,report=False)
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
plt.tight_layout()
plt.savefig(path_data+"/x_finite_ribbon_bands.pdf")
plt.close()

print("Compute 1D ribbon energy bands (finite along y)")

start_time = timeit.default_timer()
yribbon=my_model.cut_piece(20,1,glue_edgs=False)
(yribbon_k_vec,yribbon_k_dist,yribbon_k_node) = yribbon.k_path([[-0.5],[0.0],[0.5]],201,report=False)
yribbon_evals = yribbon.solve_all(yribbon_k_vec)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

plt.figure()
for n in range(len(yribbon_evals[:,0])):
    plt.plot(yribbon_k_dist,yribbon_evals[n,:])
plt.xticks(yribbon_k_node,[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$\overline{k}_x$",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.title(r"ribbon finite along $y$",fontsize=20)
plt.tight_layout()
plt.savefig(path_data+"/y_finite_ribbon_bands.pdf")
plt.close()

########### 0D finite size energy spectrum ###########

print("Compute 0D finite size energy spectrum")

# 0D finite size calculation
start_time = timeit.default_timer()
temp=my_model.cut_piece(30,0,glue_edgs=False)
finite_model=temp.cut_piece(30,1,glue_edgs=False)
(finite_evals,finite_evecs)=finite_model.solve_all(eig_vectors=True)
print("--- %s seconds ---" % (timeit.default_timer() - start_time))
plt.figure()
within_ind=[i for i in range(len(finite_evals)) if -0.75<=finite_evals[i] and finite_evals[i]<=0.75]
plt.plot(within_ind,finite_evals[within_ind],'.',color='black')
plt.xlim([np.min(within_ind)-1,np.max(within_ind)+1])
plt.xlabel(r"eigenstate index",fontsize=20)
plt.ylabel(r"$E$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_data+"/zoom_in_finite_size_energy_spectrum.pdf")
plt.close()

print("Plot the probability distribution of corner modes")

pos_array=[]
for i in range(int(finite_model._nsta/my_model._nsta)):
    pos_array.append(finite_model.get_orb()[4*i])
pos_array=np.array(pos_array)

wvfn_ind=np.arange(finite_model._nsta//2-2,finite_model._nsta//2+2)
wvfn_middle=finite_evecs[wvfn_ind,:]
prob_middle=np.sum(np.abs(wvfn_middle)**2,axis=0)
prob_cell_middle=[]
for i in range(int(finite_model._nsta/my_model._nsta)):
    prob_cell_middle.append(np.sum(prob_middle[i*my_model._nsta:(i+1)*my_model._nsta]))
prob_cell_middle=np.array(prob_cell_middle)

plt.figure()
avg_prob_cell_middle=prob_cell_middle/len(wvfn_ind)
print("The energies of the four states at the middle of the spectrum are")
print(finite_evals[wvfn_ind])
print("The total probability of the averaged probability distribution for the four states at the middle of the spectrum is")
print(np.sum(avg_prob_cell_middle))
assert np.isclose(np.sum(avg_prob_cell_middle),1.0)
assert np.min(avg_prob_cell_middle)>=0.0
plt.scatter(pos_array[:,0],pos_array[:,1],c=avg_prob_cell_middle,vmin=0.0,cmap='binary')
plt.axis("square")
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel(r"$x$",fontsize=20)
plt.ylabel(r"$y$",fontsize=20)
plt.title("{} corner nearly zero modes".format(len(wvfn_ind)),fontsize=15)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
cb.ax.set_title(r"$\langle P(\mathbf{x}) \rangle$",fontsize=20)
plt.tight_layout()
plt.savefig(path_data+"/corner_mode_probability_distribution.pdf")
plt.close()

########### nested Wilson loop calculation ###########

print("Nested Wilson loop calculation: first along k1 and then k2")

print("First do Wilson loop along k1 to obtain the k1-directed Wannier bands")

start_time = timeit.default_timer()
dir1=0 # we have two directions, k1 and k2, to choose, and here dir1=0 means the first direction, namely k1
energy_band_ind=[0,1]
array1=wf_array(my_model,[101,101]) 
array1.solve_on_grid([-0.5,-0.5]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

window_1=-0.75*np.pi
window_2=+0.0*np.pi

for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'o')
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.xlabel(r"$k_2$",fontsize=20)
plt.ylabel(r"$k_1$-directed $\gamma_1$",fontsize=20)
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(path_data+"/k1_directed_Wannier_bands.pdf")
plt.close()

print("Then do nested Wilson loop along k2")

start_time = timeit.default_timer()
dir2=0 # since k2 is the only remaining direction, choosing dir2=0 here means integrating along k2, this line can be commented out without affecting the results
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1) # can remove the part wnum=1 without affecting the results
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1) # can remove the part wnum=1 without affecting the results
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

with open(path_data+'/nested_Wilson_loop_result.txt','w') as f:
    f.write("If we do nested Wilson loop calculation first along k1 and then k2, then the nested Wilson loop eigenphase for the Wannier band within the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_inner/np.pi)
    f.write(", and the nested Wilson loop eigenphase for the Wannier band outside the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_outer/np.pi)
    f.write("Since the base point of [k1,k2] is [-0.5,-0.5] in the reduced coordinate, in this calculation we obtain the nested Wilson loop eigenphase at k1 = -0.5 in the reduced coordinate.\n")
    f.write("\n\n")

print("Nested Wilson loop calculation: first along k2 and then k1")

print("First do Wilson loop along k2 to obtain the k2-directed Wannier bands")

start_time = timeit.default_timer()
dir1=1 # we have two directions, k1 and k2, to choose, and here dir1=1 means the second direction, namely k2
energy_band_ind=[0,1]
array1=wf_array(my_model,[101,101]) 
array1.solve_on_grid([-0.5,-0.5]) 
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
pha=array2._wannier_band_energy # get the wannier band energy out of wannier_band_basis_array
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

window_1=-0.75*np.pi
window_2=+0.0*np.pi

for n in range(len(pha[0,:])):
    plt.plot(pha[:,n],'o')
plt.xticks([0,(pha.shape[0]-1)/2.0,pha.shape[0]-1],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
plt.xlim([0,pha.shape[0]-1])
plt.xlabel(r"$k_1$",fontsize=20)
plt.ylabel(r"$k_2$-directed $\gamma_1$",fontsize=20)
plt.axhline(y=window_2,ls='--',color='r',label=r"${}\pi$".format(round(window_2/np.pi,6)))
plt.axhline(y=window_1,ls='--',color='b',label=r"${}\pi$".format(round(window_1/np.pi,6)))
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(path_data+"/k2_directed_Wannier_bands.pdf")
plt.close()

print("Then do nested Wilson loop along k1")

start_time = timeit.default_timer()
dir2=0 # since k1 is the only remaining direction, choosing dir2=0 here means integrating along k1, this line can be commented out without affecting the results
window_list=[window_1,window_2]
nested_inner=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1) # can remove the part wnum=1 without affecting the results
window_list=[window_2,window_1]
nested_outer=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1) # can remove the part wnum=1 without affecting the results
print("--- %s seconds ---" % (timeit.default_timer() - start_time))

with open(path_data+'/nested_Wilson_loop_result.txt','a') as f:
    f.write("If we do nested Wilson loop calculation first along k2 and then k1, then the nested Wilson loop eigenphase for the Wannier band within the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_inner/np.pi)
    f.write(", and the nested Wilson loop eigenphase for the Wannier band outside the window [{}pi,{}pi] is: (in unit of pi)\n".format(window_1/np.pi,window_2/np.pi))
    np.savetxt(f,nested_outer/np.pi)
    f.write("Since the base point of [k1,k2] is [-0.5,-0.5] in the reduced coordinate, in this calculation we obtain the nested Wilson loop eigenphase at k2 = -0.5 in the reduced coordinate.\n")

# end of the code
print("Done")

