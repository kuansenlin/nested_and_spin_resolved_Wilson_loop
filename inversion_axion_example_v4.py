# AXI 3D, model in https://arxiv.org/abs/1810.02373
import numpy as np
import matplotlib.pyplot as pl
pl.switch_backend('agg')
import numpy.linalg as lin
import copy as c
import itertools as it
import warnings
from pythtb import *
import pickle
import os
import time

#Import the new nestedWilson library
from nestedWilsonLib_v4 import *

warnings.filterwarnings("ignore")


I=1j
sqrt3 = np.sqrt(3.0)
sx=np.array([0,1,0,0],dtype='complex')
sy=np.array([0,0,1,0],dtype='complex')
sz=np.array([0,0,0,1],dtype='complex')


#This is the I-symmetric TB model of an AXI from Ref. [2]

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
#This is tPHX=tPHY (and tPHZ=0) in the notation of Ref. [2]
tPH = 0.3


#We will work in the limit of large SOC to have large Wilson gaps

#MAGNETIC TERMS
mA = 1.2
mB = 0.3


#Terms to trivialize the fragile Wilson winding at kz=0

#FRAGILE SPIN SPLITTING
mszF = 2.2 #this is usZ in Ref. [2]

#USE TOPOLOGICAL QUANTUM CHEMISTRY TO MAKE S ORBITALS AT GENERAL POSTION LIKE S AND P (BY (ANTI)BONDING)
vFsp = 20 #this is t_sp in Ref. [2]
#spin chemical potential
muF = -0.5 #t1 in Ref. [2]


#"COUPLED PARAMETERS"
#####################################
#term to couple fragile and trivial bands
vFc1 = 4 #vC in Ref. [2]

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


my_model=axionmodel3D()

# nested Wilson loop calculation

k1_grid = 58
k2_grid = 58
k3_grid = 31
k1_base=0.0
k2_base=0.0
k3_base=0.0
window_1=-0.065*np.pi
window_2=0.065*np.pi
dir1=0
dir2=0
energy_band_ind=[1,2,3]
model_str="AXI"

if dir1==0:
    if dir2==0:
        plot_grid=k3_grid
        plot_str=r"$k_3$"
        plot_xlabel=[r"${}\pi$".format(round(k3_base*2.0,4)),r"${}\pi$".format(round((k3_base+0.5)*2.0,4)),r"${}\pi$".format(round((k3_base+1.0)*2.0,4))]
    elif dir2==1:
        plot_grid=k2_grid
        plot_str=r"$k_2$"
        plot_xlabel=[r"${}\pi$".format(round(k2_base*2.0,4)),r"${}\pi$".format(round((k2_base+0.5)*2.0,4)),r"${}\pi$".format(round((k2_base+1.0)*2.0,4))]
elif dir1==1:
    if dir2==0:
        plot_grid=k3_grid
        plot_str=r"$k_3$"
        plot_xlabel=[r"${}\pi$".format(round(k3_base*2.0,4)),r"${}\pi$".format(round((k3_base+0.5)*2.0,4)),r"${}\pi$".format(round((k3_base+1.0)*2.0,4))]
    elif dir2==1:
        plot_grid=k1_grid
        plot_str=r"$k_1$"
        plot_xlabel=[r"${}\pi$".format(round(k1_base*2.0,4)),r"${}\pi$".format(round((k1_base+0.5)*2.0,4)),r"${}\pi$".format(round((k1_base+1.0)*2.0,4))]
elif dir1==2:
    if dir2==0:
        plot_grid=k2_grid
        plot_str=r"$k_2$"
        plot_xlabel=[r"${}\pi$".format(round(k2_base*2.0,4)),r"${}\pi$".format(round((k2_base+0.5)*2.0,4)),r"${}\pi$".format(round((k2_base+1.0)*2.0,4))]
    elif dir2==1:
        plot_grid=k1_grid
        plot_str=r"$k_1$"
        plot_xlabel=[r"${}\pi$".format(round(k1_base*2.0,4)),r"${}\pi$".format(round((k1_base+0.5)*2.0,4)),r"${}\pi$".format(round((k1_base+1.0)*2.0,4))]

path_data="./" + model_str + "_grid_{}x{}x{}_window_1_{}pi_window_2_{}pi_dir1_{}_dir2_{}_solve_on_grid_{}_{}_{}_energy_band_ind_{}".format(k1_grid,k2_grid,k3_grid,round(window_1/np.pi,4),round(window_2/np.pi,4),dir1,dir2,round(k1_base,4),round(k2_base,4),round(k3_base,4),energy_band_ind)
os.mkdir(path_data)
print(path_data)

start_time = time.time()

print("Start computing the Wilson loop and nested Wilson loop")

# First use wf_array, then use wannier_band_basis_array
array1=wf_array(my_model,[k1_grid,k2_grid,k3_grid]) #mesh is [kxres,kyres,kzres]
array1.solve_on_grid([k1_base,k2_base,k3_base]) #solve wavefunction
array2=wannier_band_basis_array(array1,energy_band_ind,dir1) 
array2.solve_wannier_band_basis()
window_list=[window_1,window_2]
# in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=True)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=False)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=False)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=True,wnum=1)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=False,wnum=1)
# in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=False,wnum=1)
in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1)
window_list=[window_2,window_1]
# out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=True)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=False)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=False)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=True,wnum=2)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=True,berry_evals=False,wnum=2)
# out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=False,wnum=2)
out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)

print("--- %s seconds ---" % (time.time() - start_time))

# in_data=np.array(in_data)
# out_data=np.array(out_data)

print("shape of in_data is {}".format(in_data.shape))
print("shape of out_data is {}".format(out_data.shape))

assert in_data.shape[0]==plot_grid
assert out_data.shape[0]==plot_grid

print("Save the results")

# save the results
with open(path_data+"/wannier_band_energy.pk",'wb') as f:
    pickle.dump(array2._wannier_band_energy,f)

with open(path_data+"/in_data.pk",'wb') as f:
    pickle.dump(in_data,f)

with open(path_data+"/out_data.pk",'wb') as f:
    pickle.dump(out_data,f)

print("Plot the results")

# plot the results
fig = pl.figure(figsize=(10,10))
ax = fig.add_subplot(111)
pl.plot(np.sort((array2._wannier_band_energy).flatten()),'o',color='C0',markersize=7.5)
pl.xlim([0,len(np.sort((array2._wannier_band_energy).flatten()))-1])
pl.ylim([-np.pi,np.pi])
pl.xlabel("eigenstate index",fontsize=20)
# pl.ylabel(r"$\gamma_1 (k_2,k_3)$",fontsize=20)
pl.ylabel(r"$\gamma_1$",fontsize=20)
pl.xticks(fontsize=20)
pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$",r"$\pi$"],fontsize=20)
pl.axhline(y=window_2,ls='--',color='red',label=r"${}\pi$".format(round(window_2/np.pi,4)))
pl.axhline(y=window_1,ls='--',color='blue',label=r"${}\pi$".format(round(window_1/np.pi,4)))
pl.grid(True)
# pl.title("AXI",fontsize=20)
pl.title(model_str,fontsize=20)
pl.legend(fontsize=20)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.tight_layout()
pl.savefig(path_data+"/wannier_band_energy.pdf")
pl.savefig(path_data+"/wannier_band_energy.png")
pl.close()

fig = pl.figure(figsize=(5.75,5.75))
ax = fig.add_subplot(111)
for n in range(len(in_data[0,:])):
    pl.plot(in_data[:,n],'o',color='C0')
pl.xlim([0,len(in_data[:,0])-1])
# pl.ylim([-np.pi,np.pi])
# pl.xlabel(r"$k_3$",fontsize=20)
pl.xlabel(plot_str,fontsize=20)
# pl.ylabel(r"$\gamma_2 (k_1 = 0 , k_3 )$",fontsize=20)
pl.ylabel(r"$\gamma_2$",fontsize=20)
pl.xticks([0,(len(in_data[:,0])-1)/2.0,len(in_data[:,0])-1],plot_xlabel,fontsize=20)
# pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
pl.grid(True)
# pl.title(r"AXI, ${}$ inner".format(len(in_data[0,:])),fontsize=20)
pl.title(model_str + r", ${}$ inner".format(len(in_data[0,:])),fontsize=20)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.tight_layout()
pl.savefig(path_data+"/in_data.pdf")
pl.savefig(path_data+"/in_data.png")
pl.close()

fig = pl.figure(figsize=(5.75,5.75))
ax = fig.add_subplot(111)
pl.plot(np.mod(np.sum(in_data,axis=1)+np.pi,2*np.pi)-np.pi,'o',color='C0')
pl.xlim([0,len(in_data[:,0])-1])
# pl.ylim([-np.pi,np.pi])
# pl.xlabel(r"$k_3$",fontsize=20)
pl.xlabel(plot_str,fontsize=20)
# pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2 (k_1 = 0 , k_3 )$",fontsize=20)
pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2$",fontsize=20)
pl.xticks([0,(len(in_data[:,0])-1)/2.0,len(in_data[:,0])-1],plot_xlabel,fontsize=20)
# pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
pl.grid(True)
pl.title(model_str + r", ${}$ inner".format(len(in_data[0,:])),fontsize=20)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.tight_layout()
pl.savefig(path_data+"/in_data_summed.pdf")
pl.savefig(path_data+"/in_data_summed.png")
pl.close()

fig = pl.figure(figsize=(5.75,5.75))
ax = fig.add_subplot(111)
for n in range(len(out_data[0,:])):
    pl.plot(out_data[:,n],'o',color='C0')
pl.xlim([0,len(out_data[:,0])-1])
# pl.ylim([-np.pi,np.pi])
# pl.xlabel(r"$k_3$",fontsize=20)
pl.xlabel(plot_str,fontsize=20)
# pl.ylabel(r"$\gamma_2 (k_1 = 0 , k_3 )$",fontsize=20)
pl.ylabel(r"$\gamma_2$",fontsize=20)
pl.xticks([0,(len(out_data[:,0])-1)/2.0,len(out_data[:,0])-1],plot_xlabel,fontsize=20)
# pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
pl.grid(True)
pl.title(model_str + r", ${}$ outer".format(len(out_data[0,:])),fontsize=20)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.tight_layout()
pl.savefig(path_data+"/out_data.pdf")
pl.savefig(path_data+"/out_data.png")
pl.close()

fig = pl.figure(figsize=(5.75,5.75))
ax = fig.add_subplot(111)
pl.plot(np.mod(np.sum(out_data,axis=1)+np.pi,2*np.pi)-np.pi,'o',color='C0')
pl.xlim([0,len(out_data[:,0])-1])
# pl.ylim([-np.pi,np.pi])
# pl.xlabel(r"$k_3$",fontsize=20)
pl.xlabel(plot_str,fontsize=20)
# pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2 (k_1 = 0 , k_3 )$",fontsize=20)
pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2$",fontsize=20)
pl.xticks([0,(len(out_data[:,0])-1)/2.0,len(out_data[:,0])-1],plot_xlabel,fontsize=20)
# pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
pl.grid(True)
pl.title(model_str + r", ${}$ outer".format(len(out_data[0,:])),fontsize=20)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.tight_layout()
pl.savefig(path_data+"/out_data_summed.pdf")
pl.savefig(path_data+"/out_data_summed.png")
pl.close()

# fig = pl.figure(figsize=(5.75,5.75))
# ax = fig.add_subplot(111)
# pl.plot(in_data,'o',color='C0')
# pl.xlim([0,len(in_data)-1])
# # pl.ylim([-np.pi,np.pi])
# # pl.xlabel(r"$k_3$",fontsize=20)
# pl.xlabel(plot_str,fontsize=20)
# # pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2 (k_1 = 0 , k_3 )$",fontsize=20)
# pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2$",fontsize=20)
# pl.xticks([0,(len(in_data)-1)/2.0,len(in_data)-1],plot_xlabel,fontsize=20)
# # pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
# pl.grid(True)
# pl.title(model_str + r", inner",fontsize=20)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# fig.tight_layout()
# pl.savefig(path_data+"/in_data_summed.pdf")
# pl.savefig(path_data+"/in_data_summed.png")
# pl.close()

# fig = pl.figure(figsize=(5.75,5.75))
# ax = fig.add_subplot(111)
# pl.plot(out_data,'o',color='C0')
# pl.xlim([0,len(out_data)-1])
# # pl.ylim([-np.pi,np.pi])
# # pl.xlabel(r"$k_3$",fontsize=20)
# pl.xlabel(plot_str,fontsize=20)
# # pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2 (k_1 = 0 , k_3 )$",fontsize=20)
# pl.ylabel(r"$ \mathrm{Im} \mathrm{ln} \mathrm{det} \mathcal{W}_2$",fontsize=20)
# pl.xticks([0,(len(out_data)-1)/2.0,len(out_data)-1],plot_xlabel,fontsize=20)
# # pl.yticks([-np.pi,0.0,np.pi],[r"$-\pi$",r"$0$", r"$\pi$"],fontsize=20)
# pl.grid(True)
# pl.title(model_str + r", outer",fontsize=20)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# fig.tight_layout()
# pl.savefig(path_data+"/out_data_summed.pdf")
# pl.savefig(path_data+"/out_data_summed.png")
# pl.close()

# end of the code
print("Done")

