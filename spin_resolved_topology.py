# Functions that can be used to perform spin-resolved topology analysis

# Authors: Kuan-Sen Lin (UIUC Physics), Benjamin J. Wieder (University of Paris-Saclay), and Barry Bradlyn (UIUC Physics)

import numpy as np
import numpy.linalg as lin
from pythtb import *

# function to get the spin Sx, Sy, and Sz operator given a spinful pythtb tight-binding model
# users need to input the following information
# model: a pythtb model, the function will check if the model has nspin = 2
# the output are Sx, Sy, and Sz, which are all matrices with shape (Nt,Nt) where Nt = the number of tight-binding basis functions per unit cell
def get_Sx_Sy_Sz(model):

    assert model._nspin==2 , "this function is for model with nspin = 2"

    Nt=model._nsta # total number of tight-binding basis functions per unit cell

    # get the Sx, Sy, and Sz matrix
    Sx=np.kron(np.identity(Nt//2,dtype=complex),np.array([[0.0,1.0],[1.0,0.0]],dtype=complex))
    Sy=np.kron(np.identity(Nt//2,dtype=complex),np.array([[0.0,-1j],[+1j,0.0]],dtype=complex))
    Sz=np.kron(np.identity(Nt//2,dtype=complex),np.array([[1.0,0.0],[0.0,-1.0]],dtype=complex))

    # check the shape of the matrices
    assert Sx.shape==Sy.shape
    assert Sy.shape==Sz.shape
    assert Sz.shape==Sx.shape
    assert Sx.shape==(Nt,Nt)
    assert Sy.shape==(Nt,Nt)
    assert Sz.shape==(Nt,Nt)

    # check the (anti-)commutation relations
    assert np.all(np.isclose(Sx @ Sy + Sy @ Sx , np.zeros((Nt,Nt),dtype=complex)))
    assert np.all(np.isclose(Sy @ Sz + Sz @ Sy , np.zeros((Nt,Nt),dtype=complex)))
    assert np.all(np.isclose(Sz @ Sx + Sx @ Sz , np.zeros((Nt,Nt),dtype=complex)))
    assert np.all(np.isclose(Sx @ Sx , np.eye(Nt,dtype=complex)))
    assert np.all(np.isclose(Sy @ Sy , np.eye(Nt,dtype=complex)))
    assert np.all(np.isclose(Sz @ Sz , np.eye(Nt,dtype=complex)))
    assert np.all(np.isclose(Sx @ Sy - Sy @ Sx , 2.0*(+1j)*Sz))
    assert np.all(np.isclose(Sy @ Sz - Sz @ Sy , 2.0*(+1j)*Sx))
    assert np.all(np.isclose(Sz @ Sx - Sx @ Sz , 2.0*(+1j)*Sy))

    return Sx, Sy, Sz


# function to obtain the eigenvalues of the projected spin operator at a given k point
# users need to input the following information
# model: a pythtb model
# spin_dir: a list, or an 1D array, of three numbers specifying the direction of the spin, this spin_dir needs not to be a unit vector
# occ: a list, or an 1D array, specifying the indices of the occupied energy bands (the first band has index 0)
# k: a list, or an 1D array, specifying a bulk k point compatible with the number of periodic directions of the model
# the output is an 1D array with len(occ) elements
def get_PsP_evals(model,spin_dir,occ,k):

    assert model._nspin==2 , "this function is for model with nspin = 2"

    Sx, Sy, Sz = get_Sx_Sy_Sz(model) # get the spin matrices
    
    Nt=model._nsta # total number of tight-binding basis functions per unit cell

    occ_ind=len(occ) # total number of occupied bands

    # diagonalize the Bloch Hamiltonian
    (evals,evecs)=model.solve_one(k,eig_vectors=True)

    # reshape the evecs into a square matrix
    evecs=evecs.reshape((Nt,Nt))

    w=evecs[occ,:] # occupied energy eigenvectors
    assert w.shape==(occ_ind,Nt)

    # get the spin matrix along the spin_dir
    Srr=np.tensordot(spin_dir,[Sx,Sy,Sz],1)/lin.norm(spin_dir)

    sred=np.conjugate(w) @ Srr @ w.T # reduced spin matrix with shape (occ_ind,occ_ind)
    assert sred.shape==(occ_ind,occ_ind)
    # check Hermiticity
    assert np.all(np.isclose(sred,(sred.conjugate()).T))
    if np.max(np.abs(sred-(sred.conjugate()).T))>(1.0*(10)**(-9)):
        raise Exception("reduced spin matrix is not hermitian?!")
    
    # diagonalize sred
    sred_evals = lin.eigvalsh(sred)
    assert np.all(np.isclose(sred_evals,np.real(sred_evals)))
    sred_evals = np.real(sred_evals)
    idx = sred_evals.argsort()
    sred_evals = sred_evals[idx]
    assert sred_evals.shape==(occ_ind,)

    return sred_evals


# function to obtain the projected spin operator eigenstates at a given k point
# users need to input the following information
# model: a pythtb model
# spin_dir: a list, or an 1D array, of three numbers specifying the direction of the spin, this spin_dir needs not to be a unit vector
# occ: a list, or an 1D array, specifying the indices of the occupied energy bands (the first band has index 0)
# k: a list, or an 1D array, specifying a bulk k point compatible with the number of periodic directions of the model
# the output is an array with shape (Nt,Nt//2,2) where Nt = the number of tight-binding basis function per unit cell of the model,
# and in particular, the elements of the output with entries [len(occ)//2:Nt-len(occ)//2,:] are zero since they correspond to states 
# in the image of 1 - P where P is the projector onto the occupied space and will not be used in any calculations.
# Also, the elements of the output with entries [np.arange(occ_ind//2),:] corrresond to the PsP eigenvectors in the lower spin bands.
# Similarly, the elements of the output with entries [np.arange(Nt-occ_ind//2,Nt),:] corrresond to the PsP eigenvectors in the upper spin bands.
def get_PsP_evecs(model,spin_dir,occ,k):

    assert model._nspin==2 , "this function is for model with nspin = 2"

    Sx, Sy, Sz = get_Sx_Sy_Sz(model) # get the spin matrices
    
    Nt=model._nsta # total number of tight-binding basis functions per unit cell

    occ_ind=len(occ) # total number of occupied bands

    # diagonalize the Bloch Hamiltonian
    (evals,evecs)=model.solve_one(k,eig_vectors=True)

    # reshape the evecs into a square matrix
    evecs=evecs.reshape((Nt,Nt))

    w=evecs[occ,:] # occupied energy eigenvectors
    assert w.shape==(occ_ind,Nt)

    # get the spin matrix along the spin_dir
    Srr=np.tensordot(spin_dir,[Sx,Sy,Sz],1)/lin.norm(spin_dir)

    sred=np.conjugate(w) @ Srr @ w.T # reduced spin matrix with shape (occ_ind,occ_ind)
    assert sred.shape==(occ_ind,occ_ind)
    # check Hermiticity
    assert np.all(np.isclose(sred,(sred.conjugate()).T))
    if np.max(np.abs(sred-(sred.conjugate()).T))>(1.0*(10)**(-9)):
        raise Exception("reduced spin matrix is not hermitian?!")
    
    # diagonalize sred
    sred_evals, sred_evecs = lin.eigh(sred)
    assert np.all(np.isclose(sred_evals,np.real(sred_evals)))
    sred_evals = np.real(sred_evals)
    idx = sred_evals.argsort()
    sred_evals = sred_evals[idx]
    sred_evecs = sred_evecs[:,idx]
    assert sred_evals.shape==(occ_ind,) # check the shape of sred_evals
    assert sred_evecs.shape==(occ_ind,occ_ind) # check the shape of sred_evecs
    
    # check unitarity
    assert np.all(np.isclose( sred_evecs @ (sred_evecs.conjugate()).T , np.identity(occ_ind,dtype=complex) ))
    assert np.all(np.isclose( (sred_evecs.conjugate()).T @ sred_evecs , np.identity(occ_ind,dtype=complex) ))
    if np.max(np.abs( sred_evecs @ (sred_evecs.conjugate()).T - np.identity(occ_ind,dtype=complex) ))>(1.0*(10)**(-9)):
        raise Exception("sred_evecs is not unitary?!")
    if np.max(np.abs( (sred_evecs.conjugate()).T @ sred_evecs - np.identity(occ_ind,dtype=complex) ))>(1.0*(10)**(-9)):
        raise Exception("sred_evecs is not unitary?!")
    
    # # check the number of positive and negative reduced spin eigenvalues 
    # # (comment this part out if the spin spectrum is not expected to have an effective chiral symmetry)
    # assert len([x for x in sred_evals if x < 0])==occ_ind//2 , "Please comment this line out, as the model does not have an effective chiral symmetry in its PsP spectrum."
    # assert len([x for x in sred_evals if x > 0])==occ_ind//2 , "Please comment this line out, as the model does not have an effective chiral symmetry in its PsP spectrum."
    
    # now change to orbital basis
    assert w.shape==(occ_ind,Nt)
    assert sred_evecs.shape==(occ_ind,occ_ind)
    sred_evecs_orbital=w.T @ sred_evecs
    assert sred_evecs_orbital.shape==(Nt,occ_ind)
    # take transpose of sred_evecs_orbital
    sred_evecs_orbital=sred_evecs_orbital.T
    assert sred_evecs_orbital.shape==(occ_ind,Nt)
    
    # create an empty matrix that will be returned
    # need to preserve the form of wave function array in pythtb when nspin = 2
    PsP_evecs=np.zeros((Nt,Nt//2,2),dtype=complex)
    PsP_evecs[np.arange(occ_ind//2),:]=sred_evecs_orbital[np.arange(occ_ind//2),:].reshape(occ_ind//2,Nt//2,2)
    PsP_evecs[np.arange(Nt-occ_ind//2,Nt),:]=sred_evecs_orbital[np.arange(occ_ind-occ_ind//2,occ_ind),:].reshape(occ_ind//2,Nt//2,2)
    
    return PsP_evecs

