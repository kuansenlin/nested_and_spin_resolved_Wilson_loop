import numpy as np
import numpy.linalg as lin
import copy
from pythtb import *
from pythtb import _wf_dpr
from pythtb import _one_phase_cont
from pythtb import _array_phases_cont
from pythtb import _offdiag_approximation_warning_and_stop
from scipy.linalg import schur

r"""

### Expected actual implementation (updated on 11/18/2022)

# Specify the model and grid numbers k1_grid*k2_grid*k3_grid 
# Use the original wf_array class of pythtb
array1=wf_array(my_model,[k1_grid,k2_grid,k3_grid])

# Specify the base point of energy eigenvectors, and get the energy eigenvectors over the grids
# Use the original wf_array class of pythtb
array1.solve_on_grid([0,0,0]) 

# Use the energy eigenvector array object, together with the index of the energy band over 
# which the Wannier band basis will be computed, and the Wilson loop direction, we form the 
# new object of class "wannier_band_basis_array"
# At this point, the Wannier band basis functions have not been computed, but several checks should already been
# done when we form the wannier_band_basis_array class.
# the dir1 is along "which entry of the array1" we will compute the Wilson loop
dir1=0
array2=wannier_band_basis_array(array1,[1,2,3],dir1) 

# Compute and return the Wannier band basis by directly calling solve_wannier_band_basis
array2.solve_wannier_band_basis()

# Specify the window_list=[window_1,window_2], will use convention that 
# if window_1<window_2 then we do inner nested Wilson loop, and 
# if window_1>window_2 then we do outer nested Wilson loop.
# Also specify the nested Wilson loop direction, contin, berry_evals
# wnum (optional) = the expected number of nested Wilson loop eigenphases
# the dir2 is along which entry of the array2 we will compute the nested Wilson loop
dir2=0
window_list=[-0.1*np.pi,0.1*np.pi]
in_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=1)
window_list=[0.1*np.pi,-0.1*np.pi]
out_data=array2.nested_berry_phase(window_list,dir2,contin=False,berry_evals=True,wnum=2)

"""

#Useful constants
I=1.*1j

class wannier_band_basis_array(object):

    def __init__(self,energy_wf_array,energy_band_ind_list,wilson_loop_dir):

        # energy_wf_array: wf_array class from pythtb after performing wf_array.solve_on_grid

        # energy_band_ind_list: a list of integers specifying the indices of the energy bands chosen to form the Wannier band basis

        # wilson_loop_dir: direction along which the Wilson loop will be computed
        #                  if wilson_loop_dir=0: the Wilson loop will be computed along the first  k-point entry of energy_wf_array
        #                  if wilson_loop_dir=1: the Wilson loop will be computed along the second k-point entry of energy_wf_array
        #                  if wilson_loop_dir=2: the Wilson loop will be computed along the third  k-point entry of energy_wf_array
        
        #                  e.g. if we have a 3D model with energy_wf_array._mesh_arr=[21,31,41], 
        #                       then wilson_loop_dir=0 means computing Wilson loop along the first direction with 21 grids
        
        #                  e.g. if we have a slab model cut from a 3D model with finite direction along the second lattice vector,
        #                       then even though we have energy_wf_array._mesh_arr=[21,41] and (energy_wf_array._model)._per=[0,2], 
        #                       the wilson_loop_dir can still be chosen only between 0 and 1, for instance in this case if 
        #                       wilson_loop_dir=1, it means that the Wilson loop will be computed along the k-point entry with 41 grids

        # store the energy_wf_array class 
        self._energy_wf_array=copy.deepcopy(energy_wf_array)

        # number of electronic states for each k-point
        self._nsta=((self._energy_wf_array)._model)._nsta
        # number of spin components
        self._nspin=((self._energy_wf_array)._model)._nspin # 0=spinless and 1=spinful
        # number of orbitals
        self._norb=((self._energy_wf_array)._model)._norb
        # store orbitals from the model in the ndarray form
        self._orb=np.array(copy.deepcopy(((self._energy_wf_array)._model)._orb))

        # store the list of periodic directions of the original model = (self._energy_wf_array)._model
        # e.g. for a slab model finite along the second lattice vector cutted from a 3D model,
        #      we have ((self._energy_wf_array)._model)._per=[0,2], which is by default a "list"
        self._per=copy.deepcopy(((self._energy_wf_array)._model)._per) 
        # self._per will be used to correctly impose the boundary conditions on the Wannier band basis
        
        # store the energy_band_ind_list
        self._energy_band_ind_list=copy.deepcopy(energy_band_ind_list)

        # if the original model is zero-dimensional, the Wannier band basis can not be computed
        # so the users are currently prevented from inputing a zero-dimensional finite model to form
        # the class wannier_band_basis_array
        if len(self._per)==0: 
            raise Exception("\n\nZero-dimensional model can not be used to compute Wannier band basis") 

        # store dimension of array of points of the "energy bands"
        self._mesh_arr_energy_band=(self._energy_wf_array)._mesh_arr # e.g. np.array([21,31,41]) for a 3D model, or np.array([21,41]) for a slab model cutted from 3D model
        self._dim_arr_energy_band=len(self._mesh_arr_energy_band) # e.g. np.array([21,31,41]) corresponds to 3 and np.array([21,41]) corresponds to 2
        assert len(self._per)==self._dim_arr_energy_band, "the number of periodic directions is not the same as the dimension of array of points of the energy bands"

        # store the wilson_loop_dir
        self._wilson_loop_dir=copy.deepcopy(wilson_loop_dir)

        # store dimension of array of points on which to keep the "Wannier band energy" and "Wannier band basis"
        # e.g. if self._mesh_arr_energy_band=[21,31,41] and wilson_loop_dir=0, we will have self._mesh_arr=np.array([31,41])
        #                                                                   1,                             np.array([21,41])
        #                                                                   2,                             np.array([21,31])
        # store also the remaining periodic directions of the model as "self._per_wannier_band".
        # e.g. if self._mesh_arr_energy_band=[21,31,41], self._per=[0,1,2], and wilson_loop_dir=0, we will have self._per_wannier_band=[1,2],
        #      which indicates that the first  entry of self._mesh_arr corresponds to the second lattice direction of the model
        #                               second                                            third  
        # e.g. if the model is a slab cutted from a 3D model finite along the second lattice vector, we have for instance
        #      self._mesh_arr_energy_band=[21,41], and self._per=[0,2].
        #      If we then perform the Wilson loop computation along the first direction (wilson_loop_dir=0) with 21 grids, 
        #      in self._mesh_arr_energy_band, the resulting Wannier band basis will need to satisfy the boundary condition using 
        #      the "third" reciprocal lattice vector, namely self._per_wannier_band=[2].
        if self._wilson_loop_dir==0:
            if len(self._per)==1: # one periodic direction for the original model
                self._mesh_arr=np.array([]) # empty array
                self._per_wannier_band=[] # empty list
            elif len(self._per)==2: # two periodic directions for the original model
                self._mesh_arr=np.array([self._mesh_arr_energy_band[1]])
                self._per_wannier_band=[self._per[1]]
            elif len(self._per)==3: # three periodic directions for the original model
                self._mesh_arr=np.array([self._mesh_arr_energy_band[1],self._mesh_arr_energy_band[2]])
                self._per_wannier_band=[self._per[1],self._per[2]]
        elif self._wilson_loop_dir==1:
            if len(self._per)==1: # one periodic direction for the original model
                raise Exception("\n\nWrong wilson_loop_dir")
            if len(self._per)==2: # two periodic directions for the original model
                self._mesh_arr=np.array([self._mesh_arr_energy_band[0]])
                self._per_wannier_band=[self._per[0]]
            elif len(self._per)==3: # three periodic directions for the original model
                self._mesh_arr=np.array([self._mesh_arr_energy_band[0],self._mesh_arr_energy_band[2]])
                self._per_wannier_band=[self._per[0],self._per[2]]
        elif self._wilson_loop_dir==2:
            if len(self._per)==1: # one periodic direction for the original model
                raise Exception("\n\nWrong wilson_loop_dir")
            if len(self._per)==2: # two periodic directions for the original model
                raise Exception("\n\nWrong wilson_loop_dir")
            elif len(self._per)==3: # three periodic directions for the original model
                self._mesh_arr=np.array([self._mesh_arr_energy_band[0],self._mesh_arr_energy_band[1]])
                self._per_wannier_band=[self._per[0],self._per[1]]
        else:
            raise Exception("\n\nWrong wilson_loop_dir")
        
        # store the d-1 where d is the number of periodic directions of the original model
        self._dim_arr=len(self._mesh_arr)

        # all dimensions of the k-points in the Wannier band basis array should be 2 or larger, because pbc can be used
        if True in (self._mesh_arr<=1).tolist():
            raise Exception("\n\nDimension of wannier_band_basis_array in each direction must be 2 or larger.")
        
        # generate temporary array used later to generate object ._wfs for "Wannier band basis" and ._wannier_band_energy for "Wannier band energy"
        wfs_dim=np.array(copy.deepcopy(self._mesh_arr)) #np.copy(self._mesh_arr)
        wfs_dim=np.append(wfs_dim,len(self._energy_band_ind_list))
        wfs_dim=wfs_dim.astype("int") # to take care of the case when self._mesh_arr is an empty array, where there is no grid in k-space for the Wannier bands
        self._wannier_band_energy=np.zeros(wfs_dim,dtype=float)
        wfs_dim=np.append(wfs_dim,self._norb)
        if self._nspin==2:
            wfs_dim=np.append(wfs_dim,self._nspin)
        # store the wannier band basis here in the form _wfs[kx_index,ky_index, ... ,band,orb,spin]
        wfs_dim=wfs_dim.astype("int")
        self._wfs=np.zeros(wfs_dim,dtype=complex)
    
    def solve_wannier_band_basis(self):

        r"""

        Solve and return the Wannier band basis array over the grid of self._mesh_arr.
        The Wannier band basis (self._wfs) |w_{n,{\bf k}}> will satisfy the boundary condition
        
        |w_{n,{\bf k}+{\bf G}}>=[V(G)]^{-1}|w_{n,{\bf k}}>,

        where [V(G)]_{ab} = e^{i*{\bf G} \cdot {\bf r}_{a}} where {\bf r}_{a} is the position vector of the 
        a^th orbital within the unit cell.
        The Wannier band energy (self._wannier_band_energy) will satisfy periodic boundary condition:

        {\gamma_1}_{n}({\bf k}) = {\gamma_1}_{n}({\bf k} + {\bf G}).

        """

        # if the original model has one periodic direction, self._energy_wf_array will have shape [kpnt,band,orb,spin] or [kpnt,band,orb]
        #                           two                                                           [kpnt1,kpnt2,band,orb,spin] or [kpnt1,kpnt2,band,orb]
        #                           three                                                         [kpnt1,kpnt2,kpnt3,band,orb,spin] or [kpnt1,kpnt2,kpnt3,band,orb]

        # check if model came from w90
        if ((self._energy_wf_array)._model)._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        
        if len(self._per)==1: # original model has one periodic direction

            # check the shape of the wannier_band_energy array to be filled
            assert len((self._wannier_band_energy).shape)==1, "Wrong shape of self._wannier_band_energy"

            # check the shape of the wannier band basis array (self._wfs) to be filled
            if self._nspin==1:
                assert len((self._wfs).shape)==2, "Wrong shape of self._wfs"
            elif self._nspin==2:
                assert len((self._wfs).shape)==3, "Wrong shape of self._wfs"

            # pick which energy wavefunctions to use
            wf_use=(self._energy_wf_array)._wfs[:,self._energy_band_ind_list,:]

            # calculate the Wilson loop matrix
            wilson_matrix=_one_berry_loop_wilson_matrix(wf_use,VG_mat((self._energy_wf_array)._model,self._per[0]))

            # compute the wannier band energy and wannier band basis
            (self._wannier_band_energy[:],self._wfs[:])=_wilson_eigs((self._energy_wf_array)._model,wilson_matrix,wnum=len(self._energy_band_ind_list))
        
        elif len(self._per)==2: # original model has two periodic directions

            assert len((self._wannier_band_energy).shape)==2, "Wrong shape of self._wannier_band_energy"

            if self._nspin==1:
                assert len((self._wfs).shape)==3, "Wrong shape of self._wfs"
            elif self._nspin==2:
                assert len((self._wfs).shape)==4, "Wrong shape of self._wfs"

            # choice along which direction you wish to calculate berry phase
            if self._wilson_loop_dir==0:

                for i in range(self._mesh_arr_energy_band[1]):

                    wf_use=(self._energy_wf_array)._wfs[:,i,:,:][:,self._energy_band_ind_list,:]
                    wilson_matrix=_one_berry_loop_wilson_matrix(wf_use,VG_mat((self._energy_wf_array)._model,self._per[0]))
                    (self._wannier_band_energy[i,:],self._wfs[i,:])=_wilson_eigs((self._energy_wf_array)._model,wilson_matrix,wnum=len(self._energy_band_ind_list))
                    
            elif self._wilson_loop_dir==1:

                for i in range(self._mesh_arr_energy_band[0]):

                    wf_use=(self._energy_wf_array)._wfs[i,:,:,:][:,self._energy_band_ind_list,:]
                    wilson_matrix=_one_berry_loop_wilson_matrix(wf_use,VG_mat((self._energy_wf_array)._model,self._per[1]))
                    (self._wannier_band_energy[i,:],self._wfs[i,:])=_wilson_eigs((self._energy_wf_array)._model,wilson_matrix,wnum=len(self._energy_band_ind_list))

            else:

                raise Exception("\n\nWrong direction for Berry phase calculation!")

            # impose boundary conditions on both self._wannier_band_energy and self._wfs
            self.impose_pbc_wannier_band_basis(0,self._per_wannier_band[0])

        elif len(self._per)==3: # original model has three periodic directions

            assert len((self._wannier_band_energy).shape)==3, "Wrong shape of self._wannier_band_energy"

            if self._nspin==1:
                assert len((self._wfs).shape)==4, "Wrong shape of self._wfs"
            elif self._nspin==2:
                assert len((self._wfs).shape)==5, "Wrong shape of self._wfs"

            # choice along which direction you wish to calculate berry phase
            if self._wilson_loop_dir==0:

                for i in range(self._mesh_arr_energy_band[1]):

                    for j in range(self._mesh_arr_energy_band[2]):

                        wf_use=(self._energy_wf_array)._wfs[:,i,j,:,:][:,self._energy_band_ind_list,:]
                        wilson_matrix=_one_berry_loop_wilson_matrix(wf_use,VG_mat((self._energy_wf_array)._model,self._per[0]))
                        (self._wannier_band_energy[i,j,:],self._wfs[i,j,:])=_wilson_eigs((self._energy_wf_array)._model,wilson_matrix,wnum=len(self._energy_band_ind_list))

            elif self._wilson_loop_dir==1:

                for i in range(self._mesh_arr_energy_band[0]):

                    for j in range(self._mesh_arr_energy_band[2]):

                        wf_use=(self._energy_wf_array)._wfs[i,:,j,:,:][:,self._energy_band_ind_list,:]
                        wilson_matrix=_one_berry_loop_wilson_matrix(wf_use,VG_mat((self._energy_wf_array)._model,self._per[1]))
                        (self._wannier_band_energy[i,j,:],self._wfs[i,j,:])=_wilson_eigs((self._energy_wf_array)._model,wilson_matrix,wnum=len(self._energy_band_ind_list))

            elif self._wilson_loop_dir==2:

                for i in range(self._mesh_arr_energy_band[0]):

                    for j in range(self._mesh_arr_energy_band[1]):

                        wf_use=(self._energy_wf_array)._wfs[i,j,:,:,:][:,self._energy_band_ind_list,:]
                        wilson_matrix=_one_berry_loop_wilson_matrix(wf_use,VG_mat((self._energy_wf_array)._model,self._per[2]))
                        (self._wannier_band_energy[i,j,:],self._wfs[i,j,:])=_wilson_eigs((self._energy_wf_array)._model,wilson_matrix,wnum=len(self._energy_band_ind_list))

            else:

                raise Exception("\n\nWrong direction for Berry phase calculation!")

            # impose boundary conditions on both self._wannier_band_energy and self._wfs
            self.impose_pbc_wannier_band_basis(0,self._per_wannier_band[0])
            self.impose_pbc_wannier_band_basis(1,self._per_wannier_band[1])

        else:

            raise Exception("\n\nWrong dimensionality!")

    def nested_berry_phase(self,window,nested_wilson_loop_dir=None,contin=True,berry_evals=False,wnum=None):

        r"""
        Solve and return the nested Wilson loop eigenphases -- the Berry phases computed over the Wannier bands.
        """

        # window: a list containing two float numbers = [w1,w2]
        #         if w1 < w2: compute the nested Wilson loop eigenphases over the Wannier bands within the range (w1,w2)
        #         if w1 > w2: compute the nested Wilson loop eigenphases over the Wannier bands outside the range of [w1,w2]

        # nested_wilson_loop_dir: direction along which the nested  Wilson loop will be computed.
        #                         If nested_wilson_loop_dir=0, the nested Wilson loop will be computed along the first  k-point entry of wannier_band_basis_array
        #                         If nested_wilson_loop_dir=1, the nested Wilson loop will be computed along the second k-point entry of wannier_band_basis_array
        #                         nested_wilson_loop_dir can be None for systems with two periodic directions for the original model, namely len(self._per)=2.
        #                         but nested_wilson_loop_dir needs to be either 0 or 1 for systems with two periodic directions for the original model, namely len(self._per)=3.
        # currently the nested Wilson loop calculation is only for systems whose original model has two or three periodic direction, namely len(self._per)=2 or 3.

        # e.g. if len(self._per)=3, and we choose both wilson_loop_dir=1 and nested_wilson_loop_dir=1, then the nested Wilson loop
        #      will be computed along G1 where {G1,G2,G3} are the three reciprocal lattice vectors.

        # (optional) wnum = expected number of Wannier bands fixed by the window variable 

        # berry_evals: True = obtain individual eigenphases, False = obtain the summation over all eigenphases mod 2pi within range [-pi,pi)

        # check if model came from w90
        if ((self._energy_wf_array)._model)._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        assert len(self._per)==len((self._wannier_band_energy).shape), "the number of periodic directions of the model is not consistent with the shape of Wannier band energy array"
        
        if len((self._wannier_band_energy).shape)==1: # model has only one periodic directions
            # this means that there will no more k-point index in the self._wannier_band_energy array, 
            # and thus we should not try to do nested Wilson loop

            raise Exception("\n\n1D system!")

        if len((self._wannier_band_energy).shape)==2: # model has two periodic directions
            
            # if the model has two periodic directions, namely len(((self._energy_wf_array)._model)._per)=2
            # there is no need for contin=True since we will just get a number 
            # (or a 1D ndarray with length = number of individual eigenphases of the nested Wilson loop matrix) 
            # for the nested Berry phase, therefore we automatically set contin to False

            contin=False

        if len((self._wannier_band_energy).shape)==2: # model has two periodic directions
            
            if (type(nested_wilson_loop_dir)==int) and (nested_wilson_loop_dir!=0): 
                # note that we allow for nested Wilson loop calculation that the second dir to be None
                # what is not allowed is that if the user input an integral nested_wilson_loop_dir but this nested_wilson_loop_dir is not equal to 0
                raise Exception("\n\nIncorrect nested_wilson_loop_dir for 2D system!")

        if len((self._wannier_band_energy).shape)==3: # model has three periodic directions
            
            if (nested_wilson_loop_dir==None) or (type(nested_wilson_loop_dir)==int and (nested_wilson_loop_dir<0 or nested_wilson_loop_dir>1)):
                # We do not allow nested_wilson_loop_dir=None for models with three periodic directions
                # We also do not allow, if nested_wilson_loop_dir is input as an integer, that nested_wilson_loop_dir is not equal to 0 or 1
                raise Exception("\n\nIncorrect nested_wilson_loop_dir for 3D system!")
        
        # generate the appropriate VG matrix
        if len((self._wannier_band_energy).shape)==2: # model has two periodic directions
            VG=VG_mat((self._energy_wf_array)._model,self._per_wannier_band[0])
        elif len((self._wannier_band_energy).shape)==3: # model has three periodic directions
            VG=VG_mat((self._energy_wf_array)._model,self._per_wannier_band[nested_wilson_loop_dir])

        # Begin the computation of the nested Wilson loop eigenphases
        if len((self._wannier_band_energy).shape)==2: # model has two periodic directions
            
            # nested_wilson_loop_dir is allowed to be None when len((self._wannier_band_energy).shape)==2

            # calculate nested berry phase
            ret=_one_nested_berry_loop_wannier_window(self._wannier_band_energy,self._wfs,window,VG,berry_evals,wnum)
            # this will just return a total phase, or an 1D ndarray of phases

        elif len((self._wannier_band_energy).shape)==3: # model has three periodic directions

            # choose along which direction you wish to calculate the nested berry phase
            if nested_wilson_loop_dir==0:
                ret=[]
                for i in range((self._wannier_band_energy).shape[1]):
                    ret.append(_one_nested_berry_loop_wannier_window(self._wannier_band_energy[:,i,:],self._wfs[:,i,:,:],window,VG,berry_evals,wnum))

            elif nested_wilson_loop_dir==1:
                ret=[]
                for i in range((self._wannier_band_energy).shape[0]):
                    ret.append(_one_nested_berry_loop_wannier_window(self._wannier_band_energy[i,:,:],self._wfs[i,:,:,:],window,VG,berry_evals,wnum))

            else:
                raise Exception("\n\nWrong direction for nested Berry phase calculation!")

        else:
            raise Exception("\n\nWrong dimensionality!")

        # for models with len(self._per)=2: if berry_evals = False, then we have the ret as just a number
        #                                                    True , then we have the ret as a 1D ndarray with shape [band]

        # for models with len(self._per)=3: if berry_evals = False, then we have the ret as a 1D ndarray with shape [kpnt]
        #                                                    True , then we have the ret as a 2D ndarray with shape [kpnt,band]
        
        # only if we are in the case of len(self._per)=3 will we require contin=True

        # convert phases to numpy array
        if len((self._wannier_band_energy).shape)>2 or berry_evals==True:
            ret=np.array(ret,dtype=float)

        # make phases of eigenvalues continuous
        if contin==True:
            # iron out 2pi jumps, make the gauge choice such that first phase in the
            # list is fixed, others are then made continuous.
            if berry_evals==False:
                
                # only need to deal with the model with three periodic directions
                if len((self._wannier_band_energy).shape)==3:
                    ret=_one_phase_cont(ret,ret[0])
                else: 
                    raise Exception("\n\nWrong dimensionality of the model for contin==True!")

            # make eigenvalues continuous. This does not take care of band-character
            # at band crossing for example it will just connect pairs that are closest
            # at neighboring points.
            else:
                
                # only need to deal with the model with three periodic directions
                if len((self._wannier_band_energy).shape)==3:
                    ret=_array_phases_cont(ret,ret[0,:])
                else: 
                    raise Exception("\n\nWrong dimensionality of the model for contin==True!")

        return ret

    # zero-dimensional finite size model has already been prevented in __init__
    def __check_key(self,key):

        # do some checks for models with one periodic direction, which
        # after obtaining the Wannier band basis will not have any kpnt index
        if self._dim_arr==0:
            if key != None:
                raise Exception("For models with one periodic direction, key must be specified as None")
        # do some checks for models with two periodic directions, which
        # after obtaining the Wannier band basis will have one kpnt index
        elif self._dim_arr==1:
            if type(key).__name__!='int':
                raise TypeError("Key should be an integer!")
            if key<(-1)*self._mesh_arr[0] or key>=self._mesh_arr[0]:
                raise IndexError("Key outside the range!")
        # do some checks for model with larger numbers of periodic directions
        else:
            if len(key)!=self._dim_arr:
                raise TypeError("Wrong dimensionality of key!")
            for i,k in enumerate(key):
                if type(k).__name__!='int':
                    raise TypeError("Key should be set of integers!")
                if k<(-1)*self._mesh_arr[i] or k>=self._mesh_arr[i]:
                    raise IndexError("Key outside the range!")

    def __getitem__(self,key):

        # For model with one periodic direction,
        # the Wannier band basis array takes the shape (self._wfs).shape=[band,orbit,spin] or [band,orbit,spin]
        # key = None will be targeted to this case, since we do not have any kpnt index to specify with the 
        # [] (indexer) operators. For such models with only one periodic direction, currently it will be strictly
        # require that the user input key=None, namely wannier_band_basis_array[key=None] to get its wave function with
        # shape [band,orbit,spin] or [band,orbit,spin]
        # __check_key will check that the users should input key=None when the model has only one periodic direction

        # For model with two periodic directions,
        # the Wannier band basis array takes the shape (self._wfs).shape=[kpnt,band,orbit,spin] or [kpnt,band,orbit,spin]
        # And we need to specify an integer for the key, which will be checked by __check_key

        # For model with three periodic directions,
        # the Wannier band basis array takes the shape (self._wfs).shape=[kpnt1,kpnt2,band,orbit,spin] or [kpnt1,kpnt2,band,orbit,spin]
        # And we need to specify two integers for the key, which will be checked by __check_key

        # check that key is in the correct range
        self.__check_key(key)

        # for models with one periodic direction
        if self._dim_arr==0:
            return self._wfs # return back the Wannier band basis array with shape [band,orbit,spin] or [band,orbit]
        else:
            return self._wfs[key] # return back the Wannier band basis array with shape [band,orbit,spin] or [band,orbit]

    def __setitem__(self,key,value):

        # value must have shape [band,orbital,spin] or [band,orbital,spin]
        if self._nspin==1:
            assert value.shape == (len(self._energy_band_ind_list),self._norb) , "Wrong shape of value!"
        elif self._nspin==2:
            assert value.shape == (len(self._energy_band_ind_list),self._norb,self._nspin) , "Wrong shape of value!"

        # check that key is in the correct range
        self.__check_key(key)

        # store wave function
        if self._dim_arr==0:
            self._wfs=np.array(value,dtype=complex)
        else:
            self._wfs[key]=np.array(value,dtype=complex)

    def impose_pbc_wannier_band_basis(self,mesh_dir,k_dir): 

        ## self.model: pythtb model of the system we are considering now

        ## self._wfs: array for wannier band basis functions
        ## will deal with only 2D (len(self._per)=2) and 3D (len(self._per)=3) systems
        ## when we have 2D system, we have self._wfs be [kpnt,band ind,orb,spin] or [kpnt,band ind,orb]
        ## when we have 3D system, we have self._wfs be [kpnt1,kpnt2,band ind,orb,spin] or [kpnt1,kpnt2,band ind,orb]
        
        ## mesh_dir: along which direction of the self._wfs array we would like to impose the boundary condition
        ## k_dir: which reciprocal lattice vector we would like to impose the boundary condition along
        
        ## For example, suppose we have a 3D system. And we choose mesh_dir = 0 and k_dir = 1,
        ## this means that we will impose the boundry condition along kpnt1 in self._wfs=[kpnt1,kpnt2,band ind,orb,(spin)]
        ## and the boundary condition for the alpha tight-binding basis function will be using exp(-i \vec{r}_alpha \cdot \vec{G}_2 )
        ## assuming {G1,G2,G3} is the set of the reciprocal lattice vectors for this 3D system.

        # make sure that the model has periodic directions more than 1
        if len(self._per) < 2:
            raise Exception("\n\nThe number of periodic directions of the model needs to be greater than 1 for impose_pbc_wannier_band_basis to work.")

        # make sure that the k_dir is in self._per
        if k_dir not in self._per:
            raise Exception("\n\nPeriodic boundary condition can be specified only along periodic directions!")

        # Compute phase factors
        ffac=np.exp(-2.j*np.pi*self._orb[:,k_dir])
        if self._nspin==1:
            phase=ffac # a 1D array
        else:
            # for spinors, same phase multiplies both components
            phase=np.zeros((self._norb,2),dtype=complex) # a 2D array
            phase[:,0]=ffac 
            phase[:,1]=ffac

        # Copy first eigenvector onto last one, multiplying by phase factors
        # We can use numpy broadcasting since the orbital index is last
        # For the consideration of nested Wilson loops, up to 3D cases, we will only need to
        # consider mesh_dir up to 1.
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]*phase
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]*phase
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

        ## additionaly, we also impose an "exact periodic" boundary condition on the Wannier band energy
        if mesh_dir==0:
            self._wannier_band_energy[-1,:]=self._wannier_band_energy[0,:]
        elif mesh_dir==1:
            self._wannier_band_energy[:,-1,:]=self._wannier_band_energy[:,0,:]

#=======================================================================
# Begin internal definitions
#=======================================================================

def _prettify(nparray):
    mask=1-np.isclose(nparray,0).astype(float)
    return nparray*mask

def VG_mat(model,G_dir=None):

    # model: pythtb model
    # G_dir: a direction that is integer and contained in model._per
    # e.g. for 3D (len(self._per)=3) system G_dir should be within [0,1,2]
    # e.g. for a slab cutted from a 3D system finite along the second lattice vector, G_dir should be within [0,2]
    # for 1D system the user can sepcify "None" for G_dir

    # check the dimension of the model
    if len(model._per)<1:
        raise Exception("\n\nThere exist no [V(G)] matrix for models without periodicity") 

    # check that G_dir is within model._per for models with len(model._per)>1
    if (len(model._per)==1):
        # if the user does specify a G_dir for a model with len(model._per)=1, then G_dir must be an integer
        if (G_dir != None) and (type(G_dir) != int):
            raise Exception("\n\nWrong G_dir")
        # if the user does specify an integer G_dir for a model with len(model._per)=1, then
        # we will need to make sure that G_dir is in model._per
        if (G_dir not in model._per) and (G_dir != None) and (type(G_dir) == int):
            raise Exception("\n\nWrong G_dir")
    elif len(model._per)>1:
        # for a model with len(model._per)>1, we require the user to specify an integer G_dir
        if G_dir == None:
            raise Exception("\n\nWrong G_dir")
        elif type(G_dir) != int:
            raise Exception("\n\nWrong G_dir")
        if G_dir not in model._per:
            raise Exception("\n\nWrong G_dir")

    # First get the diagonal elements, and then from it form the diagonal [V(G)] matrix, and we 
    # will will take care of the spins of the model
    if len(model._per)==1: 
        # for 1D we just take the model._per[0]th component of the orbital position vector (in reduced coordinate)
        VG_diag=np.array([np.exp(+2j*np.pi*model._orb[n][model._per[0]]) for n in range(len(model._orb))])
    elif len(model._per)==2:
        VG_diag=np.array([np.exp(+2j*np.pi*model._orb[n][G_dir]) for n in range(len(model._orb))])
    elif len(model._per)==3:
        VG_diag=np.array([np.exp(+2j*np.pi*model._orb[n][G_dir]) for n in range(len(model._orb))])

    # spinless or spinful
    if model._nspin==1:
        VG=np.diag(VG_diag)
    elif model._nspin==2:
        VG=np.kron(np.diag(VG_diag),np.eye(2,dtype=complex))
    
    return VG

def _one_berry_loop_wilson_matrix(wf,VG): 

    # wf: [kpnt,band,orbital,spin] or [kpnt,band,orbital] and kpnt has to be one dimensional
    # VG: the [V(\mathbf{G})] matrix with elements [V(\mathbf{G})]_{ab}=e^{i \mathbf{G} \cdot \mathbf{r}_a} \delta_{ab} 
    #     where \mathbf{G} is the direction in k-space along which the Wilson loop is performed
    # return: a Wilson loop matrix with shape (len(wf[0,0,:].flatten()),len(wf[0,0,:].flatten()))

    # number of occupied states
    nocc=wf.shape[1] # this holds for either spinless or spinful model, since we are taking out the number of band index

    ## form the energy eigenstate projectors, and check the boundary conditions of them
    Vk=np.array([wf[0,j,:].flatten() for j in range(nocc)]).T
    assert Vk.shape==(len(wf[0,0,:].flatten()),nocc)
    VkG=np.array([wf[-1,j,:].flatten() for j in range(nocc)]).T
    assert VkG.shape==(len(wf[-1,0,:].flatten()),nocc)
    Pk=np.matmul(Vk,(Vk.conjugate()).T)
    PkG=np.matmul(VkG,(VkG.conjugate()).T)
    assert np.max(np.abs(Pk-np.matmul(Pk,Pk)))<1.0E-8, "the energy eigenstate projector [P(k)] is not idempotent!"
    assert np.max(np.abs(PkG-np.matmul(PkG,PkG)))<1.0E-8, "the energy eigenstate projector [P(k+G)] is not idempotent!"
    assert np.max(np.abs(PkG-np.matmul(np.matmul((VG.conjugate()).T,Pk),VG)))<1.0E-8, "the energy eigenstate projector [P(k)] does not satisfy [P(k+G)]=[V(G)]^\dagger [P(k)] [V(G)]!"

    # temporary matrices
    proj=np.identity(len(wf[0,0,:].flatten()),dtype=complex) # the flatten is used in case we have a spinful model with nspin=2

    # go over all k-points
    # wf.shape[0] is the number of k points along the loop
    for i in range(wf.shape[0]): #range(wf.shape[0]-1):

        # update the projector by doing projector product in terms of P(k1)P(k2)...P(kN) where kN is the last k point
        V=np.array([wf[i,j,:].flatten() for j in range(nocc)])
        V=V.T
        proj=np.matmul(proj,np.matmul(V,(V.conjugate()).T))

    # multiply the result of projector products by appropriate (VG matrix)^dagger from the "right"
    proj=np.matmul(proj,(VG.conjugate()).T)

    # sandwich by the base point wave functions
    V0=np.array([wf[0,j,:].flatten() for j in range(nocc)])
    V0=V0.T
    W=np.matmul( np.matmul( (V0.conjugate()).T , proj ) , V0 )

    # do SVD
    matU,sing,matV=np.linalg.svd(W)
    W=np.matmul(matU,matV)

    # change back to the orbital basis
    wilson_matrix=np.sum([np.outer(V0[:,j],np.conj(V0[:,i]))*W[j,i] for i in range(nocc) for j in range(nocc)],axis=0)

    return wilson_matrix

def _wilson_eigs(model,wham,wnum=None): 

    ## model = pythtb model
    ## wham = Wilson loop matrix with shape (model._nsta,model._nsta)
    ## (optional) wnum = expected number of Wannier bands, default is None

    # check if the wham is normal (A A^dag = A^dag A)
    # in principle, wham can be decomposed into \sum_{j=1}^{nocc} e^{i*\gamma_j} |j> <j|
    # |j> is a _nsta-component vector since a pre-svd decomposition has been performed
    if np.max(np.abs(np.matmul(wham,wham.T.conjugate())-np.matmul(wham.T.conjugate(),wham)))>1.0E-8:
        raise Exception("\n\nWilson loop matrix should be normal!")
    
    # do schur decomposition
    matT,evecs=schur(wham)
    
    # check that matT is diagonal
    if np.max(np.abs(np.diag(np.diag(matT))-matT))>1.0E-8:
        raise Exception("\n\nSchur decomposition does not yield diagonal matrix T!")
    
    # get evals from the diagonal elements of matT
    evals=np.diag(matT) # evals is a 1D array
    
    # check that evecs is unitary
    if np.max(np.abs(np.matmul(evecs,evecs.T.conjugate())-np.eye(len(evecs[:,0]),dtype=complex)))>1.0E-8:
        raise Exception("\n\nSchur decomposition does not yield unitary matrix evecs!")
    if np.max(np.abs(np.matmul(evecs.T.conjugate(),evecs)-np.eye(len(evecs[:,0]),dtype=complex)))>1.0E-8:
        raise Exception("\n\nSchur decomposition does not yield unitary matrix evecs!")
    
    # recall that wham can be decomposed into \sum_{j=1}^{nocc} e^{i*\gamma_j} |j> <j|
    # get the indices of nonzero eigenvalues in evals
    # recall that there will be zero eigenvalues in the Wilson Hamiltonian which correspond to
    # the unoccupied states, and we would like to drop them
    ind=np.where(_prettify(evals)!=0)[0]

    # check that the nonzero eigenvalues are close enough to be unimodular
    if not np.all(np.isclose(np.abs(evals[ind]),1.0)):
        raise Exception("\n\nThe eigenvalues of the wilson loop matrix is not unimodular")

    # get the phases of the nonzero eigenvalues and the corresponding eigenvectors of wham
    (tempvals,tempvecs)=((-1.0)*np.angle(evals[ind]),evecs[:,ind]) # the tempvecs will in general not be a square matrix 
    # note that we have put (-1.0)* in front of the np.angle 

    # check that the number of tempvals is expected if we specify the wnum
    # check also that the shape of tempvecs is expected if we specify the wnum
    if (wnum!=None) and (type(wnum)==int) and (wnum>0):
        if len(tempvals)!=wnum:
            raise Exception("\n\nSchur decomposition does not yield the expected number of Wannier bands!")
        if tempvecs.shape!=(model._nsta,wnum):
            raise Exception("\n\nSchur decomposition does not yield the expected shape of Wannier band basis!")

    # return the sorted phases and the eigenvectors
    # we would like to return the eigenphase array as a 1D array as [band]
    # we would like to return the eigenvector array as either [band,orbital,spin] or [band,orbital] depending on whether the model is spinful
    q=tempvals.argsort()
    if model._nspin==1:
        return (tempvals[q],tempvecs[:,q].T) 
    elif model._nspin==2:
        return (tempvals[q],tempvecs[:,q].T.reshape(len(q),model._norb,model._nspin)) 

def _reshape_wf_1D_array(model,wf_1D_array): 

    # model: pythtb model 
    # wf_1D_array: a 1D array 
    
    # make sure that wf_1D_array is an 1D array
    assert len(wf_1D_array.shape)==1, "wf_1D_array is not an 1D array"

    # make sure that wf_1D_array is compatible with the model
    assert len(wf_1D_array)==model._nsta, "the wave function is not compatible with the model"

    # reshape the wf_1D_array if the model has model._nspin==2, otherwise just return back the wf_1D_array
    if model._nspin==1:
        return wf_1D_array
    elif model._nspin==2:
        return wf_1D_array.reshape(model._norb,model._nspin)

def _one_nested_berry_loop_wannier_window(wevals_use,wevecs_use,window,VG,berry_evals=False,wnum=None):

    ## wevals_use: [kpnt,band] and kpnt has to be one dimensional
    ## wevecs_use: [kpnt,band,orbital,spin] or [kpnt,band,orbital] and kpnt has to be one dimensional
    ## window: a list of form [window_1,window_2].
    ##         if window_1 < window_2: choose the bands within window_1~window_2
    ##         if window_1 > window_2: choose the bands outside window_2~window_1
    ## VG: the [V(\mathbf{G})] matrix with elements [V(\mathbf{G})]_{ab}=e^{i \mathbf{G} \cdot \mathbf{r}_a} \delta_{ab} 
    ##     where \mathbf{G} is the direction in k-space along which the Wilson loop is performed
    ## berry_evals: True = obtain individual eigenphases, False = obtain the summation over all eigenphases mod 2pi
    ## wnum: the number of Wannier bands we would like to use to perform the nested Berry phase calculation
    ##       it is not necessary that the user input a wnum, but it is good to input the expected wnum, so that
    ##       the function will help double check

    # pre-calculation to get the occ_list based on wevals_use, w1, w2, and inoutflag
    # this occ_list will serve as the tool for later construction of projector such that we select the
    # correct vectors to form the projector

    if len(wevals_use.shape)!=2:
        raise Exception("\n\nwevals_use shape is wrong")
    
    if (len(wevecs_use.shape)!=3) and (len(wevecs_use.shape)!=4):
        raise Exception("\n\nwevecs_use shape is wrong")

    if len(window)!=2:
        raise Exception("\n\nwindow should only contain two elements")
    
    # a list to store the Wannier band index within the range we would like to consider
    occ_list=[]
    for w in wevals_use:
        if window[0]<window[1]: # will select inner Wannier states
            occ_list.append([x for x in range(len(w)) if (window[0]<w[x] and w[x]<window[1])])
        elif window[0]>window[1]: # will select outer Wannier states
            occ_list.append([x for x in range(len(w)) if (w[x]<window[1] or window[0]<w[x])])
        else:
            raise Exception("\n\nThe two elements in the window list can not be identical")
        
    # check that all the lists in occ_list have the same length
    if not all(len(x) == len(occ_list[0]) for x in occ_list):
        raise Exception("\n\nThe Wannier window is wrong! (not separating inner from outer)")

    # if the user indeed inputs wnum, then we can check whether what we obtain for occ_list is consistent
    # with the input wnum
    if (wnum!=None) and (type(wnum)==int) and (wnum>0):
        if len(occ_list[0])!=wnum:
            raise Exception("\n\nThe Wannier window is wrong! (the number of Wannier bands does not match with the input wnum)")
    
    ## form the Wannier band basis projectors, and check the boundary conditions of them
    Vk_W=np.array([wevecs_use[0,j,:].flatten() for j in occ_list[0]]).T
    assert Vk_W.shape==(len(wevecs_use[0,0,:].flatten()),len(occ_list[0]))
    VkG_W=np.array([wevecs_use[-1,j,:].flatten() for j in occ_list[-1]]).T
    assert VkG_W.shape==(len(wevecs_use[-1,0,:].flatten()),len(occ_list[-1]))
    Pk_W=np.matmul(Vk_W,(Vk_W.conjugate()).T)
    PkG_W=np.matmul(VkG_W,(VkG_W.conjugate()).T)
    assert np.max(np.abs(Pk_W-np.matmul(Pk_W,Pk_W)))<1.0E-8, "the Wannier band basis projector [P_W(k)] is not idempotent!"
    assert np.max(np.abs(PkG_W-np.matmul(PkG_W,PkG_W)))<1.0E-8, "the Wannier band basis projector [P_W(k+G)] is not idempotent!"
    assert np.max(np.abs(PkG_W-np.matmul(np.matmul((VG.conjugate()).T,Pk_W),VG)))<1.0E-8, "the Wannier band basis projector [P_W(k)] does not satisfy [P_W(k+G)]=[V(G)]^\dagger [P_W(k)] [V(G)]!"

    # temporary matrices
    proj=np.identity(len(wevecs_use[0,0,:].flatten()),dtype=complex) # this will work for either spinless (nspin=1) or spinful (nspin=2) model

    # go over all k-points
    # wevecs_use.shape[0] is the number of k points along the loop
    # construct the projector and then do projector product: P_W(k1)P_W(k2)...P_W(kN) where kN is the last point
    for i in range(wevecs_use.shape[0]): #range(wevecs_use.shape[0]-1):

        # update the projector by doing projector product
        V=np.array([wevecs_use[i,j,:].flatten() for j in occ_list[i]])
        V=V.T
        proj=np.matmul(proj,np.matmul(V,(V.conjugate()).T))

    # multiply the result of projector products by appropriate (VG matrix)^dagger from the "right"
    proj=np.matmul(proj,(VG.conjugate()).T)

    # sandwich by the base point wave functions
    V0=np.array([wevecs_use[0,j,:].flatten() for j in occ_list[0]])
    V0=V0.T
    W=np.matmul( np.matmul( (V0.conjugate()).T , proj ) , V0 )

    # do SVD
    matU,sing,matV=np.linalg.svd(W)
    W=np.matmul(matU,matV)

    # calculate Berry phase
    if berry_evals==False:
        det=np.linalg.det(W)
        pha=(-1.0)*np.angle(det)
        return pha # this will return a number
    # calculate phases of all eigenvalues
    else:
        evals=np.linalg.eigvals(W)
        eval_pha=(-1.0)*np.angle(evals)
        # sort these numbers as well
        eval_pha=np.sort(eval_pha)
        return eval_pha # this will in general return a 1D array

