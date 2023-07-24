# nested_and_spin_resolved_Wilson_loop

**Implemented by Kuan-Sen Lin, Benjamin J. Wieder, and Barry Bradlyn**

This package is an extension of the original [PythTB Package](https://www.physics.rutgers.edu/pythtb/), which was developed and is mantained by Sinisa Coh (University of California at Riverside), David Vanderbilt (Rutgers University) and [other team members](https://www.physics.rutgers.edu/pythtb/about.html#history). All credit for the essential implementation of PythTB goes to the original developers. 
For documentation for the original PythTB Package, please visit the [PythTB page](https://www.physics.rutgers.edu/pythtb/usage.html), as the interface is identical.

This extension contains the following modules:
- **nestedWilsonLib**: perform a nested Wilson loop calculation to facilitate the study of higher-order Wannier spectral evolution for a given tight-binding model constructed from [PythTB](https://www.physics.rutgers.edu/pythtb/) 
- **spin_resolved_analysis**: compute the spin-resolved spectra of bulk band and Wannier spectra for a given spinful tight-binding model with constructed from [PythTB](https://www.physics.rutgers.edu/pythtb/)

These modules are written within the framework of original [PythTB](https://www.physics.rutgers.edu/pythtb/). 

The analytical framework underlying our implementation of the nested Wilson loop is detailed in Refs. [1,3], and the underlying physical principles of spin-resolved topology are detailed in Ref. [3]. Our nested Wilson loop module (nestedWilsonLib) can be used to compute both the Wilson loop (Berry phase or Wannier charge center) spectrum and the nested Wilson loop spectrum for a given tight-binding model constructed from [PythTB](https://www.physics.rutgers.edu/pythtb/). Combining the two modules (nestedWilsonLib and spin_resolved_analysis), one can compute the spin-resolved Wilson loop and the nested spin-resolved Wilson loop spectrum for a given [PythTB](https://www.physics.rutgers.edu/pythtb/) tight-binding model with a spin degree of freedom, as detailed in Ref. [3].  

## Installation

Install by running

``` 
python setup.py install 
```

## Requirements

nested_and_spin_resolved_Wilson_loop requires the following packages:

- pythtb
- numpy
- scipy
- matplotlib

## Examples

We provide a set of python example codes in the "examples" directory for the following computations:

- Wilson loop spectrum of 1D Rice-Mele chain.

- spin-resolved Wilson loop spectrum of 2D topological insulator, and 2D fragile topological insulator (see Ref. [2]).

- nested Wilson loop spectrum of 2D quadrupole insulator, 3D inversion-protected or C2T-protected axion insulator, and 3D helical higher-order topological insulator (see Refs. [1,2,3]).

- nested spin-resolved Wilson loop spectrum of 3D helical higher-order topological insulator (see Ref. [3]).

The output figures and data are also included in each subdirectory of the "examples" directory.

The python example codes are run using python3 and PythTB version 1.7.2.

## Reference

Please cite the following papers when using this package:

[1] B. J. Wieder and B. A. Bernevig, The axion insulator as a pump of fragile topology, arXiv:1810.02373

[2] B. J. Wieder, Z. Wang, J. Cano, X. Dai, L. M. Schoop, B. Bradlyn, and B. A. Bernevig, Strong and fragile topological Dirac semimetals with higher-order Fermi arcs, Nat. Commun. 11, 627 (2020)

[3] K.-S. Lin, G. Palumbo, Z. Guo, J. Blackburn, D. P. Shoemaker, F. Mahmood, Z. Wang, G. A. Fiete, B. J. Wieder, and B. Bradlyn, Spin-resolved topology and partial axion angles in three-dimensional insulators, arXiv:2207.10099

## License

This project is released under the [GNU General Public License](https://github.com/kuansenlin/nested_and_spin_resolved_Wilson_loop/blob/main/LICENSE)
