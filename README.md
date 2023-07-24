# nested_and_spin_resolved_Wilson_loop

This package is an extension of the original [PythTB](https://www.physics.rutgers.edu/pythtb/), developed and mantained by Sinisa Coh (University of California at Riverside), David Vanderbilt (Rutgers University) and [others](https://www.physics.rutgers.edu/pythtb/about.html#history). All credit of the essential implementation goes to the original developers. 
For documentation, please visit the [PythTB page](https://www.physics.rutgers.edu/pythtb/usage.html), as the interface is identical.

This extension contains the following modules:
- **nestedWilsonLib**: perform the nested Wilson loop calculation which can facilitate the study of, for instance, higher-order topology, for a given tight-binding model constructed from [PythTB](https://www.physics.rutgers.edu/pythtb/) 
- **spin_resolved_analysis**: perform the spin-resolved analysis and can be used to discover new spin-resolved topology for a given tight-binding model with spins constructed from [PythTB](https://www.physics.rutgers.edu/pythtb/) 
These modules are within the framework of original [PythTB](https://www.physics.rutgers.edu/pythtb/). 

The physics of spin-resolved topology and the numerical scheme of nested Wilson loop calculations are detailed in Ref. [3]. Our nested Wilson loop module (nestedWilsonLib) can be used to compute both the Wilson loop (Berry phase, or Wannier charge center) and the nested Wilson loop spectrum for a given tight-binding model constructed from [PythTB](https://www.physics.rutgers.edu/pythtb/). Combining the two modules (nestedWilsonLib and spin_resolved_analysis) one can compute the spin-resolved Wilson loop and the nested spin-resolved Wilson loop spectrum for a given [PythTB](https://www.physics.rutgers.edu/pythtb/) tight-binding model with a spin degree of freedom.  

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
