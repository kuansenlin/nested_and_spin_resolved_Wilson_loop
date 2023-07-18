#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nested_and_spin_resolved_Wilson_loop", # This is the name of the package
    version="1.0", # The initial release version
    author="Kuan-Sen Lin, Benjamin J. Wieder, and Barry Bradlyn", # Full name of the author    
    author_email="kuansen2@illinois.edu, benjamin.wieder@ipht.fr, and bbradlyn@illinois.edu", # Email of the author at the time of first release
    keywords="tight binding, Wilson loop, nested Wilson loop, spin-resolved topology",
    description="Computation of nested Wilson loop and the analysis of spin-resolved topology for tight binding models in the framework of PythTB",
    long_description=long_description, # Long description read from the the readme file
    long_description_content_type="text/markdown",
    py_modules=["nestedWilsonLib_v4","spin_resolved_analysis"], # Name of the python package
    package_dir={'':'src'}, # Directory of the source code of the package
    install_requires=["pythtb","numpy","scipy"] # Install other dependencies if any
)

