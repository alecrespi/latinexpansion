# latinexpansion: Expanded Latin Hypercube Sampling

## Overview

This repository contains the `latinexpansion` C module for Python, developed as part of a Bachelor's thesis at the University of Milano-Bicocca, Bachelor's degree in Computer Science. This is an early access release.

### Abstract

Stochastic methods are crucial for managing computationally demanding simulations because they effectively handle complex systems with many involved variables, offering high scalability where deterministic methods fall short. On the other hand, determinis- tic methods cut result variance. Latin Hypercube Sampling (LHS), a quasi-Monte Carlo technique, combines the benefits of both stochastic and deterministic approaches. It draws random samples from a model’s parameter space within predetermined specific intervals, or strata, a property known as stratification. This research aims to horizontally scale an existing LHS distribution while maintaining stratification as much as possible, a challeng- ing task. The proposed technique, named expanded ‘Latin Hypercube Sampling‘ (eLHS), enhances the flexibility of LHS-based simulations in the later stages of an evolutionary simulation process. Evolutionary apparatus are involved when a simulation’s surrogate model is insufficient for experiments, requiring the simulation to continue. The eLHS algo- rithm, available in the authors’ Python package ‘latinexpansion‘, allows users to expand a hyperparameter set and provides control over other desired properties.

## Authors

- **Crespi Alessandro** (alessandro.crespi.2002@gmail.com)
- **Prof. Davide Gerosa** (davide.gerosa@unimib.it)
- **Dr. Davide Gerosa** (m.boschini1@campus.unimib.it)

## Installation and compiling

To compile and install the module on macOS and Linux distributions, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/alecrespi/latinexpansion.git
   cd latinexpansion

2. **Compile and Install C module for Python Compiler**

   You need to install Python build module, then run:

   ```bash
   python -m build
   pip install ./dist/latinexpansion-1.0.0-cp311-cp311-macosx_11_0_arm64.whl
   

3. **Import**

   In python script, import and usage:

   ```python
   from latinexpansion import degree, eLHS


## Usage example

* **Degree of a sample set**

   ```python
   from scipy.stats.qmc import LatinHypercube, Sobol
   # set hyperparameters
   N, M, P = 100, 77, 2
   # create a LHS
   ss = LatinHypercube(P).random(N)
   # LHS has always maximum degree
   print( "✅" if degree(ss) == 1.0 else "❌")  # prints always ✅
   # compute degree with different binning grid of ss
   print( degree(ss, M) )  # may vary 0 (not included) to 1 (if LHS)
   # also, Sobol' is not native LHS
   sob = Sobol(P).random(N)
   print( "✅" if degree(sob) == 1.0 else "❌")  # probably ❌


* **Expand a Latin Hypercube**

   ```python
   from scipy.stats.qmc import LatinHypercube
   from numpy import concatenate
   from matplotlib import pyplot as plt
   ## set hyperparameters
   # N = # of intial samples; M = # new samples to add; P = # dimensions
   N, M, P = 100, 77, 2
   # create a LHS
   ss = LatinHypercube(P).random(N)
   # create expansion set
   expansion = eLHS(ss, M)
   # concatenate initial set and expansion set and get full expanded set
   expanded_set = concatenate( (ss, expansion) )
   # plot sets
   plt.plot(ss[:, 0], ss[:, 1])
   plt.show()
   plt.plot(expanded_set[:, 0], expanded_set[:, 1])
   plt.show()


   