# Function Approximation using Neural Networks in Matlab

**Author:** Alex Clymo  
**Date:** 24 June 2025

This repository demonstrates how to set up and train a shallow feedforward neural network in MATLAB to approximate a function $`y=f(x)`$ where both input $`x`$ and output $`y`$ are vector-valued, and provides custom tools for evaluating the output and Jacobian of the approximated function.

> 🚧 **Warning!** This code is very much in early development. I put it online at this early stage to encourage myself to start using Github. Please use with caution, and comments are always welcome. 

## 🔍 Purpose

1. To provide simple examples of how to define and train a neural net using MATLAB’s `feedforwardnet` and `train` functions.
2. To offer custom, vectorized functions for evaluating the output and Jacobian of the trained network by extracting the network structure into a custom `netParams` structure.

## 📂 Files Included

- `main_test1.m`, `main_test2.m` — example scripts showing how to train a network and testing the custom evaluation functions.
- `netExtractParams.m` — extracts the trained network’s parameters into a structure `netParams`.
- `netEvalF.m` — evaluates the output of the network for a batch of inputs.
- `netEvalFp.m` — evaluates the Jacobian of the network output with respect to the input for a batch of inputs.

## 📘 Full Documentation

For a complete explanation of the code structure, mathematical background, and examples, see the full [pdf readme](https://github.com/alexclymo/netEvalFp/blob/main/readme/netEvalFp_readme.pdf).

## 🙌 Acknowledgement

These codes and notes build very heavily on the work of Alessandro Villa and Vytautas Valaitis, whose paper "A Machine Learning Projection Method for Macro-finance Models" (QE, 2024) is a fantastic reference for explaining the basics of neural nets and machine learning to a macroeconomist. My codes build on the codes they made available at their Github repository [here](https://github.com/forket86/ANNEA/).

## 🛠 Requirements

- MATLAB (tested on R2024b) with Optimization and Deep Learning toolboxes
