# Molecular Dynamics Simulation Code from Basics using Python

## Overview

This repository contains a collection of molecular dynamics simulation codes designed to simulate particles under
different thermodynamic ensembles using Python and C++. Below are detailed descriptions of the main scripts included in this
repository:

## MD_NVE_LJ: Molecular Dynamics in NVE Ensemble

The `MD_NVE_LJ.py` script simulates particles contained in a box using a Lennard-Jones potential under an NVE (constant
Number, Volume, and Energy) ensemble.

### Inputs:

- **N**: Number of particles
- **T**: Temperature
- **d**: Density

This simulation is particularly useful for understanding system dynamics and energy conservation in a closed system.

## MC_NVT_LJ: Monte Carlo Simulation in NVT Ensemble

The `MC_NVT_LJ.py` script performs a Monte Carlo simulation of particles interactions using a Lennard-Jones potential,
constrained within an NVT (constant Number, Volume, and Temperature) ensemble.

### Inputs:

- **N**: Number of particles
- **d**: Density

This method is effective for understanding equilibrium properties of fluids.

## NVE_simulation_avg_density: Transport Properties Calculation

The `NVE_simulation_avg_density.py` script is used to compute transport properties such as viscosity and diffusivity
using LAMMPS, a classical molecular dynamics code.

### Features:

- Computes transport properties for systems modeled under NVE conditions.
- Useful for understanding how particles flow and spread within the given system density.

## How to Use

To run these simulations, ensure that all dependencies are installed as per the `requirements.txt` file. You might need
additional Python libraries such as NumPy and SciPy for numerical computations. Each script can be executed with the
required parameters.

## Requirements

- Python >3.10
- NumPy
- SciPy
- Optional: LAMMPS for `NVE_simulation_avg_density`

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
