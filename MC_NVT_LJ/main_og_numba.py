"""MC NVT Simulation code."""
import json

import numpy as np
from loguru import logger
from pathlib import Path
import itertools
import pandas as pd # type: ignore
import numba

from typing import NamedTuple

np.random.seed(0)

PI = 3.14159265

@numba.njit
def lj_numba_scalar_prange(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot

@numba.njit
def force_numba_scalar_prange(r):
    sr6 = (1./r)**6
    pot = (sr6*sr6 - 0.5*sr6)
    return pot


@numba.njit
def distance_numba_scalar_prange(atom1, atom2, boxLength):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]
    
    dx = np.where(dx > boxLength, dx - boxLength, np.where(dx < 0, dx + boxLength, dx))
    dy = np.where(dy > boxLength, dy - boxLength, np.where(dy < 0, dy + boxLength, dy))
    dz = np.where(dz > boxLength, dz - boxLength, np.where(dz < 0, dz + boxLength, dz))

    r = (dx * dx + dy * dy + dz * dz) ** 0.5

    return r

@numba.njit(parallel=True)
def potential_numba_scalar_prange(cluster, boxLength):
    energy = 0.0
    # numba.prange requires parallel=True flag to compile.
    # It causes the loop to run in parallel in multiple threads.
    for i in numba.prange(len(cluster)):
        for j in range(i + 1, len(cluster)):
            r = distance_numba_scalar_prange(cluster[i], cluster[j], boxLength)
            e = lj_numba_scalar_prange(r)
            energy += e

    return energy

@numba.njit(parallel=True)
def virial_numba_scalar_prange(cluster, boxLength):
    virial = 0.0
    # numba.prange requires parallel=True flag to compile.
    # It causes the loop to run in parallel in multiple threads.
    for i in numba.prange(len(cluster)-1):
        for j in range(i + 1, len(cluster)):
            r = distance_numba_scalar_prange(cluster[i], cluster[j], boxLength)
            vir = force_numba_scalar_prange(r)
            virial += vir

    return virial

class MCNVTConfig(NamedTuple):
    """
    Configuration for the MD simulation engine.
    
    Args:
        N (int): Number of particles in the simulation.
        T (float): Temperature of the simulation [in reduced units].
        d (float): Density of the simulation [in reduced units].
        NSteps (int): Number of steps
        cutoff (float): cutoff distance between particles [in reduced units].
    """
    N: int = 500
    T: float = 0.85
    d: float = 0.9
    NSteps: int = 500000
    cutoff: float = 5.0
    
    def __str__(self):
        string = f"MCNVTConfig(N={self.N}, T={self.T}, "
        string += f"d={self.d}, NSteps={self.NSteps}, "
        string += f"cutoff={self.cutoff})"
        return string
    
    def save_config(self, file_path: Path):
        """
        Save the configuration to a json file.
        """
        config_dict = self._asdict()
        with open(file_path, "w") as file:
            json.dump(config_dict, file, indent=4)
    
    @classmethod
    def load_config(cls, file_path: Path):
        """
        Load the configuration from a json file.
        """
        with open(file_path, "r") as file:
            config_dict = json.load(file)
        return cls(**config_dict)

class MCNVTSimulation:
    """
    A class for running a MC NVT simulation.
    """
    def __init__(self, config: MCNVTConfig):
        self.config = config
        self.e_cut = self.compute_e_cut()
        self.e_cor = self.compute_e_cor()
        self.p_cor = self.compute_p_cor()
        self.nvt_id = self.get_exp_id()
        self.path = self.create_folder()
        self.boxLength = self.get_boxLength()
        self.history = []
        
        self.config.save_config(self.path / "config.json")
        logger.add(self.path / "file.log")
        logger.info(f"Box length: {self.get_boxLength()}")

    def compute_e_cut(self):
        """Lennard-Jones potential at cutoff"""
        return 4 * ((1/self.config.cutoff**12) - (1/self.config.cutoff**6))
    
    def compute_e_cor(self):
        """long range correction"""
        return (8/3)*PI*self.config.d*(1/self.config.cutoff**9/3 - 1/self.config.cutoff**3)

    def compute_p_cor(self):
        """particle correlation"""
        return (16/3)*PI*self.config.d**2*(2/3/self.config.cutoff**9 - 1/self.config.cutoff**3)
    
    def compute_pressure(self):
        """Calculate and return the pressure."""
        volume = self.config.N / self.config.d
        vir = self.compute_vir()
        return self.config.d * self.config.T + vir/volume
    
    def compute_vir(self):
        """Calculate and return the virial."""
        return 48 * virial_numba_scalar_prange(self.positions,
            self.boxLength) / 3 + self.p_cor
    
    def create_folder(self):
        """
        Create a folder for the simulation with the experiment ID.
        """
        folder_path = Path("experiments") / self.nvt_id
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def get_boxLength(self):
        """
        Calculate and return the box length.
        """
        volume = self.config.N / self.config.d
        box_length = volume ** (1/3)
        return box_length
    
    def get_exp_id(self):
        """
        Generate and return a unique experiment ID.
        """
        string = f"NVT_N={self.config.N}_T={self.config.T}_"
        string += f"d={self.config.d}_NSteps={self.config.NSteps}_"
        string += f"cutoff={self.config.cutoff}"
        return string
    
    
    def get_energy(self, positions):
        """
        Calculate the potential energy of the system.
        """
        return potential_numba_scalar_prange(positions, self.boxLength) + self.config.N * self.e_cor
    
    def initialize(self):
        """
        Initialize the simulation.
        """
        self.positions = self.initialize_positions()
        np.savetxt(self.path / "coord_0.txt", self.positions)
    
    def initialize_positions(self):
        """
        Initialize the particle positions randomly within the simulation box.
        """
        n3 = 2
        while n3**3 < self.config.N:
            n3 += 1
        
        frac = [i + 0.5 for i in range(n3)]
        positions = np.array(list(itertools.product(frac, repeat=3))) * self.get_boxLength()/n3
        positions = positions % self.boxLength
        return positions[:self.config.N]
    
    def run_simulation(self):
        """
        Run the Monte-Carlo simulation.
        """
        logger.info("Running simulation...")
        energies = []
        pressures = []
        self.initialize()
        self.energy = self.get_energy(self.positions)
        energies.append(self.energy)
        for step in range(self.config.NSteps):
            self.metropolis_step()
            energies.append(self.energy)
            pressure = self.compute_pressure()
            pressures.append(pressure)
            if step % 50 == 0:
                logger.info(f"Step: {step} | energy: {self.energy: 0.2e} | pressure: {pressure: 0.2e}")
        self.save_simulation(energies)
        logger.info("Simulation completed.")
    
    def metropolis_step(self):
        """
        Perform a single MC step.
        """
        iPart = np.random.randint(self.config.N)
        
        new_positions = self.positions.copy()
        new_positions[iPart] = self.positions[iPart] + (np.random.rand(3) - 0.5)
        new_positions = new_positions % self.boxLength
        
        new_energy = self.get_energy(new_positions)
        energy_diff = new_energy - self.energy
        
        if energy_diff <= 0 or np.random.rand() < np.exp(-energy_diff / self.config.T):
            self.energy = new_energy
            self.positions = new_positions
        
    def save_simulation(self, energy_list):
        """
        Save the final positions and velocities of the particles to a file.
        
        Args:
            energy_list (list): List of potential energies at each step.
        """
        positions_file = self.path / "positions.txt"
        energy_file = self.path / "energy.txt"
        
        np.savetxt(positions_file, self.positions)
        np.savetxt(energy_file, energy_list)


if __name__ == "__main__":
    config = MCNVTConfig(N=500,
                         T=2,
                         d=0.9,
                         cutoff=3,
                         NSteps=int(3E8))
    simulation = MCNVTSimulation(config)
    simulation.run_simulation()
    # pos = np.random.rand(10, 3)
    # print(pos)
    # print(simulation.get_energy(pos))