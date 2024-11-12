"""MD Simulation code."""
import json

import numpy as np
from loguru import logger
from pathlib import Path
import itertools
import pandas as pd
import sys
import numba as nb

from typing import NamedTuple

np.random.seed(0)

PI = 3.14159265

@nb.njit(parallel=True)
def respect_pbc_distance(dx, boxLength):
        """
        Correct the distance vector for periodic boundary conditions.
        
        Args:
            dx (float or numpy.ndarray): Distance(s) to be corrected.
        
        Returns:
            float or numpy.ndarray: Corrected distance(s).
        """
        hL = 0.5 * boxLength
        return np.where(dx > hL, dx - boxLength, np.where(dx < -hL, dx + boxLength, dx))
    
@nb.njit(parallel=True)
def determine_forces(positions, N, e_cut, e_cor, cutoff, boxLength):
        """
        Determine the forces on each particle using LJ potential.
        
        Returns:
            forces (numpy.ndarray): Forces on each particle.
            energy (float): Total energy.
            vir (float): Virial.
        """
        forces = np.zeros((N, 3))
        energy = 0
        vir = 0
        
        e_cut = e_cut
        e_cor = config.N * e_cor
        cutoff_sq = cutoff**2
        
        for i in nb.prange(N):
            for j in nb.prange(i+1, N):
                r_ij = positions[i] - positions[j]
                r_ij = respect_pbc_distance(r_ij, boxLength)
                r_ij_sq = np.sum(r_ij**2)
                
                if r_ij_sq < cutoff_sq:
                    r_ij_inv = 1 / r_ij_sq
                    r_ij_inv6 = r_ij_inv ** 6
                    ff = 48 * r_ij_inv * (r_ij_inv6 - 0.5 * r_ij_inv**3)
                    vir += 48 * (r_ij_inv6 - 0.5 * r_ij_inv**3)
                    forces[i] += ff * r_ij
                    forces[j] -= ff * r_ij
                    energy += 4 * (r_ij_inv6 - r_ij_inv**3) - e_cut
        
        # return forces, energy + e_cor, vir
        return forces, energy + e_cor, vir

class MDConfig(NamedTuple):
    """
    Configuration for the MD simulation engine.
    
    Args:
        N (int): Number of particles in the simulation.
        T (float): Temperature of the simulation [in reduced units].
        d (float): Density of the simulation [in reduced units].
        dt (float): Timestep for the simulation [in reduced units].
        tmax (float): Maximum time to run the simulation [in reduced units].
        cutoff (float): cutoff distance between particles [in reduced units].
    """
    N: int = 500
    T: float = 0.85
    d: float = 0.9
    dt: float = 0.01
    tmax: float = 5.0
    cutoff: float = 5.0
    
    def __str__(self):
        string = f"MDConfig(N={self.N}, T={self.T}, "
        string += f"d={self.d}, dt={self.dt}, tmax={self.tmax}, "
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

class MDSimulation:
    """
    A class for running a molecular dynamics simulation.
    """
    def __init__(self, config: MDConfig):
        self.config = config
        self.e_cut = self.compute_e_cut()
        self.e_cor = self.compute_e_cor()
        # self.p_cor = self.compute_p_cor()
        self.nve_id = self.get_exp_id()
        self.path = self.create_folder()
        self.boxLength = self.get_boxLength()
        self.history = []
        
        self.config.save_config(self.path / "config.json")
        logger.add(self.path / "file.log")
        logger.info(f"Box length: {self.get_boxLength()}")

    def compute_e_cut(self):
        """Lennard-Jones potential at cutoff"""
        return 4 * (1/self.config.cutoff**12 - 1/self.config.cutoff**6)
    
    def compute_e_cor(self):
        """long range correction"""
        return (8/3)*PI*self.config.d*(1/self.config.cutoff**9/3 - 1/self.config.cutoff**3)
    
    def compute_p_cor(self):
        """particle correlation"""
        return (16/3)*PI*self.config.d**2*(2/3/self.config.cutoff**9 - 1/self.config.cutoff**3)
    
    def create_folder(self):
        """
        Create a folder for the simulation with the experiment ID.
        """
        folder_path = Path("experiments") / self.nve_id
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    def determine_forces(self):
        """
        Determine the forces on each particle using LJ potential.
        
        Returns:
            forces (numpy.ndarray): Forces on each particle.
            energy (float): Total energy.
            vir (float): Virial.
        """
        forces, energy, vir = determine_forces(self.positions,
            self.config.N, self.e_cut, self.e_cor, self.config.cutoff, 
            self.boxLength)
        return forces, energy, vir
    # def determine_forces(self):
    #     """
    #     Determine the forces on each particle using LJ potential.
        
    #     Returns:
    #         forces (numpy.ndarray): Forces on each particle.
    #         energy (float): Total energy.
    #         vir (float): Virial.
    #     """
    #     forces = np.zeros((self.config.N, 3))
    #     energy = 0
    #     vir = 0
        
    #     e_cut = self.e_cut
    #     e_cor = self.config.N * self.e_cor
    #     cutoff_sq = self.config.cutoff**2
        
    #     for i in range(self.config.N):
    #         for j in range(i+1, self.config.N):
    #             r_ij = self.positions[i] - self.positions[j]
    #             r_ij = self.respect_pbc_distance(r_ij)
    #             r_ij_sq = np.sum(r_ij**2)
                
    #             if r_ij_sq < cutoff_sq:
    #                 r_ij_inv = 1 / r_ij_sq
    #                 r_ij_inv6 = r_ij_inv ** 6
    #                 ff = 48 * r_ij_inv * (r_ij_inv6 - 0.5 * r_ij_inv**3)
    #                 vir += 48 * (r_ij_inv6 - 0.5 * r_ij_inv**3)
    #                 forces[i] += ff * r_ij
    #                 forces[j] -= ff * r_ij
    #                 energy += 4 * (r_ij_inv6 - r_ij_inv**3) - e_cut
        
    #     # return forces, energy + e_cor, vir
    #     return forces, energy + e_cor, vir

    # def determine_forces(self):
    #     """
    #     Determine the forces on each particle using the Lennard-Jones potential.
    #     Optimized version using vectorization and reduced redundant calculations.
        
    #     Returns:
    #         tuple: (forces, energy, virial)
    #             - forces: numpy array of shape (N, 3) containing forces on each particle
    #             - energy: total potential energy of the system
    #             - virial: virial term for pressure calculation
    #     """
    #     N = self.config.N
    #     cutoff_sq = self.config.cutoff**2
    #     forces = np.zeros(shape=(N, 3))
    #     energy = 0
    #     vir = 0
        
    #     # Pre-compute all pairwise distances
    #     pos_diff = self.positions[:, np.newaxis] - self.positions[np.newaxis, :]
        
    #     # Apply PBC to all distances at once
    #     for dim in range(3):
    #         pos_diff[:, :, dim] = self.respect_pbc_distance(pos_diff[:, :, dim])
        
    #     # Calculate squared distances for all pairs
    #     r_sq = np.sum(pos_diff**2, axis=2)
        
    #     # Create mask for pairs within cutoff (upper triangular to avoid double counting)
    #     mask = np.triu((r_sq < cutoff_sq) & (r_sq > 0), k=1)
        
    #     # Calculate forces only for pairs within cutoff
    #     r_sq_inv = np.zeros_like(r_sq)
    #     r_sq_inv[mask] = 1.0 / r_sq[mask]
    #     r6_inv = r_sq_inv ** 3
    #     r12_inv = r6_inv ** 2
        
    #     # Force magnitudes (48 * r⁻¹⁴ - 24 * r⁻⁸)
    #     ff = 48 * r_sq_inv * (r6_inv - 0.5 * r_sq_inv ** 3)
        
    #     # Calculate forces preserving Newton's third law
    #     for dim in range(3):
    #         force_matrix = ff * pos_diff[:, :, dim]
    #         # Add force to particle i
    #         forces[:, dim] += np.sum(force_matrix * mask, axis=1)
    #         # Subtract force from particle j (transpose the mask and force matrix)
    #         forces[:, dim] -= np.sum(force_matrix.T * mask.T, axis=1)
        
    #     # Calculate energy (4 * (r⁻¹² - r⁻⁶))
    #     pair_energies = 4 * (r12_inv - r6_inv) - self.e_cut
    #     energy = np.sum(pair_energies * mask)  # Only sum upper triangle
        
    #     # Calculate virial
    #     vir = np.sum(ff * mask)
        
    #     return forces, energy + self.config.N * self.e_cor, vir
    
    # def determine_forces(self):
    #     """Determine the forces on each particle using LJ potential."""
    #     forces = np.zeros((self.config.N, 3))
    #     energy = 0
    #     vir = 0
    #     cutoff_sq = self.config.cutoff ** 2
    #     e_cut = self.e_cut
    #     e_cor = self.config.N * self.e_cor

    #     # Compute pairwise distance vectors and their squared magnitudes
    #     pos_diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
    #     pos_diff = self.respect_pbc_distance(pos_diff)
    #     r_ij_sq = np.sum(pos_diff ** 2, axis=-1)

    #     # Apply cutoff
    #     mask = (r_ij_sq < cutoff_sq) & (r_ij_sq > 0)
    #     r_ij_sq = r_ij_sq[mask]
    #     pos_diff = pos_diff[mask]

    #     # Compute inverse distances
    #     r_ij_inv_sq = 1.0 / r_ij_sq
    #     r_ij_inv_6 = r_ij_inv_sq ** 3
    #     r_ij_inv_12 = r_ij_inv_6 ** 2

    #     # Compute forces and energy
    #     ff = 48 * r_ij_inv_sq * (r_ij_inv_12 - 0.5 * r_ij_inv_6)
    #     forces_contrib = ff[:, np.newaxis] * pos_diff
    #     energy_contrib = 4 * (r_ij_inv_12 - r_ij_inv_6) - e_cut

    #     # Accumulate forces
    #     indices = np.array(np.where(mask)).T
    #     np.add.at(forces, indices[:, 0], forces_contrib)
    #     np.add.at(forces, indices[:, 1], -forces_contrib)

    #     # Accumulate energy and virial
    #     energy = np.sum(energy_contrib)
    #     vir = np.sum(ff)

    #     return forces, energy + e_cor, vir

    
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
        string = f"NVE_N={self.config.N}_T={self.config.T}_"
        string += f"d={self.config.d}_dt={self.config.dt}_"
        string += f"tmax={self.config.tmax}_cutoff={self.config.cutoff}"
        return string
    
    def initialize(self):
        """
        Initialize the simulation.
        """
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities()
        v_com = np.sum(self.velocities, axis=0) / self.config.N
        mean_v_squared = np.sum(self.velocities**2) / self.config.N
        fs = np.sqrt(3 * self.config.T / mean_v_squared)
        self.velocities -= v_com
        self.velocities *= fs
        self.prevPositions = self.positions - (self.velocities * self.config.dt)
        self.prevPositions = self.prevPositions % self.get_boxLength()
    
    def initialize_positions(self):
        """
        Initialize the particle positions randomly within the simulation box.
        """
        n3 = 2
        while n3**3 < self.config.N:
            n3 += 1
        
        frac = [i + 0.5 for i in range(n3)]
        positions = np.array(list(itertools.product(frac, repeat=3))) * self.get_boxLength()/n3
        return positions[:self.config.N]
    
    def initialize_velocities(self):
        """
        Initialize the particle velocities with a uniform distribution U[-.5, 0.5].
        """
        velocities = np.random.uniform(low=-0.5, high=0.5, size=(self.config.N, 3))
        return velocities
    
    def integrate_eom(self, forces, energy):
        """
        Update the positions and velocities using the verlet algorithm.
        """
        new_positions = 2 * self.positions - self.prevPositions + forces * self.config.dt**2
        new_positions = new_positions % self.get_boxLength()
        # distances = np.array([self.respect_pbc_distance(n - p) 
        #     for n, p in zip(new_positions, self.prevPositions)])
        hL = 0.5 * self.boxLength
        distances = new_positions - self.prevPositions
        distances = np.where(distances > hL, distances - self.boxLength, np.where(distances < -hL, distances + self.boxLength, distances))
        self.velocities = distances/(2*self.config.dt)
        self.prevPositions = self.positions
        self.positions = new_positions % self.get_boxLength()
        temp = np.sum(self.velocities**2)/(3*self.config.N)
        e_total = (energy + 0.5 * np.sum(self.velocities**2))/self.config.N
        return temp, e_total, 0.5 * np.sum(self.velocities**2)/self.config.N
    
    def respect_pbc_distance(self, r_ij):
        """
        Respect periodic boundary conditions for distance calculations.
        
        Args:
            r_ij (np.ndarray): Distance vector between particles.
        
        Returns:
            np.ndarray: Modified distance vector.
        """
        # return np.array([self.get_corrected_dx(dx) for dx in r_ij])
        return self.get_corrected_dx_vectorized(r_ij)
    
    def get_corrected_dx(self, dx):
        """
        Correct the distance vector for periodic boundary conditions.
        
        Args:
            dx (float): Distance to be corrected.
        
        Returns:
            float: Corrected distance.
        """
        hL = 0.5 * self.boxLength
        if dx > hL:
            dx -= self.boxLength
        elif dx < -hL:
            dx += self.boxLength
        return dx
    
    def get_corrected_dx_vectorized(self, dx):
        """
        Correct the distance vector for periodic boundary conditions.
        
        Args:
            dx (float or numpy.ndarray): Distance(s) to be corrected.
        
        Returns:
            float or numpy.ndarray: Corrected distance(s).
        """
        hL = 0.5 * self.boxLength
        return np.where(dx > hL, dx - self.boxLength, np.where(dx < -hL, dx + self.boxLength, dx))
    
    def run_simulation(self):
        """
        Run the molecular dynamics simulation.
        """
        logger.info("Running simulation...")
        self.initialize()
        t = 0
        while t < self.config.tmax:
            self.step(t=t)
            t += self.config.dt
        self.save_simulation()
        logger.info("Simulation completed.")
    
    def step(self, t):
        """
        Perform a single molecular dynamics step.
        
        Args:
            t (float): Current time.
        """
        forces, energy, virial = self.determine_forces()
        temp, e_total, e_kin = self.integrate_eom(forces, energy)
        pressure = (self.config.d * e_kin * 2. / 3.) + (virial / 3.0 / (self.config.N / self.config.d))
        self.history.append([t, temp, e_kin, energy/self.config.N, e_total, pressure])
        logger.info(f"Time: {t:.3f}, Temperature: {temp:.2e}, KE: {e_kin:0.2e} PE: {energy/self.config.N:0.2e} Energy: {e_total:.2e} Pressure: {pressure:0.2f}")
        
    def save_simulation(self):
        """
        Save the final positions and velocities of the particles to a file.
        """
        positions_file = self.path / "positions.txt"
        velocities_file = self.path / "velocities.txt"
        
        np.savetxt(positions_file, self.positions)
        np.savetxt(velocities_file, self.velocities)
        
        df = pd.DataFrame(self.history, 
            columns=["time", "temperature", "KE", "PE", "TE", "P"])
        df.to_csv(self.path / "history.csv", index=False)


if __name__ == "__main__":
    # parameters = [
    #     (0.851, 0.776),
    #     (0.853, 0.780),
    #     (0.852, 0.820),
    #     (0.851, 0.840),
    #     (0.849, 0.860),
    #     (0.851, 0.900)
    #     ]
    
    # for T, d in parameters:
    #     logger.info(f"Running simulation for T={T}, d={d}...")
    #     config = MDConfig(N=500,
    #                     T=T,
    #                     d=d,
    #                     cutoff=3,
    #                     tmax=5,
    #                     dt=0.005)
    #     simulation = MDSimulation(config)
    #     simulation.run_simulation()
    config = MDConfig(N=500,
                    T=0.851,
                    d=0.776,
                    cutoff=3,
                    tmax=5,
                    dt=0.005)
    simulation = MDSimulation(config)
    simulation.run_simulation()