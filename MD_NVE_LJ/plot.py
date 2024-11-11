from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path
import pandas as pd # type: ignore
import json

def get_volume(file_path):
    """
    Returns the volume of the simulation box from the 'output.txt' file.
    
    Parameters:
        file_path (pathlib.Path): Path to the directory containing the 'output.txt' file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()[0]
    boxLength = float(lines.split(' ')[-1])
    return boxLength**3

def plot_rdf(file_path):
    """
    Plots the radial distribution function (RDF) from a text file.
    
    Parameters:
        file_path (pathlib.Path): Path to the directory containing the 'positions.txt' file.
    """
    positions = np.loadtxt(file_path / "positions.txt")
    distances = pdist(positions)
    hist, bin_edges = np.histogram(distances, bins=1000)
    volume = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    rdf = hist / volume / len(positions)

    plt.plot(bin_edges[:-1], rdf)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.xlim([0, 4])
    plt.grid(True)
    plt.savefig(file_path / "rdf.png")
    plt.close()

def plot_energy_profile(file_path):
    """
    Plots the total energy, kinetic energy, and potential energy profiles from a CSV file.
    Also calculates and displays the average kinetic and potential energy after discarding the first 120 seconds of data.
    
    Parameters:
        file_path (pathlib.Path): Path to the directory containing the 'history.csv' file.
    """
    df = pd.read_csv(file_path / "history.csv")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["TE"], label="Total Energy")
    plt.plot(df["time"], df["KE"], label="Kinetic Energy")
    plt.plot(df["time"], df["PE"], label="Potential Energy")
    
    df = df[df['time'] >= 2]
    avg_ke = df['KE'].mean()
    std_ke = df['KE'].std()
    avg_pe = df['PE'].mean()
    std_pe = df['PE'].std()
    
    plt.text(0.05, 0.70, f"Average Kinetic Energy: {avg_ke:.2f} $\pm$ {std_ke:0.3f}", transform=plt.gca().transAxes, va='top', fontsize=14)
    plt.text(0.05, 0.65, f"Average Potential Energy: {avg_pe:.2f} $\pm$ {std_pe:0.3f}", transform=plt.gca().transAxes, va='top', fontsize=14)
    
    plt.xlabel('Time [min]')
    plt.ylabel('Energy')
    plt.title('Energy vs time profile')
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(file_path / "energy_profile.png")
    plt.close()

def plot_temp_profile(file_path):
    """
    Plots the temperature profile from a CSV file.
    Also calculates and displays the average temperature after discarding the first 120 seconds of data.
    
    Parameters:
        file_path (pathlib.Path): Path to the directory containing the 'history.csv' file.
    """
    with open(file_path / "config.json") as f:
        config = json.load(f)
    df = pd.read_csv(file_path / "history.csv")
    plt.plot(df["time"], df["temperature"])
    plt.axhline(config["T"], linewidth=2, c="cyan")
    
    df = df[df['time'] >= 2]
    avg_temp = df['temperature'].mean()
    std_temp = df['temperature'].std()
    plt.text(0.2, 0.55, f"Average Temperature: {avg_temp:.2f} $\pm$ {std_temp:0.3f}", transform=plt.gca().transAxes, va='top', fontsize=14)
    
    plt.xlabel('Time [min]')
    plt.ylabel('Temperature')
    plt.title('Temperature vs time profile')
    plt.grid(True)
    plt.savefig(file_path / "temp_profile.png")
    plt.close()
    
paths = Path("experiments_test").glob("*")
for path in paths:
    plot_rdf(path)
    plot_energy_profile(path)
    plot_temp_profile(path)
