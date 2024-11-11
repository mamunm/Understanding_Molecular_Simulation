from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path
import pandas as pd # type: ignore
import json

def get_volume(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[0]
    boxLength = float(lines.split(' ')[-1])
    return boxLength**3

def plot_rdf(file_path):
    positions = np.loadtxt(file_path / "positions.txt")
    distances = pdist(positions)
    hist, bin_edges = np.histogram(distances, bins=1000)
    volume = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    rdf = hist / volume / len(positions)

    plt.plot(bin_edges[:-1], rdf)
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.xlim([0, 4])
    plt.grid(True)
    plt.savefig(file_path / "rdf.png")
    plt.close()
    
def plot_energy_profile(file_path):
    df = pd.read_csv(file_path / "history.csv")
    plt.plot(df["time"], df["TE"], label="Total Energy")
    plt.plot(df["time"], df["KE"], label="Kinetic Energy")
    plt.plot(df["time"], df["PE"], label="Potential Energy")
    plt.xlabel('Time [s]')
    plt.ylabel('Energy')
    plt.title('Energy vs time profile')
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(file_path / "energy_profile.png")
    plt.close()

def plot_temp_profile(file_path):
    with open(file_path / "config.json") as f:
        config = json.load(f)
    df = pd.read_csv(file_path / "history.csv")
    plt.plot(df["time"], df["temperature"])
    plt.axhline(config["T"], linewidth=2, c="cyan")
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature')
    plt.title('Temperature vs time profile')
    plt.grid(True)
    plt.savefig(file_path / "temp_profile.png")
    plt.close()
    
paths = Path("experiments_new").glob("*")
for path in paths:
    plot_rdf(path)
    plot_energy_profile(path)
    plot_temp_profile(path)