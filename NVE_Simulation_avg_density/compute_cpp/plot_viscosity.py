import matplotlib.pyplot as plt
import numpy as np

# Read data from the file
data = np.loadtxt('viscosity.txt', skiprows=1)

# Extract time and average viscosity
time = data[:, 0]
avg_viscosity = data[:, 4]

# Convert time from fs to ps
time_ps = time / 1000

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time_ps, avg_viscosity, linestyle='-', color='#002147')

plt.axhline(y=0.728, color='black', linewidth=2, linestyle='--')

# Set labels and title
plt.xlabel('Time (ps)', fontsize=12)
plt.ylabel('Average Viscosity (cP)', fontsize=12)
plt.title('Average Viscosity vs. Time', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Customize tick labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add a legend
plt.legend(['Viscosity'], loc='best')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig("viscosity.png")
plt.show()