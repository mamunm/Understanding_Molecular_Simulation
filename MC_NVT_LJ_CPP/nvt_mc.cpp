#include <iostream>
#include <random>
#include <vector>
#include <cmath>

class NVTSimulation
{
private:
    // System parameters
    double temperature;          // Reduced temperature
    double density;              // Reduced density
    double cutoff;               // Reduced cutoff
    int n_particles;             // Number of particles
    double box_length;           // Box length
    std::vector<double> x, y, z; // Particle positions

    // Random number generators
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> uniform_dist;

    // Maximum displacement
    double max_displacement;

    // System properties
    double energy;      // Total energy
    double pressure;    // Pressure
    double acceptance;  // Acceptance ratio
    int accepted_moves; // Number of accepted moves
    int total_moves;    // Total number of moves

    // Constants
    const double cutoff_squared = cutoff * cutoff;

public:
    NVTSimulation(int num_particles, double temp, double dens, double coff)
        : n_particles(num_particles), temperature(temp), density(dens), cutoff(coff),
          gen(rd()), uniform_dist(0.0, 1.0)
    {

        // Initialize system
        box_length = std::cbrt(n_particles / density);
        max_displacement = box_length / 20.0; // Initial guess

        // Initialize particle positions on a simple cubic lattice
        initializePositions();

        // Calculate initial energy
        energy = calculateTotalEnergy();
        pressure = calculatePressure();

        acceptance = 0.0;
        accepted_moves = 0;
        total_moves = 0;
    }

    void initializePositions()
    {
        x.resize(n_particles);
        y.resize(n_particles);
        z.resize(n_particles);

        int n = std::ceil(std::cbrt(n_particles));
        double spacing = box_length / n;

        int p = 0;
        for (int i = 0; i < n && p < n_particles; i++)
        {
            for (int j = 0; j < n && p < n_particles; j++)
            {
                for (int k = 0; k < n && p < n_particles; k++)
                {
                    x[p] = (i + 0.5) * spacing;
                    y[p] = (j + 0.5) * spacing;
                    z[p] = (k + 0.5) * spacing;
                    p++;
                }
            }
        }
    }

    double calculatePairEnergy(double r_squared)
    {
        if (r_squared > cutoff_squared)
            return 0.0;
        double r6i = 1.0 / (r_squared * r_squared * r_squared);
        return 4.0 * (r6i * r6i - r6i);
    }

    double calculateTailCorrection()
    {
        double rho = density;
        double rc3 = cutoff * cutoff * cutoff;
        double rc9 = rc3 * rc3 * rc3;
        return 8.0 * M_PI * rho * n_particles * (1.0 / (9.0 * rc9) - 1.0 / (3.0 * rc3)) / 3.0;
    }

    double calculatePressureTailCorrection()
    {
        double rho = density;
        double rc3 = cutoff * cutoff * cutoff;
        double rc9 = rc3 * rc3 * rc3;
        return 16.0 * M_PI * rho * rho * (2.0 / (9.0 * rc9) - 1.0 / (3.0 * rc3)) / 3.0;
    }

    double calculateTotalEnergy()
    {
        double total = 0.0;

        for (int i = 0; i < n_particles - 1; i++)
        {
            for (int j = i + 1; j < n_particles; j++)
            {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dz = z[i] - z[j];

                // Minimum image convention
                dx -= box_length * std::round(dx / box_length);
                dy -= box_length * std::round(dy / box_length);
                dz -= box_length * std::round(dz / box_length);

                double r_squared = dx * dx + dy * dy + dz * dz;
                if (r_squared < cutoff_squared)
                {
                    total += calculatePairEnergy(r_squared);
                }
            }
        }

        return total + calculateTailCorrection();
    }

    double calculatePressure()
    {
        double virial = 0.0;

        for (int i = 0; i < n_particles - 1; i++)
        {
            for (int j = i + 1; j < n_particles; j++)
            {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dz = z[i] - z[j];

                // Minimum image convention
                dx -= box_length * std::round(dx / box_length);
                dy -= box_length * std::round(dy / box_length);
                dz -= box_length * std::round(dz / box_length);

                double r_squared = dx * dx + dy * dy + dz * dz;
                if (r_squared < cutoff_squared)
                {
                    double r6i = 1.0 / (r_squared * r_squared * r_squared);
                    virial += 48.0 * (r6i * r6i - 0.5 * r6i);
                }
            }
        }

        return density * temperature + virial / (3.0 * box_length * box_length * box_length) + calculatePressureTailCorrection();
    }

    void performMove()
    {
        total_moves++;

        // Select random particle
        int particle = uniform_dist(gen) * n_particles;

        // Save old position
        double old_x = x[particle];
        double old_y = y[particle];
        double old_z = z[particle];

        // Calculate old energy contribution
        double old_energy = 0.0;
        for (int i = 0; i < n_particles; i++)
        {
            if (i != particle)
            {
                double dx = old_x - x[i];
                double dy = old_y - y[i];
                double dz = old_z - z[i];

                dx -= box_length * std::round(dx / box_length);
                dy -= box_length * std::round(dy / box_length);
                dz -= box_length * std::round(dz / box_length);

                double r_squared = dx * dx + dy * dy + dz * dz;
                if (r_squared < cutoff_squared)
                {
                    old_energy += calculatePairEnergy(r_squared);
                }
            }
        }

        // Generate new position
        double new_x = old_x + (2.0 * uniform_dist(gen) - 1.0) * max_displacement;
        double new_y = old_y + (2.0 * uniform_dist(gen) - 1.0) * max_displacement;
        double new_z = old_z + (2.0 * uniform_dist(gen) - 1.0) * max_displacement;

        // Apply periodic boundary conditions
        new_x -= box_length * std::floor(new_x / box_length);
        new_y -= box_length * std::floor(new_y / box_length);
        new_z -= box_length * std::floor(new_z / box_length);

        // Calculate new energy contribution
        double new_energy = 0.0;
        for (int i = 0; i < n_particles; i++)
        {
            if (i != particle)
            {
                double dx = new_x - x[i];
                double dy = new_y - y[i];
                double dz = new_z - z[i];

                dx -= box_length * std::round(dx / box_length);
                dy -= box_length * std::round(dy / box_length);
                dz -= box_length * std::round(dz / box_length);

                double r_squared = dx * dx + dy * dy + dz * dz;
                if (r_squared < cutoff_squared)
                {
                    new_energy += calculatePairEnergy(r_squared);
                }
            }
        }

        // Accept or reject move based on Metropolis criterion
        double delta_energy = new_energy - old_energy;
        if (delta_energy < 0.0 || uniform_dist(gen) < std::exp(-delta_energy / temperature))
        {
            x[particle] = new_x;
            y[particle] = new_y;
            z[particle] = new_z;
            energy += delta_energy;
            accepted_moves++;
        }

        // Adjust max displacement to maintain ~50% acceptance
        if (total_moves % 100 == 0)
        {
            double ratio = double(accepted_moves) / total_moves;
            if (ratio < 0.45)
                max_displacement *= 0.95;
            else if (ratio > 0.55)
                max_displacement *= 1.05;

            if (max_displacement > box_length / 2.0)
                max_displacement = box_length / 2.0;
        }
    }

    void run(int equilibration_steps, int production_steps)
    {
        std::cout << "Starting equilibration..." << std::endl;
        for (int step = 0; step < equilibration_steps; step++)
        {
            performMove();
            if (step % 100000 == 0)
            {
                std::cout << "Step " << step << std::endl;
            }
        }

        std::cout << "Starting production..." << std::endl;
        accepted_moves = 0;
        total_moves = 0;

        for (int step = 0; step < production_steps; step++)
        {
            performMove();

            if (step % 100000 == 0)
            {
                pressure = calculatePressure();
                double instant_temp = 2.0 * energy / (3.0 * n_particles);
                double acceptance_ratio = double(accepted_moves) / total_moves;

                std::cout << "Step " << step << ": "
                          << "Energy = " << energy / n_particles << " "
                          << "Temperature = " << instant_temp << " "
                          << "Pressure = " << pressure << " "
                          << "Acceptance = " << acceptance_ratio << std::endl;
            }
        }
    }
};

int main()
{
    // Simulation parameters
    int n_particles = 500;
    double temperature = 0.85;
    double density = 1E-3;
    double cutoff = 3;
    int equilibration_steps = 2E6;
    int production_steps = 8E6;

    NVTSimulation sim(n_particles, temperature, density, cutoff);
    sim.run(equilibration_steps, production_steps);

    return 0;
}