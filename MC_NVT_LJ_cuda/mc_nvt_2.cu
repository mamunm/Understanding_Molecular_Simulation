#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

const double SIGMA = 1.0;
const double EPSILON = 1.0;
const double CUTOFF = 3.0 * SIGMA;
const int BLOCK_SIZE = 128;

// Tail correction factors for energy and pressure
__host__ __device__ double tail_correction_energy(double rho, double cutoff)
{
    double r3 = pow(cutoff, 3);
    return (8.0 / 3.0) * M_PI * rho * (1.0 / (9 * r3 * r3) - 1.0 / (3 * r3));
}

__host__ __device__ double tail_correction_pressure(double rho, double cutoff)
{
    double r3 = pow(cutoff, 3);
    return (16.0 / 3.0) * M_PI * rho * (2.0 / (9 * r3 * r3) - 1.0 / (3 * r3));
}

// Kernel to initialize random states
__global__ void init_rng(curandState *states, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Lennard-Jones potential calculation with cutoff
__device__ double lennard_jones(double r2, double sigma, double epsilon)
{
    double sr2 = (sigma * sigma) / r2;
    double sr6 = sr2 * sr2 * sr2;
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

// Kernel to perform a single Monte Carlo step
__global__ void mc_step(double3 *positions, double3 *box, curandState *states, double beta, double sigma, double epsilon, double rho, double cutoff, int *accept_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    curandState localState = states[idx];
    double3 old_pos = positions[idx];
    double3 new_pos = old_pos;

    // Propose a random displacement
    new_pos.x += (curand_uniform(&localState) - 0.5) * 0.1;
    new_pos.y += (curand_uniform(&localState) - 0.5) * 0.1;
    new_pos.z += (curand_uniform(&localState) - 0.5) * 0.1;

    // Apply periodic boundary conditions
    new_pos.x = fmod(new_pos.x + box->x, box->x);
    new_pos.y = fmod(new_pos.y + box->y, box->y);
    new_pos.z = fmod(new_pos.z + box->z, box->z);

    // Calculate energy change
    double dE = 0.0;
    for (int i = 0; i < n; i++)
    {
        if (i == idx)
            continue;

        double3 dr_old = {old_pos.x - positions[i].x, old_pos.y - positions[i].y, old_pos.z - positions[i].z};
        double3 dr_new = {new_pos.x - positions[i].x, new_pos.y - positions[i].y, new_pos.z - positions[i].z};

        double r2_old = dr_old.x * dr_old.x + dr_old.y * dr_old.y + dr_old.z * dr_old.z;
        double r2_new = dr_new.x * dr_new.x + dr_new.y * dr_new.y + dr_new.z * dr_new.z;

        if (r2_old < cutoff * cutoff)
            dE -= lennard_jones(r2_old, sigma, epsilon);
        if (r2_new < cutoff * cutoff)
            dE += lennard_jones(r2_new, sigma, epsilon);
    }

    // Metropolis criterion
    if (dE < 0.0 || exp(-beta * dE) > curand_uniform(&localState))
    {
        positions[idx] = new_pos;
        atomicAdd(accept_count, 1);
    }

    states[idx] = localState;
}

// Host function to perform simulation
void run_simulation(int n, double T, double rho, int steps, int eq_steps)
{
    double beta = 1.0 / T;
    double box_size = pow(n / rho, 1.0 / 3.0);

    double3 *d_positions;
    curandState *d_states;
    double3 *d_box;
    int *d_accept_count;

    cudaMalloc(&d_positions, n * sizeof(double3));
    cudaMalloc(&d_states, n * sizeof(curandState));
    cudaMalloc(&d_box, sizeof(double3));
    cudaMalloc(&d_accept_count, sizeof(int));

    double3 box = {box_size, box_size, box_size};
    cudaMemcpy(d_box, &box, sizeof(double3), cudaMemcpyHostToDevice);

    init_rng<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, time(0));

    for (int step = 0; step < steps; step++)
    {
        cudaMemset(d_accept_count, 0, sizeof(int));
        mc_step<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_box, d_states, beta, SIGMA, EPSILON, rho, CUTOFF, d_accept_count, n);

        if (step >= eq_steps && step % 1000 == 0)
        {
            // Compute energy, pressure with tail corrections
            double tail_e = tail_correction_energy(rho, CUTOFF);
            double tail_p = tail_correction_pressure(rho, CUTOFF);

            std::cout << "Step: " << step << " | Energy (w/ tail): " << tail_e << " | Pressure (w/ tail): " << tail_p << "\n";
        }
    }

    cudaFree(d_positions);
    cudaFree(d_states);
    cudaFree(d_box);
    cudaFree(d_accept_count);
}

int main()
{
    int n = 500;          // Number of particles
    double T = 0.9;       // Reduced temperature
    double rho = 0.9;     // Reduced density
    int steps = 100000;   // Number of total steps
    int eq_steps = 50000; // Number of equilibration steps

    run_simulation(n, T, rho, steps, eq_steps);

    return 0;
}
