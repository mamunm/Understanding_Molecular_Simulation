#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

// Constants
#define BLOCK_SIZE 256
#define CUTOFF 3.0f // Reduced units
#define MAX_PARTICLES 4096

// Global variables for GPU
__constant__ float d_box_length;
__constant__ float d_temperature;
__constant__ float d_cutoff_squared;

// Structures
struct Particle
{
    float x, y, z;
};

// CUDA kernel for random number generator initialization
__global__ void init_rng(curandState *states, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// CUDA kernel for energy calculation
__global__ void calculate_energy(Particle *particles, int n_particles, float *energy)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_particles)
        return;

    __shared__ float s_energy[BLOCK_SIZE];
    s_energy[threadIdx.x] = 0.0f;

    Particle p1 = particles[idx];
    float e_local = 0.0f;

    for (int j = 0; j < n_particles; j++)
    {
        if (j == idx)
            continue;

        Particle p2 = particles[j];

        // Calculate minimum image convention
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;

        dx -= d_box_length * rintf(dx / d_box_length);
        dy -= d_box_length * rintf(dy / d_box_length);
        dz -= d_box_length * rintf(dz / d_box_length);

        float r2 = dx * dx + dy * dy + dz * dz;

        if (r2 < d_cutoff_squared)
        {
            float r2i = 1.0f / r2;
            float r6i = r2i * r2i * r2i;
            e_local += 4.0f * r6i * (r6i - 1.0f);
        }
    }

    s_energy[threadIdx.x] = e_local;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_energy[threadIdx.x] += s_energy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(energy, s_energy[0]);
    }
}

// Calculate tail corrections
__host__ float calculate_tail_correction(int n_particles, float box_length)
{
    float rho = n_particles / (box_length * box_length * box_length);
    float rc3 = CUTOFF * CUTOFF * CUTOFF;
    float rc9 = rc3 * rc3 * rc3;

    return (8.0f / 3.0f) * M_PI * rho * n_particles *
           (1.0f / (3.0f * rc9) - 1.0f / rc3);
}

// Calculate pressure tail correction
__host__ float calculate_pressure_tail(int n_particles, float box_length)
{
    float rho = n_particles / (box_length * box_length * box_length);
    float rc3 = CUTOFF * CUTOFF * CUTOFF;
    float rc9 = rc3 * rc3 * rc3;

    return (16.0f / 3.0f) * M_PI * rho * rho *
           (2.0f / (3.0f * rc9) - 1.0f / rc3);
}

// Main MC move kernel
__global__ void mc_move(Particle *particles, int n_particles,
                        curandState *states, float max_displacement,
                        float *acceptance)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_particles)
        return;

    curandState local_state = states[idx];

    // Select random particle
    int particle_idx = (int)(curand_uniform(&local_state) * n_particles);

    // Store old position
    Particle old_pos = particles[particle_idx];

    // Generate trial move
    Particle new_pos;
    new_pos.x = old_pos.x + (curand_uniform(&local_state) - 0.5f) * max_displacement;
    new_pos.y = old_pos.y + (curand_uniform(&local_state) - 0.5f) * max_displacement;
    new_pos.z = old_pos.z + (curand_uniform(&local_state) - 0.5f) * max_displacement;

    // Apply periodic boundary conditions
    new_pos.x -= d_box_length * floorf(new_pos.x / d_box_length);
    new_pos.y -= d_box_length * floorf(new_pos.y / d_box_length);
    new_pos.z -= d_box_length * floorf(new_pos.z / d_box_length);

    // Calculate energy change
    float delta_e = 0.0f;
    for (int j = 0; j < n_particles; j++)
    {
        if (j == particle_idx)
            continue;

        Particle other = particles[j];

        // Old distance
        float dx = old_pos.x - other.x;
        float dy = old_pos.y - other.y;
        float dz = old_pos.z - other.z;

        dx -= d_box_length * rintf(dx / d_box_length);
        dy -= d_box_length * rintf(dy / d_box_length);
        dz -= d_box_length * rintf(dz / d_box_length);

        float r2_old = dx * dx + dy * dy + dz * dz;

        // New distance
        dx = new_pos.x - other.x;
        dy = new_pos.y - other.y;
        dz = new_pos.z - other.z;

        dx -= d_box_length * rintf(dx / d_box_length);
        dy -= d_box_length * rintf(dy / d_box_length);
        dz -= d_box_length * rintf(dz / d_box_length);

        float r2_new = dx * dx + dy * dy + dz * dz;

        if (r2_old < d_cutoff_squared)
        {
            float r2i = 1.0f / r2_old;
            float r6i = r2i * r2i * r2i;
            delta_e -= 4.0f * r6i * (r6i - 1.0f);
        }

        if (r2_new < d_cutoff_squared)
        {
            float r2i = 1.0f / r2_new;
            float r6i = r2i * r2i * r2i;
            delta_e += 4.0f * r6i * (r6i - 1.0f);
        }
    }

    // Metropolis criterion
    if (delta_e < 0.0f ||
        curand_uniform(&local_state) < expf(-delta_e / d_temperature))
    {
        particles[particle_idx] = new_pos;
        atomicAdd(acceptance, 1.0f);
    }

    states[idx] = local_state;
}

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        printf("Usage: %s <temperature> <density> <n_particles> <n_steps> <n_equil>\n", argv[0]);
        return 1;
    }

    // Parse input parameters
    float temperature = atof(argv[1]);
    float density = atof(argv[2]);
    int n_particles = atoi(argv[3]);
    int n_steps = atoi(argv[4]);
    int n_equil = atoi(argv[5]);

    if (n_particles > MAX_PARTICLES)
    {
        printf("Error: Number of particles exceeds maximum (%d)\n", MAX_PARTICLES);
        return 1;
    }

    // Calculate box length from density
    float box_length = powf(n_particles / density, 1.0f / 3.0f);
    float cutoff_squared = CUTOFF * CUTOFF;

    // Copy constants to device
    cudaMemcpyToSymbol(d_box_length, &box_length, sizeof(float));
    cudaMemcpyToSymbol(d_temperature, &temperature, sizeof(float));
    cudaMemcpyToSymbol(d_cutoff_squared, &cutoff_squared, sizeof(float));

    // Allocate memory
    Particle *h_particles = (Particle *)malloc(n_particles * sizeof(Particle));
    Particle *d_particles;
    cudaMalloc(&d_particles, n_particles * sizeof(Particle));

    // Initialize particles on simple cubic lattice
    int n_side = (int)ceilf(powf(n_particles, 1.0f / 3.0f));
    float spacing = box_length / n_side;
    int p = 0;
    for (int i = 0; i < n_side && p < n_particles; i++)
    {
        for (int j = 0; j < n_side && p < n_particles; j++)
        {
            for (int k = 0; k < n_side && p < n_particles; k++)
            {
                h_particles[p].x = (i + 0.5f) * spacing;
                h_particles[p].y = (j + 0.5f) * spacing;
                h_particles[p].z = (k + 0.5f) * spacing;
                p++;
            }
        }
    }

    // Copy particles to device
    cudaMemcpy(d_particles, h_particles, n_particles * sizeof(Particle),
               cudaMemcpyHostToDevice);

    // Initialize RNG states
    curandState *d_states;
    cudaMalloc(&d_states, n_particles * sizeof(curandState));
    init_rng<<<(n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, time(NULL));

    // Main MC loop
    float *d_energy, *d_acceptance;
    cudaMalloc(&d_energy, sizeof(float));
    cudaMalloc(&d_acceptance, sizeof(float));

    float max_displacement = 0.1f * box_length; // Initial max displacement

    // Equilibration
    for (int step = 0; step < n_equil; step++)
    {
        float h_acceptance = 0.0f;
        cudaMemcpy(d_acceptance, &h_acceptance, sizeof(float),
                   cudaMemcpyHostToDevice);

        mc_move<<<(n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, n_particles, d_states, max_displacement, d_acceptance);

        if ((step + 1) % 1000 == 0)
        {
            cudaMemcpy(&h_acceptance, d_acceptance, sizeof(float),
                       cudaMemcpyDeviceToHost);
            float acceptance_rate = h_acceptance / (1000.0f * n_particles);

            // Adjust max displacement
            if (acceptance_rate < 0.4f)
                max_displacement *= 0.95f;
            else if (acceptance_rate > 0.6f)
                max_displacement *= 1.05f;
        }
    }

    // Production
    for (int step = 0; step < n_steps; step++)
    {
        float h_energy = 0.0f;
        float h_acceptance = 0.0f;
        cudaMemcpy(d_energy, &h_energy, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acceptance, &h_acceptance, sizeof(float),
                   cudaMemcpyHostToDevice);

        mc_move<<<(n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, n_particles, d_states, max_displacement, d_acceptance);

        if ((step + 1) % 1000 == 0)
        {
            calculate_energy<<<(n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE,
                               BLOCK_SIZE>>>(d_particles, n_particles, d_energy);

            cudaMemcpy(&h_energy, d_energy, sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_acceptance, d_acceptance, sizeof(float),
                       cudaMemcpyDeviceToHost);

            // Add tail corrections
            float e_tail = calculate_tail_correction(n_particles, box_length);
            float p_tail = calculate_pressure_tail(n_particles, box_length);

            // Calculate instantaneous temperature and pressure
            float inst_temp = h_energy / (1.5f * n_particles);
            float virial = -h_energy / 3.0f; // For Lennard-Jones
            float pressure = density * temperature +
                             virial / (box_length * box_length * box_length) + p_tail;

            printf("Step %d: E/N = %.4f (with tail = %.4f), T = %.4f, P = %.4f\n",
                   step + 1, h_energy / n_particles,
                   (h_energy + e_tail) / n_particles,
                   inst_temp, pressure);

            float acceptance_rate = h_acceptance / (1000.0f * n_particles);
            if (acceptance_rate < 0.4f)
                max_displacement *= 0.95f;
            else if (acceptance_rate > 0.6f)
                max_displacement *= 1.05f;
        }
    }

    // Cleanup
    free(h_particles);
    cudaFree(d_particles);
    cudaFree(d_states);
    cudaFree(d_energy);
    cudaFree(d_acceptance);

    return 0;
}