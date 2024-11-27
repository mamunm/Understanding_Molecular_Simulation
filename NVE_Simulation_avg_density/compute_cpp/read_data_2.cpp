#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cmath>

// Structure to hold each line of data
struct StressData
{
    int timestep;
    double pxx, pyy, pzz, pxy, pxz, pyz;
};

// Structure to hold correlation values for all shear components
struct ShearCorrelation
{
    double xy, yz, zx;
};

// Structure to hold viscosity components
struct ShearViscosity
{
    double xy, yz, zx;
};

// Create a function to generate the dt vector
std::vector<int> generateIntVector(int start, int end, int step)
{
    std::vector<int> vec;
    int n = (end - start) / step + 1;
    vec.reserve(n);

    for (int i = start; i <= end; i += step)
    {
        vec.push_back(i);
    }

    return vec;
}

// Structure to store correlations for all times
struct CorrelationData
{
    std::vector<int> dt;
    std::vector<ShearCorrelation> correlations;
};

// Time measurement function
auto getTime()
{
    return std::chrono::high_resolution_clock::now();
}

// Time difference in seconds
double getTimeDiff(const auto &start, const auto &end)
{
    return std::chrono::duration<double>(end - start).count();
}

void log(const std::string &message)
{
    static auto start_time = getTime();
    auto current_time = getTime();
    double elapsed = getTimeDiff(start_time, current_time);
    std::cout << "[" << std::fixed << std::setprecision(3) << elapsed << "s] "
              << message << std::endl;
}

std::vector<StressData> readStressFile(const std::string &filename)
{
    auto start_time = getTime();

    // Reserve vector size (assuming 7e7 points)
    std::vector<StressData> data;
    data.reserve(70000000); // Preallocate memory to avoid reallocations

    // Open file with optimization flags
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        log("Error: Unable to open file " + filename);
        return data;
    }

    log("Started reading file: " + filename);

    // Buffer for reading
    constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1MB buffer
    file.rdbuf()->pubsetbuf(new char[BUFFER_SIZE], BUFFER_SIZE);

    // Skip header lines
    std::string line;
    std::getline(file, line);
    std::getline(file, line);

    // Variables outside loop to avoid reallocation
    StressData stress;
    char buffer[256]; // Buffer for line reading

    // Read data lines
    while (file.getline(buffer, sizeof(buffer)))
    {
        char *ptr = buffer;

        // Fast string to number conversion
        stress.timestep = std::strtol(ptr, &ptr, 10);
        stress.pxx = std::strtod(ptr, &ptr);
        stress.pyy = std::strtod(ptr, &ptr);
        stress.pzz = std::strtod(ptr, &ptr);
        stress.pxy = std::strtod(ptr, &ptr);
        stress.pxz = std::strtod(ptr, &ptr);
        stress.pyz = std::strtod(ptr, &ptr);

        data.push_back(stress);

        // Progress reporting every 5M points
        // if (data.size() % 5000000 == 0)
        // {
        //     log("Read " + std::to_string(data.size()) + " points");
        // }
    }

    auto end_time = getTime();
    log("Finished reading " + std::to_string(data.size()) + " entries in " +
        std::to_string(getTimeDiff(start_time, end_time)) + " seconds");

    return data;
}

// Function to compute autocorrelation at time dt
ShearCorrelation computeAutocorrelation(const std::vector<StressData> &data, int dt)
{
    ShearCorrelation correlation = {0.0, 0.0, 0.0};
    int valid_counts = 0;

    // Number of points we can use (N-dt)
    int n_points = data.size() - dt;

    if (n_points <= 0)
    {
        log("Error: dt is larger than data size");
        return correlation;
    }

    // Compute <P_xy(t+dt)P_xy(t)>
    for (int t = 0; t < n_points; ++t)
    {
        correlation.xy += data[t + dt].pxy * data[t].pxy;
        correlation.yz += data[t + dt].pyz * data[t].pyz;
        correlation.zx += data[t + dt].pxz * data[t].pxz;
        valid_counts++;
    }

    // Return the averages if we have valid points
    if (valid_counts > 0)
    {
        correlation.xy /= valid_counts;
        correlation.yz /= valid_counts;
        correlation.zx /= valid_counts;
    }
    else
    {
        log("Warning: No valid pairs found for dt = " + std::to_string(dt));
    }

    return correlation;
}

// Function to print or save results
void saveCorrelations(const CorrelationData &corr_data, const std::string &filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        log("Error: Unable to open output file: " + filename);
        return;
    }

    // Write header
    outfile << "# dt xy_correlation yz_correlation zx_correlation\n";

    // Write data
    for (size_t i = 0; i < corr_data.dt.size(); ++i)
    {
        outfile << std::fixed << std::setprecision(6)
                << corr_data.dt[i] << " "
                << corr_data.correlations[i].xy << " "
                << corr_data.correlations[i].yz << " "
                << corr_data.correlations[i].zx << "\n";
    }

    outfile.close();
    log("Saved correlation data to: " + filename);
}

// Function to save viscosity evolution
void saveViscosity(const std::vector<ShearViscosity> &visc_evolution,
                   const std::vector<int> &dt_values,
                   const std::string &filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        log("Error: Unable to open output file: " + filename);
        return;
    }

    // Write header
    outfile << "# time xy_viscosity yz_viscosity zx_viscosity average_viscosity\n";

    // Write data
    for (size_t i = 0; i < visc_evolution.size(); ++i)
    {
        double avg_visc = (visc_evolution[i].xy +
                           visc_evolution[i].yz +
                           visc_evolution[i].zx) /
                          3.0;

        outfile << std::fixed << std::setprecision(6)
                << dt_values[i] << " "
                << visc_evolution[i].xy << " "
                << visc_evolution[i].yz << " "
                << visc_evolution[i].zx << " "
                << avg_visc << "\n";
    }

    outfile.close();
    log("Saved viscosity evolution to: " + filename);
}

// Function to compute viscosity using trapezoidal rule
std::vector<ShearViscosity> computeViscosity(const CorrelationData &corr_data,
                                             double V, double kBT, double dt)
{
    std::vector<ShearViscosity> viscosityVector;
    viscosityVector.reserve(corr_data.dt.size());
    ShearViscosity runningViscosity = {0.0, 0.0, 0.0};
    // Prefactor V/kBT
    double prefactor = V / kBT;
    double P_conv = 1e13; // 1 bar2 = 1e10 Pa, 1000 Pa.s = 1 cP
    double factor = P_conv * prefactor * dt / 2.0;

    // First point (weight = 1)
    runningViscosity.xy += corr_data.correlations[0].xy;
    runningViscosity.yz += corr_data.correlations[0].yz;
    runningViscosity.zx += corr_data.correlations[0].zx;

    for (size_t i = 1; i < corr_data.dt.size() - 1; ++i)
    {
        // Integration using trapezoidal rule
        // η = (V/kBT)*(dt/2)*[C(0) + 2*C(dt) + 2*C(2dt) + ... + 2*C((n-1)dt) + C(ndt)]
        ShearViscosity viscosity = {0.0, 0.0, 0.0};
        viscosity.xy = corr_data.correlations[i].xy + runningViscosity.xy;
        viscosity.yz = corr_data.correlations[i].yz + runningViscosity.yz;
        viscosity.zx = corr_data.correlations[i].zx + runningViscosity.zx;

        // Multiply by dt/2 and prefactor
        viscosity.xy *= factor;
        viscosity.yz *= factor;
        viscosity.zx *= factor;

        viscosityVector.push_back(viscosity);

        runningViscosity.xy += 2 * corr_data.correlations[i].xy;
        runningViscosity.yz += 2 * corr_data.correlations[i].yz;
        runningViscosity.zx += 2 * corr_data.correlations[i].zx;

        if (i % 500 == 0)
        {
            log("Computed viscosity for dt = " + std::to_string(corr_data.dt[i]) +
                ", average = " + std::to_string((viscosity.xy + viscosity.yz + viscosity.zx) / 3.0));
        }
    }

    // Multiply by dt/2 and prefactor
    ShearViscosity viscosity = {0.0, 0.0, 0.0};
    viscosity.xy = corr_data.correlations.back().xy + runningViscosity.xy;
    viscosity.yz = corr_data.correlations.back().yz + runningViscosity.yz;
    viscosity.zx = corr_data.correlations.back().zx + runningViscosity.zx;
    viscosity.xy *= factor;
    viscosity.yz *= factor;
    viscosity.zx *= factor;

    viscosityVector.push_back(viscosity);

    return viscosityVector;
}

// Function to print viscosity results
void printViscosity(const ShearViscosity &visc)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "Shear Viscosity Components:\n"
       << "    η_xy = " << visc.xy << "\n"
       << "    η_yz = " << visc.yz << "\n"
       << "    η_zx = " << visc.zx << "\n"
       << "    Average η = " << (visc.xy + visc.yz + visc.zx) / 3.0;
    log(ss.str());
}

int main()
{
    log("Program started");

    std::string filename = "../stress.dat";
    auto start_time = getTime();

    std::vector<StressData> stressData = readStressFile(filename);

    // Print only first 5 entries to avoid flooding console
    // if (!stressData.empty())
    // {
    //     log("Last few entries (showing 5 of " + std::to_string(stressData.size()) + "):");
    //     for (size_t i = stressData.size() - 5; i < stressData.size(); ++i)
    //     {
    //         std::stringstream ss;
    //         ss << "Entry " << i << ": Timestep = " << stressData[i].timestep
    //            << ", Pxy = " << stressData[i].pxy;
    //         log(ss.str());
    //     }
    // }
    int stepSize = 10;
    std::vector<int> dt_values = generateIntVector(0, 100000, stepSize);
    CorrelationData corrResult;
    corrResult.dt = dt_values;
    corrResult.correlations.reserve(dt_values.size());

    // Compute correlation for each dt
    for (size_t i = 0; i < dt_values.size(); ++i)
    {
        corrResult.correlations.push_back(computeAutocorrelation(stressData, dt_values[i]));
        if (i % 1000 == 0)
        {
            log("Computed correlation for dt = " + std::to_string(dt_values[i]));
        }
    }

    // Save results to file
    saveCorrelations(corrResult, "correlations.txt");

    // Print first few values as verification
    // log("First few correlation values:");
    // for (size_t i = 0; i < 5; ++i)
    // {
    //     std::stringstream ss;
    //     ss << std::fixed << std::setprecision(6);
    //     ss << "dt = " << corrResult.dt[i] << ": "
    //        << "xy = " << corrResult.correlations[i].xy << ", "
    //        << "yz = " << corrResult.correlations[i].yz << ", "
    //        << "zx = " << corrResult.correlations[i].zx;
    //     log(ss.str());
    // }

    // System parameters (these should be set according to your system)
    double V = pow(25, 3) * 1e-30; // System volume in m3
    double kBT = 4.141947e-21;     // kB*T in J
    double dt = stepSize * 1e-15;  // Time step in s
    // (V/kbT) * (dt * P**2)

    // Compute viscosity
    log("Computing shear viscosity...");
    std::vector<ShearViscosity> viscosityVector = computeViscosity(corrResult, V, kBT, dt);

    // Save results to file
    saveViscosity(viscosityVector, dt_values, "viscosity.txt");

    // Print results
    // printViscosity(viscosity);

    auto end_time = getTime();
    log("Total execution time: " + std::to_string(getTimeDiff(start_time, end_time)) + " seconds");
    return 0;
}