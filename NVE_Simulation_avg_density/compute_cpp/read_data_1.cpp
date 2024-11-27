#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>

// Structure to hold each line of data
struct StressData
{
    int timestep;
    double pxx, pyy, pzz, pxy, pxz, pyz;
};

// Function to get current timestamp as string
std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
}

// Logger function
void log(const std::string &message)
{
    std::cout << "[" << getCurrentTimestamp() << "] " << message << std::endl;
}

// Function to read the file
std::vector<StressData> readStressFile(const std::string &filename)
{
    std::vector<StressData> data;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        log("Error: Unable to open file " + filename);
        return data;
    }

    log("Started reading file: " + filename);
    std::string line;

    // Skip the first two header lines
    std::getline(file, line);
    std::getline(file, line);

    // Read data lines
    while (std::getline(file, line))
    {
        StressData stress;
        std::istringstream iss(line);

        // Read each value from the line
        iss >> stress.timestep >> stress.pxx >> stress.pyy >> stress.pzz >> stress.pxy >> stress.pxz >> stress.pyz;

        if (iss) // If reading was successful
            data.push_back(stress);
    }

    log("Finished reading file. Total entries: " + std::to_string(data.size()));
    return data;
}

int main()
{
    log("Program started");

    bool run_test = false;
    std::string filename = "stress.dat";
    std::vector<StressData> stressData = readStressFile(filename);

    // Additional verification steps
    if (!stressData.empty() && run_test)
    {
        // Print first few entries
        log("Printing first few entries:");
        for (size_t i = 0; i < std::min(size_t(5), stressData.size()); ++i)
        {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(6); // Set precision for floating-point numbers
            ss << "Entry " << i << ": "
               << "Timestep = " << stressData[i].timestep << ", "
               << "Pxx = " << stressData[i].pxx << ", "
               << "Pyy = " << stressData[i].pyy << ", "
               << "Pzz = " << stressData[i].pzz << ", "
               << "Pxy = " << stressData[i].pxy << ", "
               << "Pxz = " << stressData[i].pxz << ", "
               << "Pyz = " << stressData[i].pyz;
            log(ss.str());
        }

        // Print last few entries
        log("Printing last few entries:");
        for (size_t i = std::max(size_t(0), stressData.size() - 5); i < stressData.size(); ++i)
        {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(6);
            ss << "Entry " << i << ": "
               << "Timestep = " << stressData[i].timestep << ", "
               << "Pxx = " << stressData[i].pxx << ", "
               << "Pyy = " << stressData[i].pyy << ", "
               << "Pzz = " << stressData[i].pzz << ", "
               << "Pxy = " << stressData[i].pxy << ", "
               << "Pxz = " << stressData[i].pxz << ", "
               << "Pyz = " << stressData[i].pyz;
            log(ss.str());
        }
    }
    if (!stressData.empty())
    {
        // Print some basic statistics
        log("File statistics:");
        log("Number of timesteps: " + std::to_string(stressData.size()));
        log("First timestep: " + std::to_string(stressData.front().timestep));
        log("Last timestep: " + std::to_string(stressData.back().timestep));
    }

    log("Program finished");
    return 0;
}