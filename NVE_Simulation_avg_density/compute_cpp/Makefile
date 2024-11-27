# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O3 -std=c++20

# Source file
SOURCE = compute_viscosity.cpp

# Output executable
TARGET = compute_viscosity

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean up
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
