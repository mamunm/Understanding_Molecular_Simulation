#!/bin/bash

# Enable extended globbing
shopt -s extglob

# Delete all files except those with .UPF, .sh, and .in extensions
rm -rv !(*.data|*.sh|in.lammps) > /dev/null 2>&1

# Disable extended globbing
shopt -u extglob