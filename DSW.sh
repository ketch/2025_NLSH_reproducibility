#!/bin/bash

###################################################################
#Script to generate data and plot Figure 8 (Riemann problem)
###################################################################

# Check if 'python3' command exists and is executable
if command -v python3 &>/dev/null; then
    PYTHON_COMMAND="python3"
# If 'python3' is not found, check for 'python'
elif command -v python &>/dev/null; then
    PYTHON_COMMAND="python"
else
    echo "Error: Neither 'python3' nor 'python' command found. Please install Python."
    exit 1
fi

#Go to code directory
cd ImEx
# Generate the data:
echo "Running the code to generate data for Figure 8 (DSW Riemann problem)"
echo "#################################################"
echo "This might take long time (approx. 160-180 minutes)"
echo "#################################################"
"$PYTHON_COMMAND" run_DSW.py 0.01 "SSP3-ImEx(3,4,3)"
# Here 0.01 is the value of \delta(smoothening parameter for Riemann problem)

#Plot the figure 8
"$PYTHON_COMMAND" plot_DSW.py 
