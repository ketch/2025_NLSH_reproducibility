#!/bin/bash

###############################################################
## Script to generate and plot data for Figure 7 (long-time error growth)
###############################################################

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

cd ImEx
#Generate the data by running:
#Syntax is python run_solitons_relax.py <number of solitons> <method>
echo "#################################################"
echo "Expected run time for this script: 3 to 4 minutes"
echo "#################################################"
echo "Running relaxation for 2 solitons with ARS(4,4,3)"
"$PYTHON_COMMAND" run_solitons_relax.py  2 "ARS(4,4,3)"
#Plot the Figure 7 using command:
echo "Plotting Figure 7"
"$PYTHON_COMMAND" plot_solitons_relax.py "ARS(4,4,3)" "ARS(4,4,3)" 2

