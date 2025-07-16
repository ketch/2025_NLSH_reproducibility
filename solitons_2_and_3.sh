#!/bin/bash
##################################################################
#Script for generating data and plotting Figure 5 (2-soliton and 3-soliton bound state solutions)
##################################################################


#This script generates the data for 2 and 3 soliton solutions of the NLS equation

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

#Change to the directory where the script is located
cd ImEx

#This script runs the code for 2 and 3 soliton solutions
#Syntax is python run_solitons.py <number_of_solitons> <ImEx_scheme>

#Run the 2-soltions case with "AGSA(3,4,2)" ImEx scheme
echo "#################################################"
echo "Expected run time for this script: 7 to 8 minutes"
echo "#################################################"
echo "Running code for 2-soliton case with AGSA(3,4,2) ImEx scheme"
"$PYTHON_COMMAND" run_solitons.py  2 "AGSA(3,4,2)"

#Run the 3-soltions case with "AGSA(3,4,2)" ImEx scheme
echo "Running code for 3-soliton case with AGSA(3,4,2) ImEx scheme"
"$PYTHON_COMMAND" run_solitons.py  3 "AGSA(3,4,2)"

#Plot the soliton solutions
echo "Plotting 2 and 3 soliton solutions with AGSA(3,4,2) ImEx scheme"
"$PYTHON_COMMAND" plot_solitons_2and3.py "AGSA(3,4,2)" 
