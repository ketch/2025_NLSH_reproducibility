#!/bin/bash

##############################################################################
# This script runs the NLSH time evolution code for the exact solitary wave problem
# to generate  Figure 6 (asymptotic accuracy)
################################################################################

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

# change to code directory
cd ImEx

#This generates initial conditions for the solitary wave problem
echo "#################################################"
echo "Expected run time for this script: 4 to 5 minutes"
echo "#################################################"
echo "Generating initial conditions for the exact solitary wave problem"
"$PYTHON_COMMAND" ini_conditions/generate_exact_solitary.py

#Then we run the NLSH time evolution code "run_exact_solitary_AA.py" with a given scheme and tau value
#The syntax is:
#python3 run_exact_solitary_AA.py  "scheme_name" tau_value
#Schemes used are:
#   AGSA(3,4,2), SSP3-ImEx(3,4,3), ARK3(2)4L[2]SA, ARS(4,4,3)
#tau values are:
#   1e-2, 1e-10


# AGSA(3,4,2) with tau = 1e-2
echo "Running exact solitary wave problem with AGSA(3,4,2) and tau = 1e-2"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "AGSA(3,4,2)" 1e-2
# AGSA(3,4,2) with tau = 1e-10
echo "Running exact solitary wave problem with AGSA(3,4,2) and tau = 1e-10"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "AGSA(3,4,2)" 1e-10

# SSP3-ImEx(3,4,3) with tau = 1e-2
echo "Running exact solitary wave problem with SSP3-ImEx(3,4,3) and tau = 1e-2"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "SSP3-ImEx(3,4,3)" 1e-2
# SSP3-ImEx(3,4,3) with tau = 1e-10
echo "Running exact solitary wave problem with SSP3-ImEx(3,4,3) and tau = 1e-10"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "SSP3-ImEx(3,4,3)" 1e-10

# ARK3(2)4L[2]SA with tau = 1e-2
echo "Running exact solitary wave problem with ARK3(2)4L[2]SA and tau = 1e-2"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "ARK3(2)4L[2]SA" 1e-2
# ARK3(2)4L[2]SA with tau = 1e-10
echo "Running exact solitary wave problem with ARK3(2)4L[2]SA and tau = 1e-10"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "ARK3(2)4L[2]SA" 1e-10

# ARS(4,4,3) with tau = 1e-2
echo "Running exact solitary wave problem with ARS(4,4,3) and tau = 1e-2"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "ARS(4,4,3)" 1e-2
# ARS(4,4,3) with tau = 1e-10
echo "Running exact solitary wave problem with ARS(4,4,3) and tau = 1e-10"
"$PYTHON_COMMAND" run_exact_solitary_AA.py  "ARS(4,4,3)" 1e-10



# Finally, we plot the results using the script "plot_exact_solitary_AA.py"
#The syntax is:
#"$PYTHON_COMMAND" plot_exact_solitary_AA.py "scheme1" "scheme2" "scheme3" "scheme4"
#where scheme1, scheme2, scheme3, and scheme4 are the names of the schemes used in the previous step.

"$PYTHON_COMMAND" plot_exact_solitary_AA.py "AGSA(3,4,2)" "SSP3-ImEx(3,4,3)"  "ARK3(2)4L[2]SA" "ARS(4,4,3)"
