# Reproducibility
This repository contains code and instructions for reproducing the computational results
presented in

> *A Hyperbolic Approximation of the Nonlinear Schrodinger Equation*, by A. Biswas, L. S. Busaleh,
C. Munoz-Moncayo, and Manvendra Rajvanshi.

The preprint can be found at https://arxiv.org/pdf/2505.21424.

## Requirements
The required Python packages can be found in `requirements.txt` and installed by running, e.g.,

 `pip install -r requirements.txt`
 
 To reproduce the results in Section 5, `pyfftw` and, therefore, `fftw` (see https://fftw.org) are required.

## Figures 1 (traveling wave phase portraits), 2 (standing fronts), and 3 (standing fronts)
Run the command: 

`python traveling-waves/generate_plots_Section3.py`


## Figure 5 (2-soliton and 3-soliton bound state solutions)
Run the script solitons_2_and_3.sh using following command    
 `bash ./solitons_2_and_3.sh `      
It runs a set of programs to generate data and a code to plot the figure.



## Figure 6 (asymptotic accuracy)
Run the script asymptotic_accuracy.sh using following command    
 `bash ./asymptotic_accuracy.sh`


## Figure 7 (long-time error growth)
Run the script relax_long_time.sh  using following command    
 `bash ./relax_long_time.sh ` 


## Figure 8 (Riemann problem)
Run the script DSW.sh  using following command    
 `bash ./DSW.sh `    
