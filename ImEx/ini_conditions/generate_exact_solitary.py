import numpy as np
import pickle

import os
this_script_path = os.path.dirname(os.path.abspath(__file__))

ifft = lambda q: np.fft.ifft(q).real
fft = lambda q: np.fft.fft(q)


m= 2**15
dom_length = 10*np.pi#2 * np.pi
xi = np.fft.fftfreq(m)*m*2*np.pi/dom_length
d1 = 1.j*xi
x = np.arange(-m/2,m/2)*(dom_length/m)
kappa = 1.
mu = 3.

def get_q1(q0,mu,tau):
    return (1./(1-mu*tau))*ifft(d1*np.fft.fft(q0))

sech = lambda x: 1./np.cosh(x)
def arcsech(x):
    return np.log((1 + np.sqrt(1 - x**2)) / x)

def solitary_NLS(kappa=1,mu=1,u0=1,sign=1):
    R = np.sqrt(2*mu/kappa)
    C = arcsech(u0/R)
    return lambda x: R*sech(np.sqrt(mu)*x+sign*C)

def solitary_NLSH(tau=1.,kappa=1,mu=1.,u0=1.,sign=1):
    R = np.sqrt(2*mu/kappa)
    u0 = R #Override u0 with R
    C = arcsech(u0/R)
    return lambda x: R*sech(np.sqrt(mu*(1-mu*tau))*x+sign*C)

tau_list = [0.0001,0.001,0.0078125,
            0.015625,0.01,0.03125,
            0.0625,0.125,0.1,0.25,
            1e-05,1e-06,1e-07,
            1e-08,1e-09,1e-10,1e-11,1e-12]

#Save the solution to a file for further use as initial conditions for ImEx codes
for tau in tau_list:
    print(f"Generating solitary wave solution for tau={tau}")
    q0 = solitary_NLSH(tau=tau,kappa=kappa,mu=mu,u0=1.,sign=1)(x)
    q1 = get_q1(q0,mu,tau)
    pit_sol = {"v":q0,"p":q1,"x":x,"xi":xi,"kppa":kappa,"mu":mu,"m":m,"L":dom_length}
    file_name = f"{this_script_path}/exact_solitary_"+str(tau)+"_ini.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(pit_sol,f)