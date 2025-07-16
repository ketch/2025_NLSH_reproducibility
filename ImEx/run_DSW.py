import os
import sys
import pickle
import matplotlib.pyplot as plt

cwd = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from NLS_functions import *




def load_ini_conditions(delta,m):
   
    print("Delta chosen for initial Riemann Problem is",delta)
 
    L=2.0*800.0*2.0
    rho_l = 2.0
    rho_r = 1.0
    u_l = 0.0
    u_r = 0.0

    T = 70.0
    kppa = -1.0

    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    def NLS_True_Sol_old(t,x,kppa,delta=delta):
        #N = len(x)
        cnd_list = [x<0.0,x>=0.0]
        f_left = lambda x: jnp.sqrt(0.5*(rho_l+rho_r) - 0.5*(rho_l-rho_r)*jnp.tanh((x+L/4.0)/delta))*jnp.exp(1j*0.0)
        f_right = lambda x: jnp.sqrt(0.5*(rho_l+rho_r) + 0.5*(rho_l-rho_r)*jnp.tanh((x-L/4.0)/delta))*jnp.exp(1j*0.0)  
        ut = jnp.piecewise(x,cnd_list,[f_left,f_right])
             
        
        
        return ut
    
    def NLS_True_Sol(t,x,kppa,delta=delta):
        #N = len(x)
       
        
        ut = jnp.sqrt(0.5*(rho_l+rho_r) - 0.5*(rho_l-rho_r)*jnp.tanh((x)/delta))*jnp.exp(1j*0.0)
        
        
        return ut

    def exact_soln_real(t,x,kppa):
        return jnp.real(NLS_True_Sol_old(t,x,kppa))    
    def exact_soln_imag(t,x,kppa):
        return jnp.imag(NLS_True_Sol_old(t,x,kppa))    
    def exact_soln_np(t,x,kppa):
        return np.array(NLS_True_Sol_old(t,x,kppa))



    t_ini = 0.0
    
    xj = jnp.array(x)
    sol_real_x =  grad(exact_soln_real,1)
    sol_imag_x =  grad(exact_soln_imag,1)

    sol_real_xx =  grad(sol_real_x,1)
    sol_imag_xx =  grad(sol_imag_x,1)

    #gsol_real =  grad(exact_soln_real,1)
    #gsol_imag =  grad(exact_soln_imag,1)
    sol_real_x_vm =  jax.vmap(grad(exact_soln_real,1),(0,0,None))
    sol_imag_x_vm =  jax.vmap(grad(exact_soln_imag,1),(0,0,None))

    sol_real_xx_vm =  jax.vmap(grad(sol_real_x,1),(0,0,None))
    sol_imag_xx_vm =  jax.vmap(grad(sol_imag_x,1),(0,0,None))

    dx_sol_real=sol_real_x_vm(t_ini*np.ones_like(x),xj,kppa)
    dx_sol_imag=sol_imag_x_vm(t_ini*np.ones_like(x),xj,kppa)
    dx_sol = np.array(dx_sol_real)+1j*np.array(dx_sol_imag)

    dxx_sol_real=sol_real_xx_vm(t_ini*np.ones_like(x),xj,kppa)
    dxx_sol_imag=sol_imag_xx_vm(t_ini*np.ones_like(x),xj,kppa)
    dxx_sol = np.array(dxx_sol_real)+1j*np.array(dxx_sol_imag)

    sol_real = exact_soln_real(t_ini*np.ones_like(x),xj,kppa)
    sol_imag = exact_soln_imag(t_ini*np.ones_like(x),xj,kppa)
    sol = np.array(sol_real)+1j*np.array(sol_imag)

    amp = np.square(sol_real)+np.square(sol_imag)
    npsol = np.square(np.abs(exact_soln_np(np.zeros_like(x),x,kppa)))

    dx_soln_jx = [sol_real_x_vm,sol_imag_x_vm]

    return exact_soln_np,dx_soln_jx,sol,dx_sol,kppa,T,m,L



if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

    dt=0.0001     ## Choose dt
    m = 2048*4*2
    tau_list = [0.005,0.01]#[1.0/np.power(10.0,i) for i in range(13)]
    inv_list = [H,I1,I2]

    # Choose ImEx Scheme, options are None,a,b,c
    delta = float(sys.argv[1])
    imex_sch = str(sys.argv[2])
    A_im,A_ex,C,b_im,b_ex,imex_stages = choose_imex(imex_sch)  

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex)


    # List of tau values for which u wanna run simulation
    
    
    frame_dict_list = []
    t_list = []
    #ini_type = "petviashvili" 
   
    ini_type = str(delta)+"_DSW"
    #s1,c1=setup_tau(imx,dt,xi,0.001)

    exact_soln_np,dx_soln_jx,sol,dx_sol,kppa,T,m,L = load_ini_conditions(delta,m)
    q0_ini  = sol
    q1_ini  = dx_sol
    u_ini = np.stack((q0_ini,q1_ini),axis=1)

    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L
    print("Running NLS")
    frm,tt,inv_change_dict,inv_fin,errs,mass_err_l,err_l = run_nls_example(dt,x,xi,kppa,T,imx,inv_list,u_ini,exact_soln_np,dx_soln_jx,log_errs=True,lap_fac=0.5)
    frame_dict_list.append({"ini_type":ini_type,"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"inv_final":inv_fin,"errors":errs,"x":x,"xi":xi,"kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l,"err_l":err_l})

    for tau in tau_list[:]:

        ## Options petviashvili,cubic
        ###  LOAD Initial conditions and grid setup i.e. x,xi,m,L,T,kppa,etc.
        
        exact_soln_np,dx_soln_jx,sol,dx_sol,kppa,T,m,L = load_ini_conditions(delta,m)
        x = np.arange(-m/2,m/2)*(L/m)
        xi = np.fft.fftfreq(m)*m*2*np.pi/L
        #print("Yo",tau,xi.shape,L,40.0*np.pi,m,T)
        print("Running NLSH with tau=",tau)
        
       

        q0_ini  = sol
        q1_ini  = dx_sol
        u_ini = np.stack((q0_ini,q1_ini),axis=1)
        
        
        lmda_list = setup_tau(imx,dt,xi,tau)
        frm,tt,inv_change_dict,inv_fin,errs,mass_err_l,err_l = run_nls_hyper_example(dt,x,xi,kppa,T,tau,lmda_list,imx,inv_list,u_ini,exact_soln_np,dx_soln_jx,log_errs=True,lap_fac=0.5)
        frame_dict_list.append({"tau":tau,"ini_type":ini_type,"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"inv_final":inv_fin,"errors":errs,"x":x,"xi":xi,"kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l,"err_l":err_l})
        #print("inv_fin",inv_fin)



    case=ini_type+"_imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    save_dir = "./data/"+"U_"+ini_type
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    file_name = save_dir+"/"+case+"_tau.pkl"
    #file_name = save_dir+"/"+case+"_NLS_xtnd.pkl"
    #file_name="tets.pkl"
    # Save lists of dicts containing lists of frames & corresponding times for each tau in a file
    with open(file_name, 'wb') as f:
        pickle.dump(frame_dict_list,f)