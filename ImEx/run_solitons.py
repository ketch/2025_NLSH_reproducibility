import os
import sys
import pickle

cwd = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from NLS_functions import *




def load_ini_conditions(n_solitons,m):
   
   # print("No of solitons in ini. cond. is",n_solitons)
    
    if n_solitons==2:
        q = 8; sol = 2; inv = 1
    if n_solitons==3:
        q = 18; sol = 3; inv = 1
    if n_solitons==1:
        q=2; sol=1; inv=1
    kppa = q

    if q == 8 and sol == 2 and inv == 1: 
    
        #print("2 soliton selected")
        xL = -16; xR = 16; L = xR-xL; m1 = 16; N =m1*L; t0 = 0; DT = [0.01,0.01]; SP_DT = [0.01,0.01]; T = 5
    elif q == 18 and sol == 3 and inv == 1: 
        
        #print("3 soliton selected")
        xL = -16; xR = 16; L = xR-xL; m1 = 32; N =m1*L; t0 = 0; DT = [0.01,0.01]; SP_DT = [0.01,0.01] ; T = 5

    elif q == 2 and sol == 1 and inv == 1: 
        #print("1 soliton selected")
        xL = -16; xR = 16; L = xR-xL;  T = 5


    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    def NLS_True_Sol(t,x,q):
        #N = len(x)
        if q == 2:
            ut = jnp.exp(1j*t)/jnp.cosh(x)
        elif q == 8:
            sechx = 1./jnp.cosh(x)
            ut = jnp.exp(1j*t)*sechx*( 1+(3/4)*sechx**2*(jnp.exp(8*1j*t)-1) )/( 1-(3/4)*sechx**4*jnp.sin(4*t)**2 )
        elif q == 18:
            ut = (2*(3*jnp.exp(t*25*1j)*jnp.exp(x) + 15*jnp.exp(t*9*1j)*jnp.exp(9*x) + 48*jnp.exp(t*25*1j)*jnp.exp(7*x) + 48*jnp.exp(t*25*1j)*jnp.exp(11*x) + 24*jnp.exp(t*33*1j)*jnp.exp(3*x) + 54*jnp.exp(t*33*1j)*jnp.exp(5*x) + 3*jnp.exp(t*25*1j)*jnp.exp(17*x) + 54*jnp.exp(t*33*1j)*jnp.exp(13*x) + 24*jnp.exp(t*33*1j)*jnp.exp(15*x) + 135*jnp.exp(t*41*1j)*jnp.exp(9*x) + 30*jnp.exp(t*49*1j)*jnp.exp(5*x) + 120*jnp.exp(t*49*1j)*jnp.exp(7*x) + 120*jnp.exp(t*49*1j)*jnp.exp(11*x) + 30*jnp.exp(t*49*1j)*jnp.exp(13*x) + 60*jnp.exp(t*57*1j)*jnp.exp(9*x)))/(3*(jnp.exp(t*24*1j) + 10*jnp.exp(6*x) + 10*jnp.exp(12*x) + 45*jnp.exp(t*8*1j)*jnp.exp(8*x) + 45*jnp.exp(t*8*1j)*jnp.exp(10*x) + 18*jnp.exp(t*16*1j)*jnp.exp(4*x) + 9*jnp.exp(t*24*1j)*jnp.exp(2*x) + 18*jnp.exp(t*16*1j)*jnp.exp(14*x) + 64*jnp.exp(t*24*1j)*jnp.exp(6*x) + 36*jnp.exp(t*24*1j)*jnp.exp(8*x) + 36*jnp.exp(t*24*1j)*jnp.exp(10*x) + 64*jnp.exp(t*24*1j)*jnp.exp(12*x) + 18*jnp.exp(t*32*1j)*jnp.exp(4*x) + 9*jnp.exp(t*24*1j)*jnp.exp(16*x) + jnp.exp(t*24*1j)*jnp.exp(18*x) + 18*jnp.exp(t*32*1j)*jnp.exp(14*x) + 45*jnp.exp(t*40*1j)*jnp.exp(8*x) + 45*jnp.exp(t*40*1j)*jnp.exp(10*x) + 10*jnp.exp(t*48*1j)*jnp.exp(6*x) + 10*jnp.exp(t*48*1j)*jnp.exp(12*x)))  
        
        
        return ut

    def exact_soln_real(t,x,q):
        return jnp.real(NLS_True_Sol(t,x,q))    
    def exact_soln_imag(t,x,q):
        return jnp.imag(NLS_True_Sol(t,x,q))    
    def exact_soln_np(t,x,q):
        return np.array(NLS_True_Sol(t,x,q))



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

    dx_sol_real=sol_real_x_vm(t_ini*np.ones_like(x),xj,q)
    dx_sol_imag=sol_imag_x_vm(t_ini*np.ones_like(x),xj,q)
    dx_sol = np.array(dx_sol_real)+1j*np.array(dx_sol_imag)

    dxx_sol_real=sol_real_xx_vm(t_ini*np.ones_like(x),xj,q)
    dxx_sol_imag=sol_imag_xx_vm(t_ini*np.ones_like(x),xj,q)
    dxx_sol = np.array(dxx_sol_real)+1j*np.array(dxx_sol_imag)

    sol_real = exact_soln_real(t_ini*np.ones_like(x),xj,q)
    sol_imag = exact_soln_imag(t_ini*np.ones_like(x),xj,q)
    sol = np.array(sol_real)+1j*np.array(sol_imag)

    amp = np.square(sol_real)+np.square(sol_imag)
    npsol = np.square(np.abs(exact_soln_np(np.zeros_like(x),x,q)))

    dx_soln_jx = [sol_real_x_vm,sol_imag_x_vm]

    return exact_soln_np,dx_soln_jx,sol,dx_sol,kppa,T,m,L



if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

    dt=0.0001     ## Choose dt
    m = 2048
    tau_list = [1.0/np.power(10.0,i) for i in range(5)]
    inv_list = [H,I1,I2]

    # Choose ImEx Scheme, options are None,a,b,c
    n_solitons = int(sys.argv[1])
    imex_sch = str(sys.argv[2])
    A_im,A_ex,C,b_im,b_ex,imex_stages = choose_imex(imex_sch)  

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex)


    # List of tau values for which u wanna run simulation
    
    
    frame_dict_list = []
    t_list = []
    #ini_type = "petviashvili" 
   
    ini_type = str(n_solitons)+"_solitons"
    #s1,c1=setup_tau(imx,dt,xi,0.001)
    for tau in tau_list:

        ## Options petviashvili,cubic
        ###  LOAD Initial conditions and grid setup i.e. x,xi,m,L,T,kppa,etc.
        
        exact_soln_np,dx_soln_jx,sol,dx_sol,kppa,T,m,L = load_ini_conditions(n_solitons,m)
        x = np.arange(-m/2,m/2)*(L/m)
        xi = np.fft.fftfreq(m)*m*2*np.pi/L
       # print("Yo",tau,xi.shape,L,40.0*np.pi,m)
            
       

        q0_ini  = sol
        q1_ini  = dx_sol
        u_ini = np.stack((q0_ini,q1_ini),axis=1)
        
        
        lmda_list = setup_tau(imx,dt,xi,tau)
        frm,tt,inv_change_dict,inv_fin,errs,mass_err_l,err_l = run_nls_hyper_example(dt,x,xi,kppa,T,tau,lmda_list,imx,inv_list,u_ini,exact_soln_np,dx_soln_jx,log_errs=True)
        frame_dict_list.append({"tau":tau,"ini_type":ini_type,"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"inv_final":inv_fin,"errors":errs,"x":x,"xi":xi,"kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l,"err_l":err_l})
        #print("inv_fin",inv_fin)



    case=ini_type+"_imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    save_dir = "./data/"+ini_type
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    file_name = save_dir+"/"+case+"_tau.pkl"
    #file_name="tets.pkl"
    # Save lists of dicts containing lists of frames & corresponding times for each tau in a file
    with open(file_name, 'wb') as f:
        pickle.dump(frame_dict_list,f)