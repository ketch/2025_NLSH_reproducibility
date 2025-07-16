import os
import sys
import pickle

cwd = os.getcwd()#"/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from NLS_functions import *




def load_ini_conditions(tau,skip=16):
    #m = 1024
    T=5.0
    tau_name = str(tau)

    cwd = os.getcwd()
    cwd = cwd+"/ini_conditions"
    print(cwd)
    
    file_name=cwd+"/Petviashvilil_"+tau_name+"_ini.pkl"
    
    with open(file_name, 'rb') as f:
        fdict = pickle.load(f)
    v = fdict["v"]
    p = fdict["p"]
    #x = fdict["x"]
    #xi = fdict["xi"]
    kppa = fdict["kppa"]
    mu = fdict["mu"]
    L = fdict["L"]

    #m = v.shape[0]
    #skip = 16#  # or 32
    m = int(2**15/skip)
    print("m is ",m)
    

    sol  = np.exp(1j*mu*0.0)*v
    dx_sol = np.exp(1j*mu*0.0)*p
    #skip = int(sol.shape[0]/m)

    sol = sol[::skip]
    dx_sol = dx_sol[::skip]
    #x = x[::skip]
    #xi= xi[::skip]

    v = v[::skip]
    p = p[::skip]



    def exact_soln_np(t,x,kppa,v=v,p=p):
        exp_fac = np.exp(1j*mu*t)
        q0 = exp_fac*v
        q1 = exp_fac*p
        return np.stack([q0,q1],-1)
    
    print("Ini Pits",file_name,"tau name",tau_name)

    return exact_soln_np,sol,dx_sol,kppa,mu,T,m,L

if __name__=="__main__":
    print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

    dt=0.0001     ## Choose dt
    tau = 0.00001

    # Choose ImEx Scheme, options are None,a,b,c
    imex_sch = str(sys.argv[1])
    #dt_arg = float(sys.argv[2])
    #print("dt arg",dt_arg,dt)
    #dt = 1.0/np.power(2.0,dt_arg)
    #print("dt arg",dt_arg,dt)
    A_im,A_ex,C,b_im,b_ex,imex_stages = choose_imex(imex_sch)  

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex)


    # List of tau values for which u wanna run simulation
    #tau_list = [1.0/np.power(10.0,i) for i in range(1,13)] 
    dt_list = [1.0/np.power(2.0,i) for i in range(3,13)]
    var_dt = True
    var_tau = False
    inv_list = [H,I1,I2]
    frame_dict_list = []
    t_list = []
    #ini_type = "petviashvili" 
    ini_type = "petviashvili"
    #s1,c1=setup_tau(imx,dt,xi,0.001)
    #for tau in tau_list:
    
    for dt in dt_list:

        ## Options petviashvili,cubic
        ###  LOAD Initial conditions and grid setup i.e. x,xi,m,L,T,kppa,etc.
        if var_dt:
            frame_dict_list=[]
        
        exact_soln_np,sol,dx_sol,kppa,mu,T,m,L = load_ini_conditions(tau,skip=16)
        x = np.arange(-m/2,m/2)*(L/m)
        xi = np.fft.fftfreq(m)*m*2*np.pi/L
        print("Yo",tau,xi.shape,L,40.0*np.pi,m,dt)
            
       

        q0_ini  = sol
        q1_ini  = dx_sol
        u_ini = np.stack((q0_ini,q1_ini),axis=1)
        
        
        lmda_list = setup_tau(imx,dt,xi,tau)
        
        frm,tt,inv_change_dict,inv_fin,errs,mass_err_l,err_l = run_nls_hyper_example(dt,x,xi,kppa,T,tau,lmda_list,imx,inv_list,u_ini,exact_soln_np,None,log_errs=True)
        frame_dict_list.append({"tau":tau,"ini_type":ini_type,"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"inv_final":inv_fin,"errors":errs,"x":x,"xi":xi,"kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l,"err_l":err_l})
        print("inv_fin",inv_fin,len(frame_dict_list))

        if var_dt:

            case=ini_type+"_imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
            save_dir = "./data/"+ini_type
            save_dir = save_dir+"/tau_"+str(tau)
            if not(os.path.exists(save_dir)):
                os.makedirs(save_dir)
            file_name = save_dir+"/"+case+"_tau.pkl"
            #file_name="tets.pkl"
            # Save lists of dicts containing lists of frames & corresponding times for each tau in a file
            with open(file_name, 'wb') as f:
                pickle.dump(frame_dict_list,f)


    if  var_tau:

        case=ini_type+"_imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
        save_dir = "./data/"+ini_type
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)
        file_name = save_dir+"/"+case+"_tau.pkl"
        #file_name="tets.pkl"
        # Save lists of dicts containing lists of frames & corresponding times for each tau in a file
        with open(file_name, 'wb') as f:
            pickle.dump(frame_dict_list,f)