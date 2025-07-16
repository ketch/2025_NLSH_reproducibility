import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from matplotlib.gridspec import GridSpec

from plot_scripts import *
from run_solitons_relax import load_ini_conditions


if __name__=="__main__":

    plot_dir = "../figures/"
    if not(os.path.exists(plot_dir)):
        os.makedirs(plot_dir)
 
    im_sch = str(sys.argv[1])
    im_sch2 = str(sys.argv[2])
    n_solitons = int(sys.argv[3])

    xlim=20.0

    ini_type_s = str(n_solitons)+"_solitons"
    

    soliton_data_dir_s = "./data/"+ini_type_s+"/"
   



    
    exact_soln_np_s,dx_soln_jx_s,sol_s,dx_sol_s,kppa_s,T_s,m_s,L_s = load_ini_conditions(2,2048)

    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch+"_2048_0.01_tau.pkl"
    with open(filename, 'rb') as f:
            frame_dict_list = pickle.load(f)
    
    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch+"_2048_0.01_tau_relax.pkl"
    with open(filename, 'rb') as f:
            frame_dict_list_relax = pickle.load(f)

    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch+"_2048_0.01_tauNLS.pkl"
    with open(filename, 'rb') as f:
            frame_dictNLS = pickle.load(f)[0]
    
    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch+"_2048_0.01_tau_relaxNLS.pkl"
    with open(filename, 'rb') as f:
            frame_dictNLS_relax = pickle.load(f)[0]

    errNLS_l = np.array(frame_dictNLS["err_l"])
    mass_errNLS_l = frame_dictNLS["mass_err_l"]
    t_NLS = frame_dictNLS["t_list"]

    errNLS_l_r = np.array(frame_dictNLS_relax["err_l"])
    mass_errNLS_l_r = frame_dictNLS_relax["mass_err_l"]
    t_NLS_r = frame_dictNLS_relax["t_list"]

   


    fig = plt.figure(figsize=(20, 15),dpi=160)
    spec = GridSpec(ncols=2, nrows=2,  hspace=0.35,figure=fig)

    tau_list = [f["tau"] for f in frame_dict_list ]
    #print("tau list",tau_list)

    for i,tau in enumerate(tau_list):
        ax = fig.add_subplot(spec[i//2, i%2])
        ax.set_title(r"$\tau=$"+str(tau),fontsize=20)
        ax.set_ylabel("Error in u",fontsize=20)
        ax.set_xlabel("t",fontsize=20)
        #print(i//2,i%2)

        
        #print(tau,frame_dict_list_relax[i].keys())
        #exact_list = [exact_soln_np_s,dx_soln_jx_s]
        err_l = np.array(frame_dict_list[i]["err_l"])
        mass_err_l = frame_dict_list[i]["mass_err_l"]

        err_l_r = np.array(frame_dict_list_relax[i]["err_l"])
        mass_err_l_r = frame_dict_list_relax[i]["mass_err_l"]

        t_l = frame_dict_list[i]["t_list"]
        t_l_r = frame_dict_list_relax[i]["t_list"]
        #print(len(t_l),err_l.shape,err_l_r.shape)
        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_xlim([0.01,xlim])
        #ax.plot(t_l,np.abs(err_l[:,-1]-err_l_r[:,-1]),"ro")

        ax.plot(t_NLS,errNLS_l[:,0],"k-",label="NLS baseline")
        ax.plot(t_NLS_r,errNLS_l_r[:,0],"k--",label="NLS relaxation")

        ax.plot(t_l,err_l[:,0],"b-",label="NLSH baseline")
        ax.plot(np.real(t_l_r),err_l_r[:,0],"b--",label="NLSH relaxation")
        
        #ax.plot(t_l,err_l_r[:,0],"b*",label="relax")
    
        #file_list = [filename_s]
    

        #
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(loc="best",fontsize=20)
    fig.savefig("../figures/"+im_sch+"4x4_relax.pdf")




    fig = plt.figure(figsize=(20, 10),dpi=160)
    spec = GridSpec(ncols=2, nrows=1,  hspace=0.35,figure=fig)

    tau = 0.001
    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch2+"_2048_0.01_tau.pkl"
    with open(filename, 'rb') as f:
            frame_dict_list2 = pickle.load(f)
    
    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch2+"_2048_0.01_tau_relax.pkl"
    with open(filename, 'rb') as f:
            frame_dict_list_relax2 = pickle.load(f)

    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch2+"_2048_0.01_tauNLS.pkl"
    with open(filename, 'rb') as f:
            frame_dictNLS2 = pickle.load(f)[0]
    
    filename= soliton_data_dir_s+ini_type_s+"_imex_"+im_sch2+"_2048_0.01_tau_relaxNLS.pkl"
    with open(filename, 'rb') as f:
            frame_dictNLS_relax2 = pickle.load(f)[0]

    j = np.argmin([np.abs(tau-tau_c) for tau_c in tau_list])
   
    #print("j" , j,i,frame_dict_list[j]["tau"],frame_dict_list2[j]["tau"])

    err_l = np.array(frame_dict_list[j]["err_l"])
    mass_err_l = frame_dict_list[j]["mass_err_l"]

    err_l_r = np.array(frame_dict_list_relax[j]["err_l"])
    mass_err_l_r = frame_dict_list_relax[j]["mass_err_l"]

    t_l = frame_dict_list[j]["t_list"]
    t_l_r = frame_dict_list_relax[j]["t_list"]

    errNLS_l2 = np.array(frame_dictNLS2["err_l"])
    mass_errNLS_l2 = frame_dictNLS2["mass_err_l"]
    t_NLS2 = frame_dictNLS2["t_list"]

    errNLS_l_r2 = np.array(frame_dictNLS_relax2["err_l"])
    mass_errNLS_l_r2 = frame_dictNLS_relax2["mass_err_l"]
    t_NLS_r2 = frame_dictNLS_relax2["t_list"]

    err_l2 = np.array(frame_dict_list2[j]["err_l"])
    mass_err_l2 = frame_dict_list2[j]["mass_err_l"]

    err_l_r2 = np.array(frame_dict_list_relax2[j]["err_l"])
    mass_err_l_r2 = frame_dict_list_relax2[j]["mass_err_l"]

    t_l2 = frame_dict_list2[j]["t_list"]
    t_l_r2 = frame_dict_list_relax2[j]["t_list"]

    ax00 = fig.add_subplot(spec[0, 0])
    ax00.set_title(im_sch2,fontsize=20)
    ax00.set_ylabel("Error in u",fontsize=20)
    ax00.set_xlabel("t",fontsize=20)

    ax00.set_yscale("log")
    ax00.set_xscale("log")

    ax00.set_xlim([0.01,xlim])
    #print("err_l shapes",err_l.shape,err_l2.shape,err_l_r.shape,err_l_r2.shape)

    ax00.plot(t_l2,err_l2[:,0],"b-",label="NLSH("+str(tau)+")baseline")
    ax00.plot(np.real(t_l_r2),err_l_r2[:,0],"b--",label="NLSH("+str(tau)+")relaxation")
    ax00.tick_params(axis='both', which='major', labelsize=20)

    ax00.legend(loc="best",fontsize=20)

    ax01 = fig.add_subplot(spec[0, 1])
    ax01.set_title(im_sch,fontsize=20)
    ax01.set_ylabel("Error in u",fontsize=20)
    ax01.set_xlabel("t",fontsize=20)

    ax01.set_yscale("log")
    ax01.set_xscale("log")

    ax01.set_xlim([0.01,xlim])
    

    ax01.plot(t_l,err_l[:,0],"b-",label="NLSH("+str(tau)+")baseline")
    ax01.plot(np.real(t_l_r),err_l_r[:,0],"b--",label="NLSH("+str(tau)+")relaxation")
    ax01.tick_params(axis='both', which='major', labelsize=20)

    ax01.legend(loc="best",fontsize=20)

    fig.savefig(plot_dir+"/"+im_sch2+"_"+im_sch+"1x2_relax.pdf")
    #figr_test.savefig(plot_dir+"soltions_23_solutions_"+im_sch+".png")


 