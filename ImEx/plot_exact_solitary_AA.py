import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from plot_scripts import *
from run_ground_state_petv_AA import load_ini_conditions





if __name__=="__main__":

    plot_dir = "../figures/"
    if not(os.path.exists(plot_dir)):
        os.makedirs(plot_dir)

    ############################  varying dt with two schemes (a and c ) ############################ 
    ini_type = "exact_solitary"
    sym_list = ["r-*","g-^","b-o","m-s"]
    n_schemes = len((sys.argv[1]))-1
    im_schemes_list=[]

    tau_1 = 0.01
    tau_2 = 1e-10

    data_dir_tau1 = "./data/"+ini_type+"/tau_"+str(tau_1)+"/"
    data_dir_tau2 = "./data/"+ini_type+"/tau_"+str(tau_2)+"/"
    

    for sch in (sys.argv[1:]):
        im_schemes_list.append(str(sch))
  

    

    fig = plt.figure(figsize=[20,20])
    fig.suptitle(r"NLSH Standing Solitary Waves: Errors",fontsize=20)

    
    axul = fig.add_subplot(221)
    axul.set_title(r"$\tau=$"+str(tau_1),fontsize=20)
    axul.tick_params(axis='both', which='major', labelsize=15)
    axul.set_yscale("log")
    axul.set_xlabel(r"$\Delta t$",fontsize=20)
    axul.set_ylabel("$||q_0-q_0^{ex}||_2$",fontsize=20)
    

    axur = fig.add_subplot(222)
    axur.set_title(r"$\tau=$"+str(tau_2),fontsize=20)
    axur.tick_params(axis='both', which='major', labelsize=15)
    axur.set_yscale("log")
    axur.set_xlabel(r"$\Delta t$",fontsize=20)
    axur.set_ylabel("$||q_0-q_0^{ex}||_2$",fontsize=20)

    axll = fig.add_subplot(223)
    axll.set_title(r"$\tau=$"+str(tau_1),fontsize=20)
    axll.tick_params(axis='both', which='major', labelsize=15)
    axll.set_yscale("log")
    axll.set_xlabel(r"$\Delta t$",fontsize=20)
    axll.set_ylabel("$||q_1-q_1^{ex}||_2$",fontsize=20)

    axlr = fig.add_subplot(224)
    axlr.set_title(r"$\tau=$"+str(tau_2),fontsize=20)
    axlr.tick_params(axis='both', which='major', labelsize=15)
    axlr.set_yscale("log")
    axlr.set_xlabel(r"$\Delta t$",fontsize=20)
    axlr.set_ylabel("$||q_1-q_1^{ex}||_2$",fontsize=20)

     
    #dt_list = [1.0/np.power(10.0,i) for i in range(1,5)]
    m = 2048

    

    for sch_ind,im_sch in enumerate(im_schemes_list):
        # if sch_ind==0:
        #     print("Data dir tau1",data_dir_tau1)
        #     print("Data dir tau2",data_dir_tau2)
        dt_list = [1.0/np.power(2.0,i) for i in range(3,13)]
        if im_sch=="AGSA(3,4,2)":
            dt_list = [1.0/np.power(2.0,i) for i in range(7,13)] 

        dt_file_list_tau1 = [data_dir_tau1+ini_type+"_imex_"+im_sch+"_"+str(m)+"_"+str(dt)+"_tau.pkl" for dt in dt_list]
        dt_file_list_tau2 = [data_dir_tau2+ini_type+"_imex_"+im_sch+"_"+str(m)+"_"+str(dt)+"_tau.pkl" for dt in dt_list]
        fdict_list_tau1 = []
        fdict_list_tau2 = []
        for file_nm in dt_file_list_tau1:
           # print("AAAA",file_nm)
            with open(file_nm, "rb") as f:
                fdict_list_tau1.append(pickle.load(f))
            #print(len(fdict_list_tau1),fdict_list_tau1[-1][0]["tau"],fdict_list_tau1[-1][0]["errors"][0],fdict_list_tau1[-1][0]["errors"][2])
            tau_1 = fdict_list_tau1[-1][0]["tau"]


        for file_nm in dt_file_list_tau2:
           # print("AAAA",file_nm)
            with open(file_nm, "rb") as f:
                fdict_list_tau2.append(pickle.load(f))
            #print(len(fdict_list_tau2),fdict_list_tau2[-1][0]["tau"],fdict_list_tau2[-1][0]["errors"][0],fdict_list_tau2[-1][0]["errors"][2])
            tau_2 = fdict_list_tau2[-1][0]["tau"]

        err_L1_dtlist_c1_tau1=[]
        err_L2_dtlist_c1_tau1=[]
        err_Linf_dtlist_c1_tau1=[]

        err_L1_dtlist_c2_tau1=[]
        err_L2_dtlist_c2_tau1=[]
        err_Linf_dtlist_c2_tau1=[]

        err_L1_dtlist_c1_tau2=[]
        err_L2_dtlist_c1_tau2=[]
        err_Linf_dtlist_c1_tau2=[]

        err_L1_dtlist_c2_tau2=[]
        err_L2_dtlist_c2_tau2=[]
        err_Linf_dtlist_c2_tau2=[]

        for dt,f in zip(dt_list,fdict_list_tau1):
            err_L1_c1 = f[0]["errors"][0]
            err_L2_c1 = f[0]["errors"][1]
            err_Linf_c1 = f[0]["errors"][2]

            err_L1_c2 = f[0]["errors"][3]
            err_L2_c2 = f[0]["errors"][4]
            err_Linf_c2 = f[0]["errors"][5]

            #rel_H_dtlist.append(f[0]["inv_change_dict"][change_type][0])
            err_L1_dtlist_c1_tau1.append(err_L1_c1)
            err_L2_dtlist_c1_tau1.append(err_L2_c1)
            err_Linf_dtlist_c1_tau1.append(err_Linf_c1)

            err_L1_dtlist_c2_tau1.append(err_L1_c2)
            err_L2_dtlist_c2_tau1.append(err_L2_c2)
            err_Linf_dtlist_c2_tau1.append(err_Linf_c2)
        #print("Tau 1 done")

        for dt,f in zip(dt_list,fdict_list_tau2):
            #print("tau2 dt",dt)
            err_L1_c1 = f[0]["errors"][0]
            err_L2_c1 = f[0]["errors"][1]
            err_Linf_c1 = f[0]["errors"][2]

            err_L1_c2 = f[0]["errors"][3]
            err_L2_c2 = f[0]["errors"][4]
            err_Linf_c2 = f[0]["errors"][5]

            #rel_H_dtlist.append(f[0]["inv_change_dict"][change_type][0])
            err_L1_dtlist_c1_tau2.append(err_L1_c1)
            err_L2_dtlist_c1_tau2.append(err_L2_c1)
            err_Linf_dtlist_c1_tau2.append(err_Linf_c1)

            err_L1_dtlist_c2_tau2.append(err_L1_c2)
            err_L2_dtlist_c2_tau2.append(err_L2_c2)
            err_Linf_dtlist_c2_tau2.append(err_Linf_c2)

        #print(len(err_L1_dtlist_c1_tau1),len(err_L1_dtlist_c1_tau2),len(err_L1_dtlist_c2_tau1),len(err_L1_dtlist_c2_tau2))
        try:
            axul.loglog((dt_list),err_L2_dtlist_c1_tau1,sym_list[sch_ind],label=im_sch)
            axur.loglog((dt_list),err_L2_dtlist_c1_tau2,sym_list[sch_ind],label=im_sch)

            axll.loglog((dt_list),err_L2_dtlist_c2_tau1,sym_list[sch_ind],label=im_sch)
            axlr.loglog((dt_list),err_L2_dtlist_c2_tau2,sym_list[sch_ind],label=im_sch)
            #print("diff",np.mean(np.abs(np.array(err_L2_dtlist_c1_tau1)-np.array(err_L2_dtlist_c2_tau1))))
            #print("diff",np.mean(np.abs(np.array(err_L2_dtlist_c1_tau2)-np.array(err_L2_dtlist_c2_tau2))))
            if sch_ind==3:
                axul.loglog((dt_list)[2:-2],20.0*np.square(np.array(dt_list))[2:-2],"k--",linewidth=2,label=r"$~O(\Delta t)^2$")
                axul.loglog((dt_list)[2:-2],5.0*np.power(np.array(dt_list),3.0)[2:-2],"k:",linewidth=4,label=r"$~O(\Delta t)^3$")

                axur.loglog((dt_list)[2:-2],20.0*np.square(np.array(dt_list))[2:-2],"k--",linewidth=2,label=r"$~O(\Delta t)^2$")
                axur.loglog((dt_list)[2:-2],5.0*np.power(np.array(dt_list),3.0)[2:-2],"k:",linewidth=4,label=r"$~O(\Delta t)^3$")

                axll.loglog((dt_list)[2:-2],20.0*np.square(np.array(dt_list))[2:-2],"k--",linewidth=2,label=r"$~O(\Delta t)^2$")
                axll.loglog((dt_list)[2:-2],5.0*np.power(np.array(dt_list),3.0)[2:-2],"k:",linewidth=4,label=r"$~O(\Delta t)^3$")

                axlr.loglog((dt_list)[2:-2],20.0*np.square(np.array(dt_list))[2:-2],"k--",linewidth=2,label=r"$~O(\Delta t)^2$")
                axlr.loglog((dt_list)[2:-2],5.0*np.power(np.array(dt_list),3.0)[2:-2],"k:",linewidth=4,label=r"$~O(\Delta t)^3$")
               
        except:
            print("Error")

   
    
    

    
    




    

    

    axul.legend(frameon=False,loc="best",fontsize=20)
    axur.legend(frameon=False,loc="best",fontsize=20)
    axll.legend(frameon=False,loc="best",fontsize=20)
    axlr.legend(frameon=False,loc="best",fontsize=20)


    fig.savefig(plot_dir+"/exact_solitary_AA_m2m10.pdf")