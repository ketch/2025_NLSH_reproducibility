import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

#from plot_scripts import *
from run_solitons import load_ini_conditions

linewidth=2

def plot_sol_at_t_diff_tau(fname,ax1,i=-1,exact_soln_np=None,x_lim=None,y_lim=None,plottype="abs",subtitle=None,skip=None):

        tau_list = []
        u_list = []
        t_list=[]
        if type(fname)==list:
             frame_dict_list =fname
             
                  
           
        elif type(fname)==str:
             with open(fname, 'rb') as f:
                frame_dict_list=pickle.load(f)
             
        for f in frame_dict_list:
            tau_list.append(f['tau'])
            t_list.append(f["t_list"][i])
            u_list.append(f["frame_list"][i])
        
        if type(skip) is list:
            index_list = skip
        elif type(skip) is str:
            index_list = [int(i) for i in skip.split(",")]
        else:
             if skip==None:
                skip = 1
             index_list =[i for i in range(0,len(tau_list),skip)] 
        tau_list_new =[tau_list[i] for i in index_list] 
        t_list_new =[t_list[i] for i in index_list] 
        u_list_new =[u_list[i] for i in index_list] 
        
        tau_list = tau_list_new
        t_list = t_list_new
        u_list = u_list_new
        
            

       

        t = t_list[0]
        #print("t_list",t_list)
        
        x = frame_dict_list[0]['x']
        xi = frame_dict_list[0]['xi']
        kppa = frame_dict_list[0]['kappa']
        if exact_soln_np!=None:
                u_sol = exact_soln_np(t*np.ones_like(x),x,kppa)
                #print(u_sol.shape)

        n=len(t_list)
        

        
        if plottype=="abs":
              ax1.set_ylabel("$|q_0|$",fontsize=20)
        elif plottype=="real":
            ax1.set_ylabel("$Re(q_0)$",fontsize=20)
        elif plottype=="imag":
            ax1.set_ylabel("$Im(q_0)$",fontsize=20)

        ax1.set_xlabel("x",fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
      
        mrk_list = [":b","--r","-.g"]
       

        if x_lim!=None:
            ax1.set_xlim(x_lim)
        
        if y_lim!=None:
            ax1.set_ylim(y_lim)
           
        
        if exact_soln_np!=None:
                    
                        
                    if plottype=="abs":
                        ax1.plot(x,np.abs(u_sol[:]),"-k",linewidth=linewidth,label="NLS")
                    elif plottype=="real":
                       ax1.plot(x,np.real(u_sol[:]),"-k",linewidth=linewidth,label="NLS")
                    elif plottype=="imag":
                        ax1.plot(x,np.imag(u_sol[:]),"-k",linewidth=linewidth,label="NLS")
        

        tau_list.reverse()
        u_list.reverse()          
        for j,tau in enumerate(tau_list):
           # print("tau",tau)
                
            if j<3:
                if plottype=="abs":
                    ax1.plot(x,np.abs(u_list[j][:,0]),mrk_list[j],linewidth=linewidth,label=r"$\tau=$"+str(tau))
                elif plottype=="real":
                    ax1.plot(x,np.real(u_list[j][:,0]),mrk_list[j],linewidth=linewidth,label=r"$\tau=$"+str(tau))
                elif plottype=="imag":
                    ax1.plot(x,np.imag(u_list[j][:,0]),mrk_list[j],linewidth=linewidth,label=r"$\tau=$"+str(tau))
            else:
                if plottype=="abs":
                    ax1.plot(x,np.abs(u_list[j][:,0]),color="m",linestyle=(0, (3, 1, 1, 1, 1, 1)),linewidth=linewidth,label=r"$\tau=$"+str(tau))
                elif plottype=="real":
                    ax1.plot(x,np.real(u_list[j][:,0]),color="m",linestyle=(0, (3, 1, 1, 1, 1, 1)),linewidth=linewidth,label=r"$\tau=$"+str(tau))
                elif plottype=="imag":
                    ax1.plot(x,np.imag(u_list[j][:,0]),color="m",linestyle=(0, (3, 1, 1, 1, 1, 1)),linewidth=linewidth,label=r"$\tau=$"+str(tau))
                
                


                

        ax1.legend(loc="best",fontsize=20)
        if subtitle!=None:
            ax1.set_title(subtitle,fontsize=20)

        
        return ax1,t



def plot_axb_sol_at_t_diff_tau(fname_list,a=1,b=2,i=-1,exact_soln_np_list=None,x_lim=None,y_lim=None,plottype="abs",subtitles=None):
     
    fig = plt.figure(figsize=[20,10])
    axnmbr = a*100+b*10
    ax = []
    skip_list = ["1,2,3,4","1,2,3,4"]#[[1,3,5],[3,5,6]]
    for j,fname in enumerate(fname_list):
        axnmbr=axnmbr+1
        #print("amxb",axnmbr)
        ax.append(fig.add_subplot(axnmbr))
        if subtitles!=None:
            subt = subtitles[j]
       
        ax[-1],t=plot_sol_at_t_diff_tau(fname,ax[-1],i=i,exact_soln_np=exact_soln_np_list[j],x_lim=x_lim,y_lim=y_lim,plottype=plottype,subtitle=subt,skip=skip_list[j])
        
    #fig.suptitle("Solutions @ t="+str(t)[:3],fontsize=20)
    return fig


if __name__=="__main__":

    plot_dir = "../figures/"
    if not(os.path.exists(plot_dir)):
        os.makedirs(plot_dir)
 
    im_sch = str(sys.argv[1])

    ini_type_2s = str(2)+"_solitons"
    ini_type_3s = str(3)+"_solitons"

    soliton_data_dir_2s = "./data/"+ini_type_2s+"/"
    soliton_data_dir_3s = "./data/"+ini_type_3s+"/"

    subtitles = ["2 Solitons","3 Solitons"]

    filename_2s= soliton_data_dir_2s+ini_type_2s+"_imex_"+im_sch+"_2048_0.0001_tau.pkl"
    filename_3s= soliton_data_dir_3s+ini_type_3s+"_imex_"+im_sch+"_2048_0.0001_tau.pkl"
    file_list = [filename_2s,filename_3s]
    
    exact_soln_np_2s,exact_dxsoln_np_2s,sol_2s,dx_sol_2s,kppa_2s,T_2s,m_2s,L_2s = load_ini_conditions(2,2048)
    exact_soln_np_3s,exact_dxsoln_np_3s,sol_3s,dx_sol_3s,kppa_3s,T_3s,m_3s,L_3s = load_ini_conditions(3,2048)

    exact_list = [exact_soln_np_2s,exact_soln_np_3s]


    figr_test = plot_axb_sol_at_t_diff_tau(file_list,a=1,b=2,i=-1,exact_soln_np_list=exact_list,x_lim=[-12.0,12.0],plottype="abs",subtitles=subtitles)



    figr_test.savefig(plot_dir+"soltions_23_solutions_"+im_sch+".pdf")


    ######################## AP Plots ################################################
   # scheme_list = [str(sys.argv[i]) for i in range(2,len(sys.argv))]
   # print("scheme list",scheme_list)
   # list_2s = [soliton_data_dir_2s+ini_type_2s+"_imex_"+im_sch+"_2048_0.0001_tau.pkl" for im_sch in scheme_list ]
   # list_3s = [soliton_data_dir_3s+ini_type_3s+"_imex_"+im_sch+"_2048_0.0001_tau.pkl" for im_sch in scheme_list ]

   # fname_list = [list_2s,list_3s]

   # figr_test_2 = plot_axb_Error_wrt_tau(fname_list,a=1,b=2,subtitles=subtitles,schm_list=scheme_list,err_type="L2")
   # figr_test_2.savefig(plot_dir+"solitons_23_AP.pdf")