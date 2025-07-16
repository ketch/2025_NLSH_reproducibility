import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np







def plot_H_error_wrt_tau(file_n,change_type="relative change",save_fig_name=None,IsPlot=True):

    if type(file_n)==list:
        frame_dict_list = file_n
    
        
    else:
        with open(file_n, 'rb') as f:
            frame_dict_list=pickle.load(f)
        #
       
    ini_type = frame_dict_list[0]["ini_type"]
    imex_sch = frame_dict_list[0]["scheme"]
    if save_fig_name==None:
        save_fig_name = "./figures/"+"H_Error_"+ini_type+"_"+imex_sch+".png"



    tau_list = [f["tau"] for f in frame_dict_list]


    rel_H_list=[]
    err_L1_list=[]
    err_L2_list=[]
    err_Linf_list=[]

    for tau,f in zip(tau_list,frame_dict_list):

        err_L1 = f["errors"][0]
        err_L2 = f["errors"][1]
        err_Linf = f["errors"][2]

        rel_H_list.append(f["inv_change_dict"][change_type][0])
        err_L1_list.append(err_L1)
        err_L2_list.append(err_L2)
        err_Linf_list.append(err_L1)

    if IsPlot==True:
        fig = plt.figure(figsize=[23,10])
        fig.suptitle('Case='+ini_type.replace("_"," "), fontsize=20)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        if change_type=="change":
            ax1.set_ylabel("$H_f-H_i$",fontsize=20)
        else:
            ax1.set_ylabel(r"$\frac{|H_f-H_i|}{H_i}$",fontsize=25)

        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$\tau$",fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)

        ax1.plot(tau_list,rel_H_list,"*-")




        ax2.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_xlabel(r"$\tau$",fontsize=20)

        ax2.set_ylabel("Error",fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=20)


        ax2.plot(tau_list,err_Linf_list,"o-",label=r"$L_{\inf}$")
        ax2.plot(tau_list,err_L2_list,"*-",label="$L_2$ ")

        ax2.legend(loc="best",fontsize=20)
        
        
        plt.savefig(save_fig_name)
    return tau_list,rel_H_list,err_L1_list,err_L2_list,err_Linf_list,imex_sch


def plot_Error_wrt_tau(file_list,ax,label_list=None,err_type="L2"):

    print("Using Error norm: ",err_type)

    for j,file_n in enumerate(file_list):
        with open(file_n, 'rb') as f:
                frame_dict_list=pickle.load(f)
            #
        
        ini_type = frame_dict_list[0]["ini_type"]
        imex_sch = frame_dict_list[0]["scheme"]
        



        tau_list = [f["tau"] for f in frame_dict_list]


        rel_H_list=[]
        err_L1_list=[]
        err_L2_list=[]
        err_Linf_list=[]

        ax.set_yscale("log")
        ax.set_xscale("log")
  

        ax.set_ylabel("Error",fontsize=20)
        ax.set_xlabel(r"$\tau$",fontsize=20)

        ax.tick_params(axis='both', which='major', labelsize=20)

        for tau,f in zip(tau_list,frame_dict_list):

            err_L1 = f["errors"][0]
            err_L2 = f["errors"][1]
            err_Linf = f["errors"][2]

          
            err_L1_list.append(err_L1)
            err_L2_list.append(err_L2)
            err_Linf_list.append(err_L1)


        if err_type=="L1":
             errToplot = err_L1_list
        elif err_type=="L2":
             errToplot = err_L2_list
        elif err_type=="L_Inf":
             errToplot = err_Linf_list


        if label_list!=None:
             ax.plot(tau_list,errToplot,"*-",label=label_list[j])
        else:
            ax.plot(tau_list,errToplot,"*-")

        ax.legend(loc="best",fontsize=20)
        
        
       
    return ax

def plot_axb_Error_wrt_tau(fname_list,a=1,b=2,subtitles=None,schm_list=None,err_type="L2"):

    fig = plt.figure(figsize=[20,10])
    if schm_list==None:
         schm_list=["ImEx:4 s 3 O","ImEx:6 s 4 O"]
    axnmbr = a*100+b*10
    ax = []
    for j,fname in enumerate(fname_list):
        axnmbr=axnmbr+1
        print("amxb",axnmbr)
        ax.append(fig.add_subplot(axnmbr))
        if subtitles!=None:
            subt = subtitles[j]
            ax[-1].set_title(subt,fontsize=20)
       
        ax[-1]=plot_Error_wrt_tau(fname,ax[-1],schm_list,err_type=err_type)
        
    fig.suptitle("AP Convergence",fontsize=20)
    return fig
   
def plot_1x2_sol(fname,tau=None,ilist=None,exact_soln_np=None,x_lim=None,y_lim=None):

        if type(fname)==dict:
             fdict = fname
        elif type(fname)==list:
             for f in fname:
                  tau_c = f['tau']
                  if tau==tau_c:
                       fdict = f
           
        elif type(fname)==str:
             with open(fname, 'rb') as f:
                frame_dict_list=pickle.load(f)
             for f in frame_dict_list:
                  tau_c = f['tau']
                  
                  if tau==tau_c:
                       fdict = f
             

        frames_list = fdict['frame_list']
        t_list = fdict['t_list']
        x = fdict['x']
        xi = fdict['xi']
        kppa = fdict['kappa']
        if exact_soln_np!=None:
                frames_sol = [exact_soln_np(t*np.ones_like(x),x,kppa) for t in t_list]
                print(len(frames_list),frames_sol[0].shape)
       
        n=len(t_list)
        print(len(frames_list))

        fig = plt.figure(figsize=[20,10])
        fig.suptitle(r"Solution with $\tau=$"+str(tau))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_ylabel("$Re(q_0)$")
        ax2.set_ylabel("$Im(q_0)$")

        ax1.set_xlabel("x")
        ax2.set_xlabel("x")

       

        if x_lim!=None:
            ax1.set_xlim(x_lim)
            ax2.set_xlim(x_lim)
        if y_lim!=None:
            ax1.set_ylim(y_lim)
            ax2.set_ylim(y_lim)
        if ilist is None:
             ilist = [i for i in range(0,n,18)]
        for i in ilist:
                ax1.plot(x,np.real(frames_list[i][:,0]),label="t="+str(t_list[i])[:3])
                if exact_soln_np!=None:
                    ax1.plot(x,np.real(frames_sol[i][:]),"+",label="sol t="+str(t_list[i])[:3])

                ax2.plot(x,np.imag(frames_list[i][:,0]),label="t="+str(t_list[i])[:3])
                if exact_soln_np!=None:
                    ax2.plot(x,np.imag(frames_sol[i][:]),"+",label="sol t="+str(t_list[i])[:3])

        ax1.legend(loc="best")
        ax2.legend(loc="best")

        return fig


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
        print("t_list",t_list)
        
        x = frame_dict_list[0]['x']
        xi = frame_dict_list[0]['xi']
        kppa = frame_dict_list[0]['kappa']
        if exact_soln_np!=None:
                u_sol = exact_soln_np(t*np.ones_like(x),x,kppa)
                print(u_sol.shape)
       
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
                        ax1.plot(x,np.abs(u_sol[:]),"-k",label="NLS")
                    elif plottype=="real":
                       ax1.plot(x,np.real(u_sol[:]),"-k",label="NLS")
                    elif plottype=="imag":
                        ax1.plot(x,np.imag(u_sol[:]),"-k",label="NLS")
        

        tau_list.reverse()
        u_list.reverse()          
        for j,tau in enumerate(tau_list):
            print("tau",tau)
                
            if j<3:
                if plottype=="abs":
                    ax1.plot(x,np.abs(u_list[j][:,0]),mrk_list[j],label=r"$\tau=$"+str(tau))
                elif plottype=="real":
                    ax1.plot(x,np.real(u_list[j][:,0]),mrk_list[j],label=r"$\tau=$"+str(tau))
                elif plottype=="imag":
                    ax1.plot(x,np.imag(u_list[j][:,0]),mrk_list[j],label=r"$\tau=$"+str(tau))
            else:
                if plottype=="abs":
                    ax1.plot(x,np.abs(u_list[j][:,0]),color="m",linestyle=(0, (3, 1, 1, 1, 1, 1)),label=r"$\tau=$"+str(tau))
                elif plottype=="real":
                    ax1.plot(x,np.real(u_list[j][:,0]),color="m",linestyle=(0, (3, 1, 1, 1, 1, 1)),label=r"$\tau=$"+str(tau))
                elif plottype=="imag":
                    ax1.plot(x,np.imag(u_list[j][:,0]),color="m",linestyle=(0, (3, 1, 1, 1, 1, 1)),label=r"$\tau=$"+str(tau))
                
                


                

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
        print("amxb",axnmbr)
        ax.append(fig.add_subplot(axnmbr))
        if subtitles!=None:
            subt = subtitles[j]
       
        ax[-1],t=plot_sol_at_t_diff_tau(fname,ax[-1],i=i,exact_soln_np=exact_soln_np_list[j],x_lim=x_lim,y_lim=y_lim,plottype=plottype,subtitle=subt,skip=skip_list[j])
        
    #fig.suptitle("Solutions @ t="+str(t)[:3],fontsize=20)
    return fig

def plot_1x2ReIm_sol_at_t_diff_tau(fname,i=-1,exact_soln_np=None,x_lim=None,y_lim=None):

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
            u_list.append(f["frame_list"][-1])
             

        t = t_list[0]
        print("t_list",t_list)
        
        x = frame_dict_list[0]['x']
        xi = frame_dict_list[0]['xi']
        kppa = frame_dict_list[0]['kappa']
        if exact_soln_np!=None:
                u_sol = exact_soln_np(t*np.ones_like(x),x,kppa)
                print(u_sol.shape)
       
        n=len(t_list)
        

        fig = plt.figure(figsize=[20,10])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_ylabel("$Re(q_0)$",fontsize=20)
        ax2.set_ylabel("$Im(q_0)$",fontsize=20)

        ax1.set_xlabel("x")
        ax2.set_xlabel("x")

       

        if x_lim!=None:
            ax1.set_xlim(x_lim)
            ax2.set_xlim(x_lim)
        if y_lim!=None:
            ax1.set_ylim(y_lim)
            ax2.set_ylim(y_lim)
        
        if exact_soln_np!=None:
                    ax1.plot(x,np.real(u_sol[:]),"+",label="sol")
        if exact_soln_np!=None:
                    ax2.plot(x,np.imag(u_sol[:]),"+",label="sol")

                    
        for j,tau in enumerate(tau_list):
                ax1.plot(x,np.real(u_list[j][:,0]),label=r"$\tau=$"+str(tau))
                

                ax2.plot(x,np.imag(u_list[j][:,0]),label=r"$\tau=$"+str(tau))
                

        ax1.legend(loc="best")
        ax2.legend(loc="best")

        fig.suptitle("Solutions @ t="+str(t))
        return fig
