import sys
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation

from IPython.display import HTML
import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

cwd = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from NLS_functions import *


def calc_dx(xi,u):
    u_ft = fft(u)
    u_x = ifft(1j*xi*u_ft)
    return u_x.real

def calc_dx_5p(x,u):
    
    dx = x[1]-x[0]
    u_x = np.zeros_like(u)
    u_x[2:-2] = (-u[4:]+8.0*u[3:-1]-8.0*u[1:-3]+u[0:-4])/(12.0*dx)
    return u_x/dx
    
def calc_dx_np(x,u):
   
    u_x = np.gradient(u,x)
    return u_x

def calc_dx_c(x,u):
    dx = x[1]-x[0]
    #print("dxxx",dx)
    #dx = 1.0
    u_x = np.zeros_like(u)
    u_x[1:] = u[1:]-u[:-1]
    u_x[1:] = np.where(np.abs(u_x[1:])>=np.pi,2.0*np.pi+u[1:]-u[:-1],u_x[1:])
    return u_x/dx




plot_dir = "../figures/"
if not(os.path.exists(plot_dir)):
        os.makedirs(plot_dir)


file_name = "./data/U_0.01_DSW/0.01_DSW_imex_SSP3-ImEx(3,4,3)_16384_0.0001_NLS_xtnd.pkl"
with open(file_name,"rb") as f:
    d_dict = pickle.load(f)


file_name2 = "./data/U_0.01_DSW/0.01_DSW_imex_SSP3-ImEx(3,4,3)_16384_0.0001_tau_xtnd.pkl"
with open(file_name2,"rb") as f2:
    d_dict2 = pickle.load(f2)
#print(len(d_dict),len(d_dict2))
for i in range(1,(len(d_dict2))):
    d_dict.append(d_dict2[i])

#print(len(d_dict),len(d_dict2))

x = d_dict[0]["x"]
#xi = d_dict[0]["xi"]
#x2 = d_dict2[0]["x"]
#xi2 = d_dict2[0]["xi"]

t_list = d_dict[0]["t_list"]
#t_list2 = d_dict_nls2[0]["t_list"]
#print(d_dict[0].keys())
#print(d_dict[1].keys())



fig=plt.figure(figsize=(20,20))
axes = fig.add_subplot(211)
axes2 = fig.add_subplot(212)



plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

i = 99

rho_l = 2.0
rho_r = 1.0
u_r = 0.0
u_l = 0.0
rho_0 = 0.25*np.square( np.sqrt(rho_r) + np.sqrt(rho_l) +0.5*(u_l-u_r)  )
u_0 = 0.5*(u_l+u_r) + np.sqrt(rho_l) - np.sqrt(rho_r)

tau_1 = u_r + (8.0*rho_0 - 8.0*np.sqrt(rho_0*rho_r) + rho_r)/(2.0*np.sqrt(rho_0) - np.sqrt(rho_r))
tau_2 = u_r + np.sqrt(rho_0)
tau_3 = u_0 - np.sqrt(rho_0)
tau_4 = u_l - np.sqrt(rho_l)

t1 = t_list[i]
#print(frame_list[i].shape)
xlim_v = 1.0*335.0/t1


dx = x[1]-x[0]
#print("dx",dx)
state_sym = ["\rho_","_"]
axes2.set_ylim([-2.5*u_0,2.5*u_0])


x1, x2, y1, y2 = u_r + 0.7*np.sqrt(rho_0), tau_1,rho_r-0.3*rho_0, rho_0  # subregion of the original image
axins_top = axes.inset_axes(
    [u_l - 2.5*np.sqrt(rho_l), 0.7*rho_r, 3.8, 0.6],
    xlim=(1.3*x1, 0.7*x2),ylim=(y1, 1.2*y2), xticklabels=[], yticklabels=[], transform=axes.transData)

axins_top.axvline(tau_2,color="m")
axins_top.annotate(r"$\xi_2$",(tau_2,1.0),(tau_2,0.4),fontsize=25)


x1, x2, y1, y2 = u_r + 0.7*np.sqrt(rho_0), tau_1,-2.0*u_0, u_0  # subregion of the original image
axins_bot = axes2.inset_axes(
    [u_l - 2.5*np.sqrt(rho_l), -0.9, 3.8, 0.6],
    xlim=(1.3*x1, 0.7*x2),ylim=(y1, 1.2*y2), xticklabels=[], yticklabels=[], transform=axes2.transData)

axins_bot.axvline(tau_2,color="m")
axins_bot.annotate(r"$\xi_2$",(tau_2,-0.5),(tau_2,-0.7),fontsize=25)

#axins_top.axhline(rho_0,ls="--",color="k",lw="0.5")
#axins_top.annotate(r"$\rho_0$",(x1,rho_0),(x1-0.5,rho_0),fontsize=25)




for j,ax in enumerate([axes,axes2]):

    ax.set_xlim([-xlim_v,xlim_v])
    #print(j)
    if j==0:
        state_l=rho_l
        state_0=rho_0
        state_r=rho_r

        sym_l = r"$\rho_L$"
        sym_r = r"$\rho_R$"
        sym_0 = r"$\rho_0$"

    else:
        state_l=u_l
        state_0=u_0
        state_r=u_r

        sym_l = "$\phi_L$"
        sym_r = "$\phi_R$"
        sym_0 = "$\phi_0$"
    
    ax.axvline(tau_1,color="m")
    ax.annotate(r"$\xi_1$",(tau_1,0.6),(tau_1,0.5),fontsize=25)
    ax.axvline(tau_2,color="m")
    ax.annotate(r"$\xi_2$",(tau_2,0.6),(tau_2,0.5),fontsize=25)
    ax.axvline(tau_3,color="m")
    ax.annotate(r"$\xi_3$",(tau_3,0.6),(tau_3,0.5),fontsize=25)
    ax.axvline(tau_4,color="m")
    ax.annotate(r"$\xi_4$",(tau_4,0.6),(tau_4,0.5),fontsize=25)
    #ax.legend(loc="best")
    #("tau s",tau_1,tau_2,tau_3,tau_4)
    
    ax.axhline(state_l,ls="--",color="k",lw="0.5")
    ax.annotate(sym_l,(-xlim_v,state_l),(-xlim_v-0.5,state_l),fontsize=25)
    
    ax.axhline(state_r,ls="--",color="k",lw="0.5")
    if j==0:
        ax.annotate(sym_r,(-xlim_v,state_r),(-xlim_v-0.5,state_r),fontsize=25)
    else:
        ax.annotate(sym_r,(xlim_v,state_r),(xlim_v+0.5,state_r),fontsize=25)
    
    ax.axhline(state_0,ls="--",color="k",lw="0.5")
    ax.annotate(sym_0,(-xlim_v,state_0),(-xlim_v-0.5,state_0),fontsize=25)

    #ax.tick_params(axis='both', which='major', labelsize=10)


axes.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 



dp_nls = d_dict[0]
frame_list = dp_nls["frame_list"]

#dp_nls2 = d_dict2[0]
#frame_list2 = dp_nls2["frame_list"]



x = dp_nls["x"]
xi = dp_nls["xi"]
#x2 = dp_nls2["x"]
#xi2 = dp_nls2["xi"]


    
q = frame_list[i][:]
rho = np.square(np.abs(q))

#q2= frame_list2[i][:]
#rho2 = np.square(np.abs(q2))



    



theta = np.angle(q)
theta_x = calc_dx_c(x,theta)


axes.plot(x/t1 +800.0/t1,rho,"k-",label=r" NLS")
axins_top.plot(x/t1 +800.0/t1,rho,"k-")
#axes.plot(x2/t1,rho2,"g--",label=r"$\delta=0.1$  NLS")
   
axes2.plot(x/t1 +800.0/t1,theta_x,"k-",label=r" NLS")
axins_bot.plot(x/t1 +800.0/t1,theta_x,"k-")
#axes2.plot(x2/t1,theta_x2,label=r"$\delta=0.1$ NLS")
j=0 
lt=[":b","--r","-.g","m--"]
#for dp01,dp1 in zip(d_dict[1:-1],d_dict2[1:-1]):
drray = [d_dict[k] for k in [4,3]]
for dp01 in drray:
    frame_list = dp01["frame_list"]
    #frame_list2 = dp1["frame_list"]

    tau = dp01["tau"]
    x = dp01["x"]
    #x2 = dp1["x"]
    xi = dp01["xi"]
    #xi2 = dp1["xi"]

    
    q = frame_list[i][:,0]
    #q2 = frame_list2[i][:,0]
    rho = np.square(np.abs(q))
    #rho2 = np.square(np.abs(q2))

    #cos_theta = q.real/np.abs(q)
    #cos_theta2  = q2.real/np.abs(q2)
    
    #sin_theta = q.imag/np.abs(q)
    #sin_theta2  = q2.imag/np.abs(q2)

    
    #grd = calc_dx(xi,cos_theta)
    #grd2 = calc_dx(xi2,cos_theta2)
    
    #theta_x = -grd/sin_theta
    #theta2_x = -grd2/sin_theta2


    theta = np.angle(q)
    theta_x = calc_dx_c(x,theta)

 
    axes.plot(x/t1+800.0/t1,rho,lt[j],label=r"$\tau=$ "+str(tau))
    #axes.plot(x2/t1,rho2,"--",label=r"$\delta = 0.1,\tau=$ "+str(tau))

    axins_top.plot(x/t1+800.0/t1,rho,lt[j])

    #axes2.set_xlim([1.4,1.5])
    axes2.plot(x/t1+800.0/t1,theta_x,lt[j],label=r"$\tau=$  "+str(tau))
    axins_bot.plot(x/t1+800.0/t1,theta_x,lt[j])
    #axes2.plot(x2/t1,theta2_x,"--",label=r"$\delta = 0.1,\tau=$ "+str(tau))

    #axes2.plot((x/t1)[1:-1],theta_x,label=r"$\delta = 0.01,\tau=$  "+str(tau))
    #axes2.plot((x2/t1)[1:-1],theta2_x,label=r"$\delta = 0.1,\tau=$  "+str(tau))

    axes.legend(loc="best",fontsize=20)
    axes2.legend(loc="best",fontsize=20)
    axes2.set_xlabel(r"$(\xi\equiv x/t)_{t=70}$",fontsize=20)
    #axes.tick_params(axis='both', which='major', labelsize=20)
    #axes2.tick_params(axis='both', which='major', labelsize=20)
    j=j+1

#print(rho.shape,rho2.shape)
axes.indicate_inset_zoom(axins_top, edgecolor="black")
axes2.indicate_inset_zoom(axins_bot, edgecolor="black")
axes.legend(loc="best",fontsize=20)
axes2.legend(loc="best",fontsize=20)
plt.savefig(plot_dir+"/Dhaouadi_DSW.pdf")