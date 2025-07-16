import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

import os
this_script_path = os.path.dirname(os.path.abspath(__file__))
figure_path = os.path.join(this_script_path,"..","figures")

############################################################
#Generate the plots for Section 3.1: Standing fronts
############################################################
def traveling_NLS(sign=1,kappa=-1,mu=-1,u0=0,uinf=None):
    if uinf is None:
        uinf = np.sqrt(mu/kappa)
    return lambda x: uinf*np.tanh(sign*uinf*np.sqrt(-kappa/2.)*x+sign*np.arctanh(u0/uinf))


def traveling_NLSH(tau=1.,sign=1,kappa=-1,mu=-1,u0=0,uinf=None):
    if uinf is None:
        uinf = np.sqrt(mu/kappa)
    return lambda x: uinf*np.tanh(sign*uinf*np.sqrt(-kappa*(1-mu*tau)/2.)*x+sign*np.arctanh(u0/uinf))

tau_list = [1.e-5, 0.1, 1, 1000]
style_list = [":","--","-.",(0, (3, 1, 1, 1, 1, 1))][::-1]
colot_list = ["b","r","g","m"][::-1]

fig, ax = plt.subplots(1,2,figsize=(10,2),dpi=200)


kappa = -1
mu = -1
uinf = np.sqrt(mu/kappa)
sign = 1
u0 = 0

u = traveling_NLS(sign=sign,kappa=kappa,mu=mu,u0=u0)
u_tau_list = [traveling_NLSH(tau=tau,sign=sign,kappa=kappa,mu=mu,u0=u0) for tau in tau_list]
x = np.linspace(-5,5,1000)
ax[0].plot(x,u(x),color='k',linestyle='-',label=r'$\overline{u}^{+}$ NLS')
for tau, u_tau in zip(tau_list,u_tau_list):
    ax[0].plot(x,u_tau(x),label=r"$\tau =$ "+str(tau),linestyle=style_list.pop(),color=colot_list.pop())

ax[0].legend()
ax[0].set_title(r"$\overline{q}^{-}_0$")
ax[0].set_xlabel(r"$x$")

style_list = [":","--","-.",(0, (3, 1, 1, 1, 1, 1))][::-1]
colot_list = ["b","r","g","m"][::-1]

ax[1].plot(x[2:-2],np.gradient(u(x),x)[2:-2],color='k',linestyle='-',label=r"$(\overline{u}^{+})\,'$ NLS")
for tau, u_tau in zip(tau_list,u_tau_list):
    ax[1].plot(x[2:-2], (1/(1-mu*tau))*np.gradient(u_tau(x),x)[2:-2],linestyle=style_list.pop(),color=colot_list.pop())


ax[1].set_title(r"$\overline{q}^{-}_1$")
ax[1].set_xlabel(r"$x$")
ax[1].legend()

#Save as pdf
plt.savefig(f'{figure_path}/traveling_fronts_NLSH.pdf',bbox_inches='tight')

############################################################
#Generate the plots for Section 3.3: Standing solitary waves
############################################################
def arcsech(x):
    return np.log((1 + np.sqrt(1 - x**2)) / x)
sech = lambda x: 1 / np.cosh(x)

def solitary_NLS(kappa=1,mu=1,u0=1,sign=1):
    R = np.sqrt(2*mu/kappa)
    C = arcsech(u0/R)
    return lambda x: R*sech(np.sqrt(mu)*x+sign*C)


def solitary_NLSH(tau=1.,kappa=1,mu=1.,u0=1.,sign=1):
    R = np.sqrt(2*mu/kappa)
    C = arcsech(u0/R)
    return lambda x: R*sech(np.sqrt(mu*(1-mu*tau))*x+sign*C)

tau_list = [1.e-5, 0.1, 0.5, 0.9]
style_list = [":","--","-.",(0, (3, 1, 1, 1, 1, 1))][::-1]
colot_list = ["b","r","g","m"][::-1]

fig, ax = plt.subplots(1,2,figsize=(10,2),dpi=200)


kappa = 1
mu = 1
uinf = np.sqrt(mu/kappa)
sign = 1
u0 = np.sqrt(2*mu/kappa)

u = solitary_NLS(kappa=kappa,u0=u0,sign=sign,mu=mu)
u_tau_list = [solitary_NLSH(tau=tau,sign=sign,kappa=kappa,mu=mu,u0=u0) for tau in tau_list]
x = np.linspace(-5,5,1000)

ax[0].plot(x,u(x),color='k',linestyle='-',label=r'$\overline{u}^{+}$ NLS')
for tau, u_tau in zip(tau_list,u_tau_list):
    ax[0].plot(x,u_tau(x),label=r"$\tau =$ "+str(tau),linestyle=style_list.pop(),color=colot_list.pop())
ax[0].legend()
ax[0].set_title(r"$\overline{q}^{+}_0$")
ax[0].set_xlabel(r"$x$")

style_list = [":","--","-.",(0, (3, 1, 1, 1, 1, 1))][::-1]
colot_list = ["b","r","g","m"][::-1]

ax[1].plot(x[2:-2],np.gradient(u(x),x)[2:-2],color='k',linestyle='-',label=r"$(\overline{u}^{+})\,'$ NLS")
for tau, u_tau in zip(tau_list,u_tau_list):
    ax[1].plot(x[2:-2], (1/(1-mu*tau))*np.gradient(u_tau(x),x)[2:-2],linestyle=style_list.pop(),color=colot_list.pop())


ax[1].set_title(r"$\overline{q}^{+}_1$")
ax[1].set_xlabel(r"$x$")
ax[1].legend()

plt.savefig(f'{figure_path}/solitary_waves_NLSH.pdf',bbox_inches='tight')

############################################################
#Generate the plots for Section 3: Phase plane analysis
############################################################
import matplotlib
font = {'size'   : 10}
matplotlib.rc('font', **font)
#Solitary wave focusing case
fig, ax = plt.subplots(1,1,figsize=(7.5,5),dpi=100)
mu = 1.0
tau = 0.01
kappa = 1.0
xmax = 38#20
v0=0#np.sqrt(mu/kappa)
ax.cla()
ax.set_xlabel(r'$\overline{q}_0$'); ax.set_ylabel(r'$\overline{q}_1$')
v = np.linspace(-1.5, 1.5, 50)
p = np.linspace(-1, 1., 50)
V, P = np.meshgrid(v, p)
dv = P-mu*tau*P
dp = -kappa*V**3+mu*V

stream = ax.streamplot(V,P,dv,dp,broken_streamlines=False,density=0.7,linewidth=0.5,arrowsize=1)
ax.axis('image')

def rhs(t,w):
    v,p = w
    return np.array([p-mu*tau*p,-kappa*v**3+mu*v])

w0 = np.array([v0,0.001])
t_eval = np.linspace(0,xmax,1000)
forwardsoln = scipy.integrate.solve_ivp(rhs,[0,xmax],w0,t_eval=t_eval,atol=1.e-12,rtol=1.e-12)
v = forwardsoln.y[0,:]
x = forwardsoln.t
ax.plot(v[1:],np.diff(v)/np.diff(x),'--r',lw=2)
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1,1])
#Increase font size of axis labels
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
fig.canvas.draw_idle()
ax.scatter([-np.sqrt(mu/kappa),0,np.sqrt(mu/kappa)],[0,0,0],c='k',s=50)
plt.savefig(f'{figure_path}/phase_plane_focusing_solitary_waves.png',dpi=300)
plt.show()
plt.close()

#Travelling front wave focusing case
fig, ax = plt.subplots(1,1,figsize=(7.5,5),dpi=100)
mu = -1.0
tau = 0.01
kappa = -1.0
xmax = 10
v0=np.sqrt(mu/kappa)
ax.cla()
ax.set_xlabel(r'$\overline{q}_0$'); ax.set_ylabel(r'$\overline{q}_1$')
v = np.linspace(-1.5, 1.5, 50)
p = np.linspace(-1, 1., 50)
V, P = np.meshgrid(v, p)
dv = P-mu*tau*P
dp = -kappa*V**3+mu*V

stream = ax.streamplot(V,P,dv,dp,broken_streamlines=False,density=0.7,linewidth=0.5,arrowsize=1)
ax.axis('image')

def rhs(t,w):
    v,p = w
    return np.array([p-mu*tau*p,-kappa*v**3+mu*v])

w0 = np.array([v0,-0.001])
t_eval = np.linspace(0,xmax,1000)
w01 = np.array([-v0,0.001])
t_eval1 = np.linspace(xmax,2*xmax,1000)

forwardsoln = scipy.integrate.solve_ivp(rhs,[0,xmax],w0,t_eval=t_eval,atol=1.e-12,rtol=1.e-12)
forwardsoln1 = scipy.integrate.solve_ivp(rhs,[xmax,2*xmax],w01,t_eval=t_eval1,atol=1.e-12,rtol=1.e-12)

v = forwardsoln.y[0,:]
x = forwardsoln.t
v1 = forwardsoln1.y[0,:]
x1 = forwardsoln1.t

ax.plot(v1[1:],np.diff(v1)/np.diff(x1),'--r',lw=2)
ax.plot(v[1:],np.diff(v)/np.diff(x),'--r',lw=2)
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1,1])
#Increase font size of axis labels
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
fig.canvas.draw_idle()
ax.scatter([-np.sqrt(mu/kappa),0,np.sqrt(mu/kappa)],[0,0,0],c='k',s=50)
plt.savefig(f'{figure_path}/phase_plane_defocusing_traveling_front.png',dpi=300)