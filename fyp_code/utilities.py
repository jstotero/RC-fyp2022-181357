# Utilities
from rc_datasets import rc_datasets
import numpy as np

# used in data_mg()
def test_mg(fac=10,Ttot=1000,x0=1.8,dt=0.1):
    fac_=fac
    Ttot_=Ttot
    Nt=fac*Ttot
    x0_=x0
    dt_=dt
    ds=rc_datasets.mackeyglass(fac_,Nt,dt_,x0_,0.2,0.1,17)
    return ds

# create mackey glass data
def data_mg(Ttot=10000, fac=10, x0=1.8, dt=0.1):
    Ttot = 10000
    fac = 10
    x0 = 1.8
    dt = 0.1
    ds_mg = test_mg(fac,Ttot,x0,dt).u
    return ds_mg.reshape(1,Ttot)

# create sin data
# choose order = 1,2,3 for different order superpositions
def data_sin(order=1,Ttot=10000):
    theta = np.linspace(0,Ttot/100,Ttot)
    orders = {1 : np.sin(5*theta).reshape(1,Ttot),
              2 : 0.5 + 0.25*np.sin(5*0.8*np.pi*theta).reshape(1,Ttot) + 0.15*np.sin(5*0.3*np.sqrt(2)*np.pi*theta).reshape(1,Ttot),
              3 : 0.75 + 0.4*np.sin(5*0.5*np.pi*theta).reshape(1,Ttot) - (np.sin(5*0.25*np.sqrt(3)*np.pi*theta)**2).reshape(1,Ttot) +(0.6*np.sin(5*0.25*np.pi*theta)**3).reshape(1,Ttot)}
    return orders[order]

# reshape data
def reshape(data):
    return np.array(data).reshape(1,len(data))