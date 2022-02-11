from rc_datasets import rc_datasets

def test_mg(fac=10,Ttot=1000,x0=1.8,dt=0.1):
    fac_=fac
    Ttot_=Ttot
    Nt=fac*Ttot
    x0_=x0
    dt_=dt
    ds=rc_datasets.mackeyglass(fac_,Nt,dt_,x0_,0.2,0.1,17)
    return ds