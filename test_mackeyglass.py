from rc_datasets import rc_datasets

def test_mg():
    
    fac=10
    Ttot=10000
    Nt=fac*Ttot
    x0=1.8
    dt=0.1

    ds=rc_datasets.mackeyglass(fac,Nt,dt,x0,0.2,0.1,17)
    return ds