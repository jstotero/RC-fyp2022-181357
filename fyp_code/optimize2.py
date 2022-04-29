from fyp_code import loukas_esn        # Esn class
from colorama import Fore              # print statement colours
import numpy as np                     # numpy
import math                            # math
import matplotlib.pyplot as plt        # plotting
e = np.empty(0)                        # used in cheking default values

#==================================================================================================================================================
# Optimization class for loukas_esn.Esn objects

# pass:
# @ esn = loukas_esn.Esn object initialised normally
# @ rhos = list of rho values to optimize
# @ rhos = list of alpha values to optimize
# @ rhos = list of beta values to optimize
# @ (optional) val_time = validation time used in validation
# @ (optional) test_time = test time used in testing

class Optimizer2:
    
    def __init__(self, esn, rhos=[], alphas=[], betas=[], val_time=500, test_time=500):
        assert isinstance(esn, loukas_esn.Esn), "esn must be a loukas_esn.Esn object"
        self.esn = esn
        self.rhos = rhos
        self.alphas = alphas
        self.betas = betas
        self.val_time = val_time
        self.test_time = test_time
        
    #==========================================================================================================================================
    # Train and test the esn with currently set parameters
    
    def run_esn(self):
        self.esn.train()
        self.esn.test(test_time=self.test_time)
        
    #==========================================================================================================================================
    # Util functions to return only the param/nmse pair dictionaries
    
    def collect_rhos(self, rs=e, mute=False):
        rhos = (rs if rs.any() else self.rhos)                                  # store rho parameter space
        return self.opt_rho(rs=rhos,mute=mute)[2]                               # return only rho/nmse pair dictionary
    
    def collect_alphas(self, als=e, mute=False):
        alphas = (als if als.any() else self.alphas)                            # store alpha parameter space
        return self.opt_alpha(als=alphas,mute=mute)[2]                          # return only alpha/nmse pair dictionary
    
    def collect_betas(self, bs=e, mute=False):
        betas = (bs if bs.any() else self.betas)                                # store beta parameter space
        return self.opt_beta_test(bs=betas,mute=mute)[2]                        # return only beta/nmse pair dictionary
    
    def collect_W_sparsities(self, count=20, mute=False):
        return self.opt_W_sparsity(count=count,mute=mute)[2]                    # return only sparsity/nmse pair dictionary
        
    #==========================================================================================================================================
    # Grid-search functions
    
    # Computes optimal rho value with other parameters fixed, w.r.t. test nmse
    # pass:
    # @ (optional) rs = list of rho values to optimize, otherwise uses rhos from Optimizer initialisation
    # @ (optional) mute = True to mute print statements
    # returns:
    # @ rho giving the lowest test nmse
    # @ nmse for optimal rho
    # @ dictionary containing all rho/nmse pairs
    
    def opt_rho(self, rs=e, mute=False):
        store_r = self.esn.get_rho()                                            # store original rho value
        rhos = (rs if rs.any() else self.rhos)                                  # store rho parameter space
        rn = {}                                                                 # dictionary to store rho/nmse pairs
        for r in rhos:
            self.esn.set_rho(r)                                                 # rescale W matrix using new rho
            self.run_esn()                                                      # train & test esn with new rho
            if not mute:
                print("rho: {}, nmse: {}, {}%".format(round(r,3),round(self.esn.nmse_test,6),math.floor((100*(list(rhos).index(r)+1)/len(rhos)))))
            rn[r] = self.esn.nmse_test                                          # insert rho/nmse pair into rn
        opt_r = min_from_dict(rn)                                               # extract rho value with lowest nmse
        if not mute:
            print(Fore.GREEN + "rho: {}\nnmse: {}".format(opt_r,rn[opt_r]))     # printing result
        self.esn.set_rho(store_r)                                               # reset rho to original value
        self.esn.reset_x()                                                      # reset esn activations to initial conditions
        return opt_r, rn[opt_r], rn                                             # return (rho, nmse, rn) tuple
    
    #---------------------------------------------------------------------------------------------------------------------------------
    
    # Computes optimal alpha value with other parameters fixed, w.r.t. test nmse
    # pass:
    # @ (optional) als = list of alpha values to optimize, otherwise uses alphas from Optimizer initialisation
    # @ (optional) mute = True to mute print statements
    # returns:
    # @ alpha giving the lowest test nmse
    # @ nmse for optimal alpha
    # @ dictionary containing all alpha/nmse pairs
    
    def opt_alpha(self, als=e, mute=False):
        store_a = self.esn.get_alpha()                                          # store original alpha value
        alphas = (als if als.any() else self.alphas)                            # store alpha parameter space
        an = {}                                                                 # dictionary to store alpha/nmse pairs
        for a in alphas:
            self.esn.set_alpha(a)                                               # set new alpha value
            self.run_esn()                                                      # train & test esn with new alpha
            if not mute:
                print("alpha: {}, nmse: {}, {}%".format(round(a,3),round(self.esn.nmse_test,6),math.floor((100*(list(alphas).index(a)+1)/len(alphas)))))
            an[a] = self.esn.nmse_test                                          # insert alpha/nmse pair into an
        opt_a = min_from_dict(an)                                               # extract alpha value with lowest nmse
        if not mute:
            print(Fore.GREEN + "alpha: {}\nnmse: {}".format(opt_a,an[opt_a]))   # printing result
        self.esn.set_alpha(store_a)                                             # reset alpha to original value
        self.esn.reset_x()                                                      # reset esn activations to initial conditions
        return opt_a, an[opt_a], an                                             # return (alpha, nmse, an) tuple
    
    #---------------------------------------------------------------------------------------------------------------------------------
    
    # Computes optimal beta value with other parameters fixed, w.r.t. validation nmse
    # pass:
    # @ (optional) bs = list of beta values to optimize, otherwise uses betas from Optimizer initialisation
    # @ (optional) mute = True to mute print statements
    # returns:
    # @ beta giving the lowest validation nmse
    # @ validation nmse for optimal beta
    # @ dictionary containing all beta/nmse pairs
    
    def opt_beta_val(self, bs=e, mute=False):
        store_b = self.esn.get_beta()                                           # store original beta value
        betas = (bs if bs.any() else self.betas)                                # store beta parameter space
        bn = {}                                                                 # dictionary to store beta/nmse pairs
        self.esn.train_M()                                                      # train measurement matrix (before training Wout)
        for b in betas:
            self.esn.set_beta(b)                                                # set new beta value
            self.esn.train_readouts()                                           # train Wout using new beta
            self.esn.validate(val_time=self.val_time)                           # validate esn
            if not mute:
                print("beta: {}, nmse: {}, {}%".format(b,round(self.esn.nmse_val,6),math.floor((100*(list(betas).index(b)+1)/len(betas)))))
            bn[b] = self.esn.nmse_val                                           # insert beta/nmse pair into bn
        opt_b = min_from_dict(bn)                                               # extract beta value with lowest nmse
        if not mute:
            print(Fore.GREEN + "beta: {}\nnmse: {}".format(opt_b,bn[opt_b]))    # print results
        self.esn.set_beta(store_b)                                              # reset beta to original value
        self.esn.reset_x()                                                      # reset esn activations to initial conditions
        return opt_b, bn[opt_b], bn                                             # return (beta, nmse) tuple
    
    #---------------------------------------------------------------------------------------------------------------------------------
    
    # Computes optimal beta value with other parameters fixed, w.r.t. test nmse
    # pass:
    # @ (optional) bs = list of beta values to optimize, otherwise uses betas from Optimizer initialisation
    # @ (optional) mute = True to mute print statements
    # returns:
    # @ beta giving the lowest test nmse
    # @ test nmse for optimal beta
    # @ dictionary containing all beta/nmse pairs
    
    def opt_beta_test(self, bs=e, mute=False):
        store_b = self.esn.get_beta()                                           # store original beta value
        betas = (bs if bs.any() else self.betas)                                # store beta parameter space
        bn = {}                                                                 # dictionary to store beta/nmse pairs
        self.esn.train_M()                                                      # train measurement matrix (before training Wout)
        for b in betas:
            self.esn.set_beta(b)                                                # set new beta value
            self.esn.train_readouts()                                           # train Wout using new beta
            self.esn.test(test_time=self.test_time)                             # test esn
            if not mute:
                print("beta: {}, nmse: {}, {}%".format(b,round(self.esn.nmse_test,6),math.floor((100*(list(betas).index(b)+1)/len(betas)))))
            bn[b] = self.esn.nmse_test                                          # insert beta/nmse pair into bn
        opt_b = min_from_dict(bn)                                               # extract beta value with lowest nmse
        if not mute:
            print(Fore.GREEN + "beta: {}\nnmse: {}".format(opt_b,bn[opt_b]))    # print results
        self.esn.set_beta(store_b)                                              # reset beta to original value
        self.esn.reset_x()                                                      # reset esn activations to initial conditions
        return opt_b, bn[opt_b], bn                                             # return (beta, nmse) tuple
    
    #---------------------------------------------------------------------------------------------------------------------------------
    
    # Computes optimal rho/alpha pair with other parameters fixed, w.r.t. test nmse
    # pass:
    # @ (optional) rs = list of rho values to optimize, otherwise uses rhos from Optimizer initialisation
    # @ (optional) als = list of alpha values to optimize, otherwise uses alphas from Optimizer initialisation
    # @ (optional) mute = True to mute print statements
    # returns:
    # @ rho giving the lowest test nmse
    # @ alpha giving the lowest test nmse
    # @ nmse for optimal rho/alpha pair
    
    def opt_rho_alpha(self,rs=e,als=e,mute=False):
        store_r = self.esn.get_rho()                                            # store original rho value
        store_a = self.esn.get_alpha()                                          # store original alpha value
        rhos = (rs if rs.any() else self.rhos)                                  # store rho parameter space
        alphas = (als if als.any() else self.alphas)                            # store alpha parameter space
        _nmse_test = 10                                                         # default value
        _rho = 10                                                               # default value
        _alpha = 10                                                             # default value
        for a in alphas:
            self.esn.set_alpha(a)                                               # set new alpha value
            if not mute:
                print("rho: {}-{}, alpha: {}, {}%".format(rhos[0],rhos[-1],round(a,3),math.floor((100*(list(alphas).index(a)+1)/len(alphas)))))
            for r in rhos:
                self.esn.set_rho(r)                                             # rescale W matrix using new rho
                self.run_esn()                                                  # train & test esn with new alpha/rho
                if self.esn.nmse_test < _nmse_test:                             # store rho/alpha/nmse if optimal
                    _nmse_test = self.esn.nmse_test
                    _rho = r
                    _alpha = a
        if not mute:
            print(Fore.GREEN + "rho: {}\nalpha: {}\nnmse: {}".format(_rho,_alpha,_nmse_test))     # print results
        self.esn.set_rho(store_r)                                                                 # reset rho to original value
        self.esn.set_alpha(store_a)                                                               # reset alpha to original value
        self.esn.reset_x()                                                                        # reset esn activations to initial conditions
        return _rho, _alpha, _nmse_test                                                           # return (rho, alpha, nmse) tuple
    
    #---------------------------------------------------------------------------------------------------------------------------------
    
    # Computes the optimal reservoir matrix sparsity with other parameters fixed, w.r.t. test nmse
    # pass:
    # @ (optional) start = lowest sparsity to start at; use if test nmse -> inf around low sparsities
    # @ (optional) end = highest sparsity to end on; requires start < end < 1
    # @ (optional) count = number of sparsity values to optimize for
    # @ (optional) mute = True to mute print statements
    # returns:
    # @ sparsity giving the lowest test nmse
    # @ nmse for optimal sparsity
    # @ dictionary containing all sparsity/nmse pairs
    
    def opt_W_sparsity(self,start=0,end=0.95,count=20,mute=False):
        assert 0 < count, "count must be a positive number"
        assert 0 < start < 1, "start must be between 0 and 1"
        assert start < end < 1, "requires start < end < 1"
        store_s = self.esn.get_W_sparsity()                                     # store original W sparsisty
        c = math.floor(count)                                                   # integer number of sparsity values to optimize for
        sparsities = np.linspace(start,end,c)                                   # initialise sparsity range
        sn = {}                                                                 # dictionary to store sparsity/nmse pairs
        for s in sparsities:
            self.esn.set_W_sparsity(s)                                          # resparsify W
            self.run_esn()                                                      # train & test esn with new sparsity
            if not mute:
                print("sparsity: {}, nmse: {}, {}%".format(round(s,3),round(self.esn.nmse_test,6),math.floor(100*(s+(1/c)))))
            sn[s] = self.esn.nmse_test                                                # insert sparsity/nmse pair into sn
        opt_s = min_from_dict(sn)                                                     # extract sparsity value with lowest nmse
        if not mute:
            print(Fore.GREEN + "sparsity: {}\nnmse: {}".format(opt_s,sn[opt_s]))      # printing result
        self.esn.set_W_sparsity(store_s)                                              # reset W sparsity to original value
        self.esn.reset_x()                                                            # reset esn activations to initial conditions
        return opt_s, sn[opt_s], sn                                                   # return (sparsity, nmse, sn) tuple
    
    #==========================================================================================================================================
    # Plotting
    
    # Plot test nmse against rho values
    # Optimal rho value is orange
    # pass:
    # @ rho/nmse pair dictionary (e.g. from collect_rhos())
    
    def plot_rhos(self, pn_dict):
        assert isinstance(pn_dict,dict), "pn_dict must be a param/nmse pair dictionary"
        params = list(pn_dict.keys())                                                        # extract parameter space
        nmses = list(pn_dict.values())                                                       # extract nmse values
        x_values = params                                                                    # set x values
        y_values = np.log10(nmses)                                                           # log scale nmse values = y values
        cols = {params[nmses.index(min(nmses))]:"tab:orange"}                                # generate colour for optimal value
        for i in range(len(nmses)):
            plt.scatter(x_values[i], y_values[i], color=cols.get(params[i], 'black'))        # plot each point w/ colour
        plt.xlabel("rho")
        plt.ylabel("log nmse")
        plt.show()
        
    #---------------------------------------------------------------------------------------------------------------------------------
    
    # Plot test nmse against alpha values
    # Optimal alpha value is blue
    # pass:
    # @ alpha/nmse pair dictionary (e.g. from collect_alphas())
    
    def plot_alphas(self, pn_dict):
        assert isinstance(pn_dict,dict), "pn_dict must be a param/nmse pair dictionary"
        params = list(pn_dict.keys())                                                        # extract parameter space
        nmses = list(pn_dict.values())                                                       # extract nmse values
        x_values = params                                                                    # set x values
        y_values = np.log10(nmses)                                                           # log scale nmse values = y values
        cols = {params[nmses.index(min(nmses))]:"tab:cyan"}                                  # generate colour for optimal value
        for i in range(len(nmses)):
            plt.scatter(x_values[i], y_values[i], color=cols.get(params[i], 'black'))        # plot each point w/ colour
        plt.xlabel("alpha")
        plt.ylabel("log nmse")
        plt.show()
        
    #---------------------------------------------------------------------------------------------------------------------------------
        
    # Plot test nmse against beta values
    # Optimal beta value is pink
    # pass:
    # @ beta/nmse pair dictionary (e.g. from collect_betas())
    
    def plot_betas(self, pn_dict):
        assert isinstance(pn_dict,dict), "pn_dict must be a param/nmse pair dictionary"
        params = list(pn_dict.keys())                                                        # extract parameter space
        nmses = list(pn_dict.values())                                                       # extract nmse values
        x_values = np.log10(params)                                                          # log scale beta values = x values
        y_values = np.log10(nmses)                                                           # log scale nmse values = y values
        cols = {params[nmses.index(min(nmses))]:"fuchsia"}                                   # generate colour for optimal value
        for i in range(len(nmses)):
            plt.scatter(x_values[i], y_values[i], color=cols.get(params[i], 'black'))        # plot each point w/ colour
        plt.xlabel("log beta")
        plt.ylabel("log nmse")
        plt.show()
        
    #---------------------------------------------------------------------------------------------------------------------------------
        
    # Plot test nmse against beta values
    # Optimal beta value is pink
    # pass:
    # @ beta/nmse pair dictionary (e.g. from collect_betas())
    
    def plot_W_sparsities(self, pn_dict):
        assert isinstance(pn_dict,dict), "pn_dict must be a param/nmse pair dictionary"
        params = list(pn_dict.keys())                                                        # extract parameter space
        nmses = list(pn_dict.values())                                                       # extract nmse values
        x_values = params                                                                    # set x values
        y_values = np.log10(nmses)                                                           # log scale nmse values = y values
        cols = {params[nmses.index(min(nmses))]:"limegreen"}                                 # generate colour for optimal value
        for i in range(len(nmses)):
            plt.scatter(x_values[i], y_values[i], color=cols.get(params[i], 'black'))        # plot each point w/ colour
        plt.xlabel("W sparsity")
        plt.ylabel("log nmse")
        plt.show()
    
    #==========================================================================================================================================
    # Variable setters/getters
    
    def return_esn(self):
        return self.esn
    
    #----------------------------------------
    
    def set_rhos(self,rhos):
        self.rhos = rhos
        
    def set_alphas(self,alphas):
        self.alphas = alphas
        
    def set_betas(self,betas):
        self.betas = betas
        
    #----------------------------------------
        
    def get_rhos(self):
        return self.rhos
    
    def get_betas(self):
        return self.betas
    
    def get_alphas(self):
        return self.alphas

#==================================================================================================================================================
# Other functions

# extract parameter value with lowest nmse from a param/nmse pair dictionary
def min_from_dict(pn_dict):
    return list(pn_dict.keys())[list(pn_dict.values()).index(min(pn_dict.values()))]

# shortcut to printing optimal value
def print_optimal(pn_dict, param=""):
    opt = min_from_dict(pn_dict)
    print(Fore.GREEN + "Optimal {}: {}, nmse: {}".format(param,opt,pn_dict[opt]))