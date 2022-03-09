from fyp_code import loukas_esn
from math import floor
from numpy import log, log10, logspace, linspace
from colorama import Fore
import matplotlib.pyplot as plt

# Optimization class for loukas_esn
# Initialise with input data, rhos/betas/alphas parameter spaces, initial rho,beta,alpha, reservoir size N
class Optimizer:
    def __init__(self, data, rhos=[], betas=[], alphas=[], rhoscale=1.3, beta=1e-7, alpha=0.7, Ttrain=500, Twashout=200, sparsity=0.5, val_time=500, test_time=500, N=100):
        self.data = data
        self.rhos = rhos
        self.betas = betas
        self.alphas = alphas
        self.rhoscale = rhoscale
        self.beta = beta
        self.alpha = alpha
        self.Ttrain = Ttrain
        self.Twashout = Twashout
        self.sparsity = sparsity
        self.val_time = val_time
        self.test_time = test_time
        self.N = N
        
    #----------------------------------------------------------------------------------------------------------------------------
    # Call to esn class

    # runs my esn given data and optional parameters
    # returns trained and tested esn
    def run_esn(self, rhoscale=None, beta=None, alpha=None):
        rho = (rhoscale if rhoscale else self.rhoscale)
        bet = (beta if beta else self.beta)
        alph = (alpha if alpha else self.alpha)
        my_esn = loukas_esn.Esn(self.data, rhoscale=rho, beta=bet, alpha=alph, Ttrain=self.Ttrain, Twashout=self.Twashout, sparsity=self.sparsity, N=self.N)
        my_esn.train()
        #my_esn.validate(val_time=self.val_time)
        my_esn.test(test_time=self.test_time)
        return my_esn
    
    #----------------------------------------------------------------------------------------------------------------------------
    # Object variable setters/getters
    
    # Self explanitory
    def set_rhos(self,rhos):
        self.rhos = rhos
        
    def set_betas(self,betas):
        self.betas = betas

    def set_alphas(self,alphas):
        self.alphas = alphas
    
    def set_rho(self,r):
        self.rhoscale = r
        
    def set_beta(self,b):
        self.beta = b

    def set_alpha(self,a):
        self.alpha = a
        
    def get_rhos(self):
        return self.rhos
    
    def get_betas(self):
        return self.betas
    
    def get_alphas(self):
        return self.alphas
    
    def get_rho(self):
        return self.rhoscale
    
    def get_beta(self):
        return self.beta
    
    def get_alpha(self):
        return self.alpha
    
    #----------------------------------------------------------------------------------------------------------------------------
    # Grid-search functions for each parameter combination

    # returns alpha giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def opt_rho(self,prnt=False):
        hold_nmse_test = 1
        hole_rho = 1
        for r in self.rhos:
            r_nmse = self.run_esn(rhoscale=r).nmse_test
            if r_nmse < hold_nmse_test:
                hold_nmse_test = r_nmse
                hold_rho = r
        if prnt:
            print("rho: {}\nnmse: {}".format(hold_rho,hold_nmse_test))
        return hold_rho, hold_nmse_test

    # returns beta giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def opt_beta(self,prnt=False):
        hold_nmse_test = 1
        hold_beta = 1
        for b in self.betas:
            b_nmse = self.run_esn(beta=b).nmse_test
            if b_nmse < hold_nmse_test:
                hold_nmse_test = b_nmse
                hold_beta = b
        if prnt:
            print("beta: {}\nnmse: {}".format(hold_beta,hold_nmse_test))
        return hold_beta, hold_nmse_test

    # returns alpha giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def opt_alpha(self,prnt=False):
        hold_nmse_test = 1
        hold_alpha = 1
        for a in self.alphas:
            a_nmse = self.run_esn(alpha=a).nmse_test
            if a_nmse < hold_nmse_test:
                hold_nmse_test = a_nmse
                hold_alpha = a
        if prnt:
            print("alpha: {}\nnmse: {}".format(hold_alpha,hold_nmse_test))
        return hold_alpha, hold_nmse_test

    # returns the rho-beta combination giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def rho_beta(self,prnt=False):
        hold_nmse_test = 1
        hold_rho = 1
        hold_beta = 1
        for r in self.rhos:
            for b in self.betas:
                rb_nmse = self.run_esn(rhoscale=r,beta=b).nmse_test
                if rb_nmse < hold_nmse_test:
                    hold_nmse_test = rb_nmse
                    hold_rho = r
                    hold_beta = b
        if prnt:
            print("rho: {}\nbeta: {}\nnmse: {}".format(hold_rho,hold_beta,hold_nmse_test))
        return hold_rho, hold_beta, hold_nmse_test

    # returns the rho-alpha combination giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def rho_alpha(self,prnt=False):
        hold_nmse_test = 1
        hold_rho = 1
        hold_alpha = 1
        for r in self.rhos:
            for a in self.alphas:
                ra_nmse = self.run_esn(rhoscale=r,alpha=a).nmse_test
                if ra_nmse < hold_nmse_test:
                    hold_nmse_test = ra_nmse
                    hold_rho = r
                    hold_alpha = a
        if prnt:
            print("rho: {}\nalpha: {}\nnmse: {}".format(hold_rho,hold_alpha,hold_nmse_test))
        return hold_rho, hold_alpha, hold_nmse_test

    # returns the beta-alpha combination giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def beta_alpha(self,prnt=False):
        hold_nmse_test = 1
        hold_beta = 1
        hold_alpha = 1
        for b in self.betas:
            for a in self.alphas:
                ba_nmse = self.run_esn(beta=b,alpha=a).nmse_test
                if ba_nmse < hold_nmse_test:
                    hold_nmse_test = ba_nmse
                    hold_beta = b
                    hold_alpha = a
        if prnt:
            print("beta: {}\nalpha: {}\nnmse: {}".format(hold_beta,hold_alpha,hold_nmse_test))
        return hold_beta, hold_alpha, hold_nmse_test

    # returns the rho-beta-alpha-nmse combination giving the lowest nmse for the given data in my esn
    # pass prnt=True to print results in words
    def rho_beta_alpha(self,prnt=False):
        print(Fore.YELLOW + "WARNING: Make sure you are storing the output of this function in a single variable e.g. 'rban' or unpacking directly into 4 variables e.g. 'r, b, a, n'." + Fore.RESET)
        hold_nmse_test = 1
        hold_rho = 1
        hold_beta = 1
        hold_alpha = 1
        i = 0
        for r in self.rhos:
            for b in self.betas:
                for a in self.alphas:
                    rba_nmse = self.run_esn(rhoscale=r,beta=b,alpha=a).nmse_test
                    if rba_nmse < hold_nmse_test:
                        hold_nmse_test = rba_nmse
                        hold_rho = r
                        hold_beta = b
                        hold_alpha = a
            i += 1
            print("{}%".format(floor(i/len(self.rhos)*100)))
        if prnt:
            print("rho: {}\nbeta: {}\nalpha: {}\nnmse: {}".format(hold_rho,hold_beta,hold_alpha,hold_nmse_test))
        return hold_rho, hold_beta, hold_alpha, hold_nmse_test
    
    #----------------------------------------------------------------------------------------------------------------------------
    # Collect parameters into dictionaries {param:nmse}

    # returns a dictionary of rho:nmse values for given data and rho parameter space
    def collect_rhos(self):
        print("Collecting rhos...")
        rs = {}
        i = 0
        j = 0   # used only for % printing
        for r in self.rhos:
            rs[i] = self.run_esn(rhoscale=r).nmse_test
            i += 1
            j += 1
            if j >= len(self.rhos)/5:
                print("{}%".format(floor(i/len(self.rhos)*100)))
                j = 0
        return rs

    # returns a dictionary of beta:nmse values for given data and beta parameter space
    def collect_betas(self):
        print("Collecting betas...")
        bs = {}
        i = 0
        j = 0   # used only for % printing
        for b in self.betas:
            bs[i] = self.run_esn(beta=b).nmse_test
            i += 1
            j += 1
            if j >= len(self.betas)/5:
                print("{}%".format(floor(i/len(self.betas)*100)))
                j = 0
        return bs

    # returns a dictionary of alpha:nmse values for given data and alpha parameter space
    def collect_alphas(self):
        print("Collecting alphas...")
        als = {}
        i = 0
        j = 0   # used only for % printing
        for a in self.alphas:
            als[i] = self.run_esn(alpha=a).nmse_test
            i += 1
            j += 1
            if j >= len(self.alphas)/5:
                print("{}%".format(floor(i/len(self.alphas)*100)))
                j = 0
        return als
    
    #----------------------------------------------------------------------------------------------------------------------------
    # i wrote these but they aren't necessary or particularly useful

    # returns a dictionary of rho:nmse values using collect_rhos averaged over "runs"
    def average_rho(self, runs=3):
        rs = {}
        for i in range(len(self.rhos)):
            rs[i] = 0
        for j in range(runs):
            col = self.collect_rhos()
            for key, val in col.items():
                rs[key] += val/runs
            print("{}%".format((j+1)/runs*100))
        return rs

    # returns a dictionary of beta:nmse values using collect_betas averaged over "runs"
    def average_beta(self, runs=3):
        bs = {}
        for i in range(len(self.betas)):
            bs[i] = 0
        for j in range(runs):
            col = self.collect_betas()
            for key, val in col.items():
                bs[key] += val/runs
            print("{}%".format((j+1)/runs*100))
        return bs

    # returns a dictionary of alpha:nmse values using collect_alphas averaged over "runs"
    def average_alpha(self, runs=3):
        als = {}
        for i in range(len(self.alphas)):
            als[i] = 0
        for j in range(runs):
            col = self.collect_alphas()
            for key, val in col.items():
                als[key] += val/runs
        return als
    
    #----------------------------------------------------------------------------------------------------------------------------
    # Plotting functions
    
    # plot nmse against rho.
    # param_sf/nmse_sf sets the significant figures shown on the axes ticks; pass sf=0 for order of magnitude only
    # num_ticks params set number of axes ticks if too many data points
    def plot_rhos(self, param_nmse=None, param_sf=3, nmse_sf=2, num_x_ticks=10, num_y_ticks=8):
        # initialise variables
        ns = (param_nmse if param_nmse else self.collect_rhos())
        nmses = list(ns.values())
        len_nmses = len(nmses)
        x_values = self.rhos
        y_values = log(nmses)
        # deals with the axes ticks
        n_x_ticks = (len_nmses if len_nmses<=10 else num_x_ticks)
        x_locs = lin_locs(self.rhos,num_x_ticks)
        x_ticks = param_ticks(x_locs,param_sf)
        y_locs = log_locs(nmses,num_y_ticks)
        y_ticks = param_ticks(y_locs,nmse_sf)
        y_locs = log(y_locs)
        # makes the optimal point a colour
        min_n = min(nmses)
        for k,v in ns.items():
            if v == min_n:
                min_k = k
        my_colors = {min_k:"tab:orange"}
        vocab = range(len_nmses)
        print(Fore.GREEN + "Optimal rho: {}".format(self.rhos[min_k]) + Fore.RESET)
        # plotting
        for i in range(len_nmses):
            plt.scatter(x_values[i], y_values[i], color=my_colors.get(vocab[i], 'black'))
        plt.xticks(x_locs, x_ticks)
        plt.yticks(y_locs, y_ticks)
        plt.xlabel("rho")
        plt.ylabel("nmse")
        plt.show()
    
    # plot nmse against beta.
    # param_sf/nmse_sf sets the significant figures shown on the axes ticks; pass sf=0 for order of magnitude only
    # num_ticks params set number of axes ticks if too many data points
    def plot_betas(self, param_nmse=None, param_sf=2, nmse_sf=2, num_x_ticks=10, num_y_ticks=8):
        # initialise variables
        ns = (param_nmse if param_nmse else self.collect_betas())
        nmses = list(ns.values())
        len_nmses = len(nmses)
        x_values = log(self.betas)
        y_values = log(nmses)
        # deals with the axes ticks
        n_x_ticks = (len_nmses if len_nmses<=10 else num_x_ticks)
        x_locs = log_locs(self.betas,n_x_ticks)
        x_ticks = param_ticks(x_locs,param_sf)
        x_locs = log(x_locs)
        y_locs = log_locs(nmses,num_y_ticks)
        y_ticks = param_ticks(y_locs,nmse_sf)
        y_locs = log(y_locs)
        # makes the optimal point a colour
        min_n = min(nmses)
        for k,v in ns.items():
            if v == min_n:
                min_k = k
        my_colors = {min_k:"tab:orange"}
        vocab = range(len_nmses)
        print(Fore.GREEN + "Optimal beta: {}".format(self.betas[min_k]) + Fore.RESET)
        # plotting
        for i in range(len_nmses):
            plt.scatter(x_values[i], y_values[i], color=my_colors.get(vocab[i], 'black'))
        plt.xticks(x_locs, x_ticks)
        plt.yticks(y_locs, y_ticks)
        plt.xlabel("beta")
        plt.ylabel("nmse")
        plt.show()
        
    # plot nmse against alpha.
    # param_sf/nmse_sf sets the significant figures shown on the axes ticks; pass sf=0 for order of magnitude only
    # num_ticks params set number of axes ticks if too many data points
    def plot_alphas(self, param_nmse=None, param_sf=2, nmse_sf=2, num_x_ticks=10, num_y_ticks=8):
        # initialise variables
        ns = (param_nmse if param_nmse else self.collect_alphas())
        nmses = list(ns.values())
        len_nmses = len(nmses)
        x_values = self.alphas
        y_values = log(nmses)
        # deals with the axes ticks
        n_x_ticks = (len_nmses if len_nmses<=10 else num_x_ticks)
        x_locs = lin_locs(self.alphas,num_x_ticks)
        x_ticks = param_ticks(x_locs,param_sf)
        y_locs = log_locs(nmses,num_y_ticks)
        y_ticks = param_ticks(y_locs,nmse_sf)
        y_locs = log(y_locs)
        # makes the optimal point a colour
        min_n = min(nmses)
        for k,v in ns.items():
            if v == min_n:
                min_k = k
        my_colors = {min_k:"tab:orange"}
        vocab = range(len_nmses)
        print(Fore.GREEN + "Optimal alpha: {}".format(self.alphas[min_k]) + Fore.RESET)
        # plotting
        for i in range(len_nmses):
            plt.scatter(x_values[i], y_values[i], color=my_colors.get(vocab[i], 'black'))
        plt.xticks(x_locs, x_ticks)
        plt.yticks(y_locs, y_ticks)
        plt.xlabel("alpha")
        plt.ylabel("nmse")
        plt.show()
        
#------------------------------------------------------------------------------------------------------------------------------------
# Recursion

# recursive function locates the optimal rho given an optimizer with rho range over [recursions] iterations
# pass [recursions] to set number of recursions to do
# pass [search_size] to set the number of values in the rho search space at each recursion
# returns the initial optimizer with optimized rho and rhos
def rho_recursive(optimizer, recursions=3, search_size=10):
    opt = optimizer
    if recursions:
        print("rho: {}".format(opt.get_rho()))
        print("{} recursions left".format(recursions))
        #print("rhos: {}".format(opt.get_rhos()))
        print(Fore.YELLOW + "RUNNING...\n" + Fore.RESET)
        rhos_width_factor = (max(opt.get_rhos())-min(opt.get_rhos()))/3
        b_opt, _ = opt.opt_rho()
        opt.set_rho(b_opt)
        if recursions > 1:
            opt.set_rhos(linspace(b_opt-rhos_width_factor, b_opt+rhos_width_factor, search_size))
        #print("new rho: {}\n".format(opt.get_rho()))
        #print("new rhos: {}\n".format(opt.get_rhos()))
        return rho_recursive(opt, recursions-1, search_size=search_size)
    else:
        print(Fore.GREEN + "~ RHO RECURSION COMPLETED: rho = {} ~\n".format(opt.get_rho()) + Fore.RESET)
        return opt

# recursive function locates the optimal beta given an optimizer with beta range over [recursions] iterations
# pass [recursions] to set number of recursions to do
# pass [search_size] to set the number of values in the beta search space at each recursion
# returns the initial optimizer with optimized beta and betas
def beta_recursive(optimizer, recursions=3, search_size=10):
    opt = optimizer
    if recursions:
        print("beta: {}".format(opt.get_beta()))
        print("{} recursions left".format(recursions))
        #print("betas: {}".format(opt.get_betas()))
        print(Fore.YELLOW + "RUNNING...\n" + Fore.RESET)
        betas_width_factor = (max(log10(opt.get_betas()))-min(log10(opt.get_betas())))/3
        b_opt, _ = opt.opt_beta()
        opt.set_beta(b_opt)
        if recursions > 1:
            opt.set_betas(logspace(log10(b_opt)-betas_width_factor, log10(b_opt)+betas_width_factor, search_size))
        #print("new beta: {}\n".format(opt.get_beta()))
        #print("new betas: {}\n".format(opt.get_betas()))
        return beta_recursive(opt, recursions-1, search_size=search_size)
    else:
        print(Fore.GREEN + "~ BETA RECURSION COMPLETED: beta = {} ~\n".format(opt.get_beta()) + Fore.RESET)
        return opt

# recursive function locates the optimal alpha given an optimizer with alpha range over [recursions] iterations
# pass [recursions] to set number of recursions to do
# pass [search_size] to set the number of values in the alpha search space at each recursion
# returns the initial optimizer with optimized alpha and alphos
def alpha_recursive(optimizer, recursions=3, search_size=10):
    opt = optimizer
    if recursions:
        print("alpha: {}".format(opt.get_alpha()))
        print("{} recursions left".format(recursions))
        #print("alphas: {}".format(opt.get_alphas()))
        print(Fore.YELLOW + "RUNNING...\n" + Fore.RESET)
        alphas_width_factor = (max(opt.get_alphas())-min(opt.get_alphas()))/3
        b_opt, _ = opt.opt_alpha()
        opt.set_alpha(b_opt)
        if recursions > 1:
            opt.set_alphas(linspace(max((b_opt-alphas_width_factor),0), min((b_opt+alphas_width_factor),1), search_size))
        #print("new alpha: {}\n".format(opt.get_alpha()))
        #print("new alphas: {}\n".format(opt.get_alphas()))
        return alpha_recursive(opt, recursions-1, search_size=search_size)
    else:
        print(Fore.GREEN + "~ ALPHA RECURSION COMPLETED: alpha = {} ~\n".format(opt.get_alpha()) + Fore.RESET)
        return opt

# performs all 3 recursive parameter optimizations sequentially
def full_recursive(optimizer, recursions=3, search_size=10):
    opt_rba = rho_recursive(optimizer, recursions, search_size)
    opt_rba = alpha_recursive(opt_rba, recursions, search_size)
    opt_rba = beta_recursive(opt_rba, recursions, search_size)
    return opt_rba
    
#------------------------------------------------------------------------------------------------------------------------------------
# Plotting support functions
    
# generates nicer looking axes ticks for the plots, pass sf=0 for order of magnitude only
def param_ticks(params, sf=3):
    p_ticks = []
    if sf == 0:
        for p in params:
            if 1<p<10:
                p_ticks.append("1")
            else:
                p_ticks.append("1e"+str(floor(log10(p))))
    elif sf>=1:
        for p in params:
            if 0.1<=p<1:
                p_ticks.append(round(p,sf))
            elif 1<=p<10:
                p_ticks.append(round(p,sf-1))
            elif p<0.1:
                log_floor = floor(log10(max(1e-10,p)))
                p_ticks.append(str(round((p*(10**abs(log_floor))),sf-1))+"e"+str(log_floor))
            else:
                log_floor = floor(log10(p))
                p_ticks.append(str(round((p*(10**(-abs(log_floor)))),sf-1))+"e"+str(log_floor))
    return p_ticks

# returns [num] tick locations for a logarithmic domain
def log_locs(params,num=10):
    return logspace(log10(min(params)),log10(max(params)),num)

# returns [num] tick locations for a linear domain
def lin_locs(params,num=10):
    return linspace(min(params),max(params),num)