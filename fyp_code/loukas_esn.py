import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error
plt.rcParams['figure.figsize'] = [20, 20]

def nmse(y_pred,y_true):
    return mean_squared_error(y_pred,y_true)/np.var(y_true)

class Esn:
    
    # Initialise global variables at object creation
    # pass fix_eig=False to NOT fix the eigs function to a deterministic result, otherwise it does
    def __init__(self,data,rhoscale=1.25,beta=10**-4,alpha=0.5,in_nodes=1,out_nodes=1,Ttrain=2000,Twashout=100,N=1000,sparsity=0,seed_init=0,fix_eig=True):
        # General variables
        np.random.seed(seed_init)
        self.alpha = alpha
        self.beta = beta
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.N = N
        self.Ttrain = Ttrain
        self.Twashout = Twashout
        
        # let's save the relevant time-steps for convenience
        self.T1=Twashout
        self.T2=Ttrain+Twashout
        
        # load and store the dataset
        self.data=data
        # include row of ones as an input to the network - data_input has shape [2,len(data)] with the top row as 1s
        self.data_input=np.concatenate((np.ones((1,data.shape[1])),data))
        
        # matrices
        self.x = np.zeros(N)            # column vector
        self.Win = np.random.uniform(low=-0.5, high=0.5, size=(N,in_nodes+1))
        self.W = np.random.uniform(low=-1.5, high=1.5, size=(N,N))
        self.Wout = np.zeros((out_nodes,in_nodes+N+1))
        
        # sparsify reservoir if sparsity > 0
        if sparsity:
            self.W = sparsify(sparsity,self.W)
        
        # init measurement matrix
        self.M_train = np.zeros((in_nodes+N+1,Ttrain))
        
        if fix_eig:
            eig,_=eigs(self.W,k=1,which='LM',tol=1e-8,v0=np.ones(N))
        else:
            eig,_=eigs(self.W,k=1,which='LM',tol=1e-8)
        self.rhoscale = rhoscale/abs(eig)                  # adjust rhoscale using largest eigen value
        self.W = self.W*self.rhoscale                      # scale spectral radius of W
        
    #}-------------------------------------------------------------------------------------------------------------------
    # TRAINING
        
    # Contruct M matrix
    def train_M(self):
        
        # store input and expected output in vectors
        self.X_train=self.data[:,self.T1:self.T2]      # input
        self.Y_train=self.data[:,self.T1+1:self.T2+1]  # correct output
        
        # sync internal revervoir states with input
        for i in range(self.Twashout):
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh(self.Win@self.data_input[:,i] + self.W@self.x)
        
        # train and extract M matrix
        for i in range(self.Ttrain):
            j=i+self.T1
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh(self.Win@self.data_input[:,j] + self.W@self.x)
            self.M_train[:,i] = np.concatenate((self.data_input[:,j],self.x.flatten()))
            
        # store the reservoir status after training
        self.x_store = self.x
    
    # Train Wout (provided M first)
    def train_readouts(self):
        Mt = self.M_train.transpose()
        D = self.Y_train
        self.Wout = D@Mt@inv(self.M_train@Mt + self.beta*np.identity(self.in_nodes+self.N+1))
        
        self.Yhat_train=self.Wout@self.M_train                  # perform output prediction on established data
        self.nmse_train=nmse(self.Yhat_train,D)                 # compute error on prediction with correct data
        #print("NMSE training:{:.2e}".format(self.nmse_train))
    
    def train(self):
        self.train_M()
        self.train_readouts()
        
    #}-------------------------------------------------------------------------------------------------------------------
    # VALIDATION
    
    # Validate prediction with forced system for T>Ttrain
    def validate(self, val_time=1000):
        
        self.val_time = val_time
        # store input and expected output in vectors
        self.X_val=self.data[:,self.T2:self.T2+val_time]        # val_time input steps starting at the point after training
        self.Y_val=self.data[:,self.T2+1:self.T2+val_time+1]    # correct output
        self.Yhat_val = np.zeros((self.out_nodes, val_time))    # init prediction matrix
        
        self.x=self.x_store                                     # return reservoir to freshly trained state
        
        # Test the network on unseen but similar data to training
        for i in range(val_time):
            j=i+self.T2
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh(self.Win@self.data_input[:,j]+ self.W@self.x)
            y = self.Wout@np.concatenate((self.data_input[:,j],self.x))
            self.Yhat_val[:,i] = y
        
        # compute error on prediction with correct data
        self.nmse_val=nmse(self.Yhat_val,self.Y_val)
        #print("NMSE validation:{:.2e}".format(self.nmse_val))

    #}-------------------------------------------------------------------------------------------------------------------
    # TESTING
    
    # Test function given a starting data_point and test time
    # Returns predicted data array
    def test(self,test_time=1000):
        
        self.test_time = test_time
        # store input and expected output in vectors
        self.X_test=self.data[:,self.T2:self.T2+test_time]      # test_time input steps starting at the point after training
        self.Y_test=self.data[:,self.T2+1:self.T2+test_time+1]  # correct output
        self.Yhat_test = np.zeros((self.out_nodes,test_time))   # init prediction matrix
        
        self.x=self.x_store
        y = self.data[:,self.T2]
        for i in range(test_time):
            u=np.concatenate([[1.],y])
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh(self.Win@u + self.W@self.x)
            y = self.Wout@np.concatenate((u,self.x))
            self.Yhat_test[:,i] = y
        
        self.nmse_test=nmse(self.Yhat_test,self.Y_test)
        #print("NMSE test:{:.2e}".format(self.nmse_test))
    
    #}-------------------------------------------------------------------------------------------------------------------
    # PLOTTING (copied)

    # Plot the static properties of the ESN
    def plot_static_properties(self):
        fig,_=plt.subplots(nrows=1, ncols=2)
        fig.tight_layout()
        plt.subplot(1,2,1)
        plt.plot(self.Win.transpose())
        plt.title('Win')
        
        plt.subplot(1,2,2)
        plt.pcolor(self.W)
        plt.title('W')
        
    # Plot M matrix
    def plot_M(self):
        fig,_=plt.subplots(nrows=1, ncols=1)
        plt.subplot(3,1,1)
        plt.pcolor(self.M,cmap='RdBu')
        plt.title('M')
    
    # Plot the training data
    def plot_training(self):
        fig,_=plt.subplots(nrows=2*self.out_nodes, ncols=1)
        fig.tight_layout()
        plt.subplot(2*self.out_nodes,1,1)
        plt.title('Training')    
        for i in range(self.out_nodes):
            plt.subplot(2*self.out_nodes,1,1+i*self.out_nodes)
            plt.plot(self.Yhat_train[i,:].T,label='prediction training dataset')
            plt.plot(self.Y_train[i,:].T,label='training dataset')
            plt.legend()

            plt.subplot(2*self.out_nodes,1,2+i*self.out_nodes)
            yy=(self.Yhat_train[i,:]-self.Y_train[i,:])**2
            plt.plot(yy.T,label='NMSE training')
            plt.legend()
            
    # Plot the validation data
    def plot_validation(self):
        fig,_=plt.subplots(nrows=2*self.out_nodes, ncols=1)
        fig.tight_layout()
        plt.subplot(2*self.out_nodes,1,1)
        plt.title('Validation')
            
        for i in range(self.out_nodes):
            plt.subplot(2*self.out_nodes,1,1+i*self.out_nodes)
            plt.plot(self.Yhat_val[i,:].T,label='prediction validation dataset')
            plt.plot(self.Y_val[i,:].T,label='validation dataset')
            plt.legend()

            plt.subplot(2*self.out_nodes,1,2+i*self.out_nodes)
            yy=(self.Yhat_val[i,:]-self.Y_val[i,:])**2
            plt.plot(yy.T,label='NMSE validation')
            plt.legend()
    
    # Plot the test data
    def plot_test(self):        
        fig,_=plt.subplots(nrows=2*self.out_nodes, ncols=1)
        fig.tight_layout()
        plt.subplot(2*self.out_nodes,1,1)
        plt.title('Test')
        for i in range(self.out_nodes):
            plt.subplot(2*self.out_nodes,1,1+i*self.out_nodes)
            plt.plot(self.Yhat_test[i,:].T,label='prediction test dataset')
            plt.plot(self.Y_test[i,:].T,label='test dataset')
            plt.legend()

            plt.subplot(2*self.out_nodes,1,2+i*self.out_nodes)
            yy=(self.Yhat_test[i,:]-self.Y_test[i,:])**2
            plt.plot(yy.T,label='NMSE test')
            plt.legend()
    
    def simple_plot_train(self,x_size=10,y_size=5,dpi=100,linewidth=1):
        
        figure(figsize=(x_size, y_size),dpi=dpi)         # set figure size
        
        xs = np.linspace(1,self.Ttrain,self.Ttrain)      # initialize variables
        ys = self.Yhat_train[0,:]
        correct = self.Y_train[0,:]
        
        plt.plot(xs,ys,label="esn",linewidth=linewidth)
        plt.plot(xs,correct,label="input",linewidth=linewidth)
        plt.title("Training")
        plt.legend()
        
    def simple_plot_validation(self,x_size=10,y_size=5,dpi=100,linewidth=1):
        
        figure(figsize=(x_size, y_size),dpi=dpi)         # set figure size
        
        xs = np.linspace(1,self.val_time,self.val_time)   # initialize variables
        ys = self.Yhat_val[0,:]
        correct = self.Y_val[0,:]
        
        plt.plot(xs,ys,label="esn",linewidth=linewidth)
        plt.plot(xs,correct,label="input",linewidth=linewidth)
        plt.title("Validation")
        plt.legend()
        
    def simple_plot_test(self,x_size=10,y_size=5,dpi=100,linewidth=1):
        
        figure(figsize=(x_size, y_size),dpi=dpi)         # set figure size
        
        xs = np.linspace(1,self.test_time,self.test_time)   # initialize variables
        ys = self.Yhat_test[0,:]
        correct = self.Y_test[0,:]
        
        plt.plot(xs,ys,label="esn",linewidth=linewidth)
        plt.plot(xs,correct,label="input",linewidth=linewidth)
        plt.title("Testing")
        plt.legend()
        
    def get_all(self):
        return {"alpha":self.alpha, "beta":self.beta, "in_nodes":self.in_nodes, "out_nodes":self.out_nodes, "N":self.N, "Ttrain":self.Ttrain, "Twashout":self.Twashout, "T1":self.T1, "T2":self.T2, "data":self.data, "data_input":self.data_input, "x":self.x, "Win":self.Win, "W":self.W, "Wout":self.Wout, "M_train":self.M_train, "rho":self.rhoscale}

#}-----------------------------------------------------------------------------------------------------------------------
    
# Reduce an array to a sparse version (entries = 0) proportional to [sparsity]
# sparsity = 0 means no change; sparsity = 1 means all zeros
def sparsify(sparsity, array):
    assert 0<=sparsity<=1, "sparsity must be between 0 and 1"
    L = array.size                                 # store total length of original array
    ones = np.ones(int(L*(1-sparsity)))            # generate array of ones
    zeros = np.zeros(int(L*sparsity))              # generate array of zeros
    sparse = np.concatenate([ones,zeros])          # concatenate ones and zeros into sparsity array
    np.random.shuffle(sparse)                      # randomize order of ones and zeros
    if sparse.size<L:                              # sometimes one more entry is needed due to loss when rounding using int()
        sparse = np.append(sparse,1.)
    assert sparse.size == L, "wrong size sparsity array"
    sparse = sparse.reshape(array.shape)           # reshape 1D sparsity array into original array.shape
    return np.multiply(array,sparse)               # elementwise mulitiplication - entries in original array eliminated where zero in sparsity array
