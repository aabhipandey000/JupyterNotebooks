import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
from sklearn import svm
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import mean_squared_error
from seaborn import heatmap as hmap
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
import warnings
import math
from sklearn.preprocessing import StandardScaler

##Manual functions for PRML assignment
class prml():
    """
    Class containing all the manual functions written for prml class.
    Initialize by: prml(X_train,y_train),
    By default add_intercept = True, it will add ones vector at the beginning of the feature matrix.
    Expected shape of X_train: (no of samples,no of features)
    Expected shape of y_train: (no of samples,)
    """
    
    #Initialize parameters
    def __init__(self,X_train,y_train,add_intercept = True):
        self.add_intercept = add_intercept
        if self.add_intercept:
            self.X = np.insert(X_train,0,1.0,axis=1)
        else:
            self.X  =X_train
        self.y = y_train
        self.alpha = np.zeros_like(self.y)
        self.gamma = 1
        self.coeff = 0
        self.degree = 1
        self.b = 0
        self.algo = None
        
    # Kernel used as linear as well as polynomial
    def _hybrid_kernel(self,X1, X2):          #By default linear kernel
        """
        Parameter:
        X1,X2 matrices
        gamma,coeff and degree are defined while training the model.
        returns
        Kernel matrix = (gamma(X1X2^T) + coeff)^degree
        """
        return (self.gamma*(X1 @ X2.T) + self.coeff)**self.degree
    
    # Sigmoid activation
    def _sigmoid(self,X):
        return 1/(1 + np.exp(-X))
    
    # Log loss calculation
    def _log_loss(self,K):
        y_h = self._sigmoid(K @ (self.alpha * self.y))
        return log_loss(self.y,y_h)  ##To avoid numpy array overload we used library function
    
    # Log loss gradient calculation: kernel variant
    def _log_loss_grad(self,K):
        y_h = self._sigmoid(K @ (self.alpha * self.y))
        return (self.y * (K @ (y_h - self.y)))/y_h.size
    
    # Log loss hessian calculation: kernel variant
    def _log_loss_hes(self,K):
        y_h = self._sigmoid(K @ (self.alpha * self.y))
        return (self.y.reshape(self.y.size,1) * (K @ (y_h * (1-y_h) * np.eye(y_h.shape[0])) @ K))/y_h.size
    
    # Kernel ridge regression
    def kernel_ridge_regression(self,lam = 1.0, gamma=1, coeff=0, degree = 1):
        """
        Training the kernel ridge regression on the input data to prml class.
        p = prml(X_train,y_train)
        p.kernel_ridge_regression() trains the model for the X_train and y_train
        
        Parameters:
        lam:- Regularization parameter (default = 1.0)
        gamma :- Kernel parameter (default = 1.0)
        coeff :- Kernel parameter (default = 0.0)
        degree :- Degree for polynomial kernel (default = 1.0)
        
        returns:
        dual coefficients of shape (no of samples,)
        """
        self.algo = 'kernel_ridge'
        (self.gamma,self.coeff,self.degree) = (gamma,coeff,degree)
        K = self._hybrid_kernel(self.X,self.X)
        a = (K+lam*np.eye(K.shape[0],K.shape[1])) @ K
        b = K.T @ self.y
        try:
            result = linalg.solve(a,b)
        except  np.linalg.LinAlgError or 'LinAlgWarning':
            warnings.warn("Singular matrix in solving dual problem. Using least-squares solution instead.")
            result = linalg.lstsq(a, b)[0]
        self.alpha =result
        return result
    
    # Kernel perceptron algorithm
    def kernel_perceptron(self,initial_alpha = None,T=100,lr =1,gamma=1, coeff=0, degree = 1,view =False,b = 0):
        """
        Training the kernel perceptron on the input data to prml class.
        p = prml(X_train,y_train)
        p.kernel_perceptron() trains the model for the X_train and y_train
        
        Parameters:
        T :- Max no of epochs (default = 100)
        lr :- learning rate (default = 1)
        b :- bias for perceptron decision function (default = 0)
        gamma :- Kernel parameter (default = 1.0)
        coeff :- Kernel parameter (default = 0.0)
        degree :- Degree for polynomial kernel (default = 1.0)
        view :- To view Accuracy vs Epochs for training (default = False)
        
        returns:
        dual coefficients of shape (no of samples,)
        no_of_mistakes while training the model
        bound - theoratical bound for perceptron algorithm
        rho - parameter to estimate bound
        w_norm - parameter to estimate bound
        r - parameter to estimate bound
        """
        self.algo = 'kernel_perceptron'
        (self.gamma,self.coeff,self.degree,self.b) = (gamma,coeff,degree,b)
        if str(type(initial_alpha)) == "<class 'NoneType'>":
            self.alpha = np.zeros_like(self.y)
        else:
            self.alpha = initial_alpha
        y = 2*self.y - 1
        a_score = []
        no_of_mistakes = 0
        for i in range(T):
            for j in range(y.size):
                y_hat = np.sign(((self.alpha*y).T @ self._hybrid_kernel(self.X,self.X[j,:])) + self.b)
                if y_hat != y[j]:
                    self.alpha[j]+=lr
                    no_of_mistakes+=1
            y_h = np.sign((self._hybrid_kernel(self.X,self.X) @ (self.alpha*y)) + self.b)
            a_score.append(acc_score(y,y_h))
        # Bound calculation
        K = self._hybrid_kernel(self.X,self.X)
        r = np.max(np.diag(K))  #r>0 always see report
        alpha_y = y*self.alpha
        w_norm = np.sqrt(alpha_y.T @ K @ alpha_y)
        if w_norm!=0:
            rho_choice = y *(alpha_y @ K)/w_norm
            rho = np.min([i for i in rho_choice if i > 0]) #To keep rho>0
            v = np.ones(self.X.shape[1])/np.sqrt(self.X.shape[1]) #Unit vector v
            d = rho - (y *(alpha_y @ K))
            d = d[d>0]
            delta = np.linalg.norm(d)
            bound = ((r+delta)/rho)**2
        else: # When w_norm = 0 replacing w with v. (See FOML book)
            v = np.ones(self.X.shape[1])/np.sqrt(self.X.shape[1])
            rho_choice = (y*(self.X @ v))
            rho = np.min([i for i in rho_choice if i > 0])
            d = rho - (y*(self.X @ v))
            d = d[d>0]
            delta = np.linalg.norm(d)
            bound = ((r+delta)/rho)**2
        # Visualizing convergence
        if view:
            plt.plot(np.arange(T),a_score,'bo')
            plt.xlabel('Epoches')
            plt.ylabel('Classification Accuracy')
            plt.title('Classification Accuracy vs Epoches')
        return self.alpha,no_of_mistakes,bound,rho,w_norm,r
    
    #Kernel logistic regression: optimized using gradient descent
    def kernel_logistic_grad_descent(self,itr=1000,lr = 0.01,gamma=1, coeff=0, degree = 1,view = False):
        """
        Training the kernel logistic on the input data to prml class which will be optimized by gradient descent.
        p = prml(X_train,y_train)
        p.kernel_logistic_grad_descent() trains the model for the X_train and y_train
        
        Parameters:
        itr :- no of iterations for gradient descent to be performed (default = 1000)
        lr :- learning rate (default = 0.01)
        gamma :- Kernel parameter (default = 1.0)
        coeff :- Kernel parameter (default = 0.0)
        degree :- Degree for polynomial kernel (default = 1.0)
        view :- To view MSE vs iterations for training (default = False)
        
        returns:
        dual coefficients of shape (no of samples,)
        J - log loss array for all iterations of gradient descent
        """
        self.algo = 'kernel_logistic'
        (self.gamma,self.coeff,self.degree) = (gamma,coeff,degree)
        self.alpha = np.zeros_like(self.y)
        K = self._hybrid_kernel(self.X ,self.X)
        J = []
        
        #Gradient Descent
        for i in range(itr):
            self.alpha = self.alpha - lr * self._log_loss_grad(K)
            J.append(self._log_loss(K))
        # Visualizing convergence
        if view:
            plt.plot(np.arange(0,itr),np.array(J),'bo')
            plt.xlabel('Iterations')
            plt.ylabel('Log_Loss')
            plt.title('Gradient descent on log loss')
            plt.show()
        return self.alpha,J
    #Kernel logistic regression: optimized using newton method
    def kernel_logistic_newton_method(self,itr=1000,gamma=1, coeff=0, degree = 1,lr = 1,view = False):
        """
        Training the kernel logistic on the input data to prml class which will be optimized by newton's  method.
        p = prml(X_train,y_train)
        p.kernel_logistic_newton_method() trains the model for the X_train and y_train
        
        Parameters:
        itr :- no of iterations for gradient descent to be performed (default = 1000)
        lr :- learning rate (default = 1)
        gamma :- Kernel parameter (default = 1.0)
        coeff :- Kernel parameter (default = 0.0)
        degree :- Degree for polynomial kernel (default = 1.0)
        view :- To view MSE vs iterations for training (default = False)
        
        returns:
        dual coefficients of shape (no of samples,)
        J - log loss array for all iterations of gradient descent
        """
        self.algo = 'kernel_logistic'
        (self.gamma,self.coeff,self.degree) = (gamma,coeff,degree)
        self.alpha = np.zeros_like(self.y)
        K = self._hybrid_kernel(self.X ,self.X)
        J = []
        # Newton's method
        for i in range(itr):
            try:
                self.alpha = self.alpha - lr * np.linalg.inv(self._log_loss_hes(K)) @ self._log_loss_grad(K)
            except np.linalg.LinAlgError :
                self.alpha = self.alpha - lr * np.linalg.pinv(self._log_loss_hes(K)) @ self._log_loss_grad(K)
            J.append(self._log_loss(K))
        # Visualizing convergence
        if view:
            plt.plot(np.arange(0,itr),np.array(J),'bo')
            plt.xlabel('Iterations')
            plt.ylabel('Log_Loss')
            plt.title('Newton_method on log loss')
            plt.show()
        return self.alpha,J
    
    # General predicting function after training the model
    def predict(self,X):
        """
        To predict the model output after training the model using particular algorithm
        p = prml(X_train,y_train)
        p.kernel_logistic()
        p.predict(X_test)
        
        Parameters:
        X: Feature matrix of shape (no of samples, no of features)
        
        returns:
        y: predicted vector of shape(no of samples,)
        
        Note: If add_intercept is initialized true while initializing the class then it will add intercept to the X passed in predict also.
        """
        if self.add_intercept:
            X = np.insert(X,0,1.0,axis=1)
        if self.algo == 'kernel_logistic':
            out = np.round(self._sigmoid((self._hybrid_kernel(X,self.X) @ (self.alpha*self.y))))
        elif self.algo == 'kernel_perceptron':
            y = 2*self.y - 1
            out = np.sign((self._hybrid_kernel(X,self.X) @ (self.alpha*y)) + self.b)
            out = (out+1)//2
        elif self.algo == 'kernel_ridge':
            out = self._hybrid_kernel(X,self.X) @ (self.alpha)
        return out
