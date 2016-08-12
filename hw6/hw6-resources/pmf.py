import numpy as np
from scipy.sparse import csr_matrix,rand,coo_matrix
import numpy.linalg as LA
import time
import sys
import math


path = "preprocess_train_matrix_fact"
# @profile
def matrix_fact(rating_matrix, U, V, k,alpha, lambda_u = 0.01, lambda_v=0.01):
    '''
    :param rating_matrix:
    :param U: User matrix factor MxK
    :param V: Movie matrix kXN
    :param k: Number of factors
    :param alpha: learning rate
    :param lambda_u:
    :param lambda_v:
    :return: new rating matrix
    '''
    r,c = rating_matrix.nonzero()
    I = csr_matrix((np.ones(r.shape[0]),(r,c)), shape=rating_matrix.shape)
    rating_matrix.data = rating_matrix.data - 3
    R = coo_matrix(rating_matrix)

    dot = U.dot(V.T)

    original_alpha = alpha
    err_matrix = rating_matrix - dot

    print "Lambda_U = %f"%(lambda_u)
    print "Lambda_V = %f"%(lambda_v)


    E = (1.0/2)*(np.sum(I.multiply(err_matrix).data**2)) + (lambda_u/2.0) *LA.norm(U,"fro")**2 + \
            (lambda_v/2.0)* LA.norm(V,"fro")**2
    original_E = E
    print "Initial error: %.2f"%(E)
    old_E = None
    it = 0
    threshold = 0.0001
    last_down = 0
    epoch = 0
    try:
        while epoch < 50:
            epoch +=1
            alpha = original_alpha

            for row,col,value in zip(R.row,R.col,R.data):

                
                U[row,:] = U[row,:] + alpha*((value -np.dot(U[row,:],V[col,:].T))*V[col,:] - lambda_u*U[row,:])
                V[col,:] =  V[col,:] + alpha*((value -np.dot(U[row,:],V[col,:].T))* U[row,:] - lambda_v*V[col,:])


            dot = U.dot(V.T)
            err_matrix = rating_matrix - dot
            new_E = (1.0/2)*(np.sum(I.multiply(err_matrix).data**2)) + (lambda_u/2.0) *LA.norm(U,"fro")**2 + \
                (lambda_v/2.0)* LA.norm(V,"fro")**2
            print "Epoch %d"%(epoch)
            print "Cost: %.2f"%(new_E)


            if original_E - new_E < 10 and alpha > 1e-10:
                alpha/=2
            if original_E/new_E > 1.0 + threshold:
                original_E = new_E
            else:
                break

    except KeyboardInterrupt:
        print "Stopped."
        pass
    print "Total of %d iterations."%(it)
    print "lambda_u = " + str(lambda_u)
    print "lambda_v = " + str(lambda_v)
    print "alpha = " + str(alpha)
    dot = U.dot(V.T) + 3
    return dot,it,U,V



