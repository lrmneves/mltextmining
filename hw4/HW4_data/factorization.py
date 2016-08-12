__author__ = 'lrmneves'

import numpy as np
from scipy.sparse import csr_matrix,rand
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy.linalg as LA
from scipy.special import expit

import time
import sys
import math


def sparse_corrcoef(matrix):
    # (adapted from http://stackoverflow.com/questions/16062804/)
    matrix = matrix - matrix.mean(1)
    denom = matrix.shape[1] - 1.
    cof = matrix.dot(matrix.T.conjugate()) / denom
    diag = np.diag(cof)
    coeffs = cof / np.sqrt(np.outer(diag, diag))

    return coeffs

def normalize_rating(value):
    return (((value - (-2.0))/(2-(-2))) *4) + 1

def matrix_fact(rating_matrix, U, V, k,alpha=0.0002):
    r,c = rating_matrix.nonzero()
    I = csr_matrix((np.ones((r.shape[0])),(r,c)), shape=rating_matrix.shape)

    dot = U.dot(V)
    err_matrix = rating_matrix - dot
    err_matrix_copy = err_matrix.copy()
    err_matrix.data = err_matrix.data**2
    err_matrix = I.multiply(err_matrix)

    r_var = np.var(rating_matrix.data)
    u_var = np.var(U.data)
    v_var = np.var(V.data)




    lambda_u = r_var/u_var
    lambda_v = r_var/v_var


    E = (1.0/2)*(np.sum(err_matrix.data)) + (lambda_u/2.0) * LA.norm(U.data) + (lambda_v/2.0)* LA.norm(V.data)

    old_E = None

    while old_E == None or abs(E - old_E) > 0.001:

        new_U = U + alpha*(2*err_matrix_copy*V.T  - lambda_u*U )
        new_V = V + alpha*(2*U.T*err_matrix_copy - lambda_v*V)

        U = new_U
        V = new_V
        old_E = E

        u_var = np.var(U.data)
        v_var = np.var(V.data)


        lambda_u = r_var/u_var
        lambda_v = r_var/v_var

        dot = U.dot(V)


        err_matrix = rating_matrix - dot
        err_matrix_copy = err_matrix.copy()

        err_matrix.data = err_matrix.data**2
        err_matrix = I.multiply(err_matrix)
        E = (1.0/2)*(np.sum(err_matrix.data)) + (lambda_u/2.0) * LA.norm(U.data) + \
            (lambda_v/2.0)* LA.norm(V.data)
        print E




def getkNN(value,k ,vector,cached_kNN):
    if not value in cached_kNN:

        kNN = zip(vector.data.tolist(),vector.nonzero()[1].tolist())

        kNN.sort(reverse=True)

        kNN = [v for v in kNN[1:k+1]]
        cached_kNN[value] = kNN

    else:
        kNN = cached_kNN[value]
    return kNN

def get_movie_rating_model_based(proc_matrix,rating_matrix,user,movie,k,method="dot",weighted = False,cached_kNN = {}):
    if movie >= rating_matrix.shape[1]:
        return 3.0

    movie_vector =  proc_matrix[movie,:]
    kNN = getkNN(movie,k ,movie_vector,cached_kNN)

    total_rating = 0.0
    rating_count = 0.0
    current = 0
    if len(kNN) == 0:
        return 3.0


    if weighted:
        cos_norm = sum([ kNN[x][0] for x in range(k)])
    while rating_count < k and current < len(kNN):
        try:
            factor = 1.0
            if weighted:
                factor =  kNN[current][0]/cos_norm


            total_rating += ((rating_matrix[user,kNN[current][1]])) * factor + 3


            rating_count+=1
        except IndexError:
            total_rating+= 3
            rating_count+=1
        current +=1

    if rating_count == 0:
        return 3.0

    return float(total_rating/current)


def get_movie_rating_memory_based(proc_matrix,rating_matrix,user,movie,k,method="dot",weighted = False,cached_kNN = {},\
                                  cov = None):
    user_vector = proc_matrix[user,:]



    kNN = getkNN(user,k ,user_vector,cached_kNN)

    total_rating = 0.0
    rating_count = 0.0
    current = 0

    if weighted:
        cos_norm = sum([ kNN[x][0] for x in range((k if k < len(kNN) else len(kNN)))])
        if cos_norm == 0:
            return 3.0


    while rating_count < k:
        try:
            factor = 1.0
            if weighted:
                factor =  kNN[current][0]/cos_norm
            total_rating += (rating_matrix[kNN[current][1],movie]) * factor +3

            rating_count+=1
        except IndexError:
            total_rating+= 3.0
            rating_count+=1
        current +=1

    if rating_count == 0:
        return 3.0
    return float(total_rating/rating_count)

def main():
    path = sys.argv[1]
    print "loading training"
    with open(path) as train:
        rows = []
        cols = []
        data = []
        max_row = 0
        max_col = 0
        for line in train.readlines():
            current = line.split(" ")
            r = int(current[0])
            if r > max_row:
                max_row = r
            for movie in current[1].split(","):
                #movie_list[0] = movie, movie_list[1] = rating
                movie_list = movie.split("#")
                rows.append(r)
                c = int(movie_list[0])
                cols.append(c)
                if c > max_col:
                    max_col = c
                data.append(float(movie_list[1]))

        rating_matrix =csr_matrix((data,(rows,cols)), shape=(max_row+1,max_col+1))


    if len(sys.argv) < 2:


        user_4321_vector = rating_matrix[4321,:].copy()

        k = 5


        kNN_dot_prod = []
        kNN_cos_sim = []
        dot_similarity = rating_matrix.dot(user_4321_vector.T).todense()
        cos_similarity = cosine_similarity(rating_matrix,user_4321_vector)

        for i in range(dot_similarity.shape[0]):
            kNN_dot_prod.append((dot_similarity[i][0],i))
            kNN_cos_sim.append((cos_similarity[i][0],i))

        kNN_dot_prod.sort(reverse = True)
        kNN_cos_sim.sort(reverse=True)
        print "User 4321 Dot Product Similarity:" + str([v[1] for v in kNN_dot_prod[1:k+1]])
        print "User 4321 Cosine Similarity: " + str([v[1] for v in kNN_cos_sim[1:k+1]])

        rating_matrix = rating_matrix.T


        movie_3_vector = rating_matrix[3,:].copy()

        kNN_dot_prod = []
        kNN_cos_sim = []
        dot_similarity = rating_matrix.dot(movie_3_vector.T).todense()
        cos_similarity = cosine_similarity(rating_matrix,movie_3_vector)

        for i in range(dot_similarity.shape[0]):
            kNN_dot_prod.append((dot_similarity[i][0],i))
            kNN_cos_sim.append((cos_similarity[i][0],i))

        kNN_dot_prod.sort(reverse = True)
        kNN_cos_sim.sort(reverse=True)
        print "Movie 3 Dot Product Similarity:"  + str([v[1] for v in kNN_dot_prod[1:k+1]])
        print "Movie 3 Cosine Similarity: " + str([v[1] for v in kNN_cos_sim[1:k+1]])

    else:

        experiment = sys.argv[2]
        experiment_set = sys.argv[3]
        w=False
        k= int(sys.argv[4])
        method = "dot"
        if len(sys.argv) > 5:
                method = sys.argv[5]
        if len(sys.argv) > 6:
            if sys.argv[6] == "weighted":
                w = True
        s = time.time()
        with open(experiment_set) as exp_set:
            lines = exp_set.readlines()
        results = []
        cache = {}
        print "starting " + str(experiment)+"_"+method+"_"+str(k) +("_weighted" if w  else "")


        if int(experiment) == 1 or int(experiment) == 3:

            if method.startswith("cos"):
                rating_matrix_proc = normalize(rating_matrix)
                rating_matrix_proc = rating_matrix_proc.dot(rating_matrix_proc.T)
            else:
                rating_matrix_proc = rating_matrix.dot(rating_matrix.T)

            if int(experiment) == 3:
                cov = sparse_corrcoef(rating_matrix)
                cov[np.isnan(cov)] = 0
                rating_matrix_proc = csr_matrix(rating_matrix_proc.multiply(cov))


            for line in lines:

                current = line.split(",")
                user = int(current[0])
                movie = int(current[1])


                results.append(get_movie_rating_memory_based(rating_matrix_proc,rating_matrix,user,movie, \
                                                             k,weighted=w,cached_kNN=cache))


        elif int(experiment) == 2:
            if method.startswith("cos"):
                rating_matrix_proc = normalize(rating_matrix.T)
                rating_matrix_proc = rating_matrix_proc.dot(rating_matrix_proc.T)
            else:
                rating_matrix_proc = rating_matrix.T.dot(rating_matrix)

            for line in lines:
                current = line.split(",")
                user = int(current[0])
                movie = int(current[1])

                results.append(get_movie_rating_model_based(rating_matrix_proc,rating_matrix,user,movie, \
                                                            k,weighted=w,cached_kNN=cache))
        elif int(experiment) == 4:


            U = csr_matrix(rand(rating_matrix.shape[0],k))
            U.data = U.data *4 + 1


            V = csr_matrix(rand(rating_matrix.shape[1],k)).T
            V.data = V.data*4 + 1

            matrix_fact(rating_matrix, U, V, k,alpha=0.0002)



        e = time.time()
        print "Total Time: %d"%(e-s)
        with open("timings.txt","a") as timings:
            timings.write(str(e-s) + " " + str(experiment)+"_"+method+"_"+str(k) +("_weighted\n" if w  else "\n"))
        with open("results_"+str(experiment)+"_"+method+"_"+str(k) +("_weighted" if w  else "")+".txt","a") as result_f
            for r in results:
                result_file.write(str(r)+ "\n")

if __name__ == "__main__":
    main()