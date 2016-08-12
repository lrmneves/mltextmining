__author__ = 'lrmneves'

import numpy as np
from scipy.sparse import csr_matrix,rand
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import norm
from scipy.special import expit
import time
import sys


def sparse_corrcoef(matrix):
    '''Function to compute PCC'''
    # (adapted from http://stackoverflow.com/questions/16062804/)
    matrix = matrix - matrix.mean(1)
    denom = matrix.shape[1] - 1.
    cof = matrix.dot(matrix.T.conjugate()) / denom
    diag = np.diag(cof)
    denom = np.sqrt(np.outer(diag, diag))
    denom[np.isnan(denom)] = 1
    coeffs = cof / np.sqrt(np.outer(diag, diag))

    return coeffs

def getkNN(value,k ,vector,cached_kNN):
    '''Compute and caches kNNs'''
    if not value in cached_kNN:

        kNN = zip(vector.data.tolist(),vector.nonzero()[1].tolist())

        kNN.sort(reverse=True)

        kNN = [v for v in kNN[1:k+1]]
        cached_kNN[value] = kNN

    else:
        kNN = cached_kNN[value]
    return kNN


def matrix_fact(rating_matrix, U, V, k,alpha, lambda_u , lambda_v):
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
    rating_matrix.data = (rating_matrix.data - 1)/4.0


    dot = U.dot(V)
    dot.data = expit(dot.data)

    original_alpha = alpha
    err_matrix = rating_matrix - dot

    print "Lambda_U = %f"%(lambda_u)
    print "Lambda_V = %f"%(lambda_v)


    E = (1.0/2)*(np.sum(I.multiply(err_matrix).data**2)) + (lambda_u/2.0) *norm(U,"fro")**2 + \
            (lambda_v/2.0)* norm(V,"fro")**2
    original_E = E
    print "Initial error: %.2f"%(E)
    old_E = None
    it = 0
    threshold = 0.001
    last_down = 0
    try:
        while(True):
            alpha = original_alpha
            for i in range(20):
                new_U = U + alpha*((err_matrix*V.T)  - lambda_u*U )

                old_E_aux = E

                aux_dot = new_U.dot(V)
                aux_dot.data = expit(aux_dot.data)

                err_matrix_aux = rating_matrix - aux_dot

                aux_E = (1.0/2)*(np.sum(I.multiply(err_matrix_aux).data**2)) + (lambda_u/2.0) *norm(new_U,"fro")**2 + \
                (lambda_v/2.0)* norm(V,"fro")**2

                if old_E_aux - aux_E  > threshold:
                    U = new_U
                    old_E = E
                    E = aux_E
                    dot = aux_dot
                    err_matrix = err_matrix_aux
                    it+=1
                    if last_down - it > 5:
                        alpha*=10
                    print "Error %.2f for iteration %d"%(E,it)
                elif aux_E > old_E_aux:
                    alpha/=10
                    last_down = it

                    print "Changed Alpha to %.12f"%(alpha)
                    if alpha < 1E-8:
                        break
                else:
                    break
            alpha = original_alpha


            dot = U.dot(V)
            dot.data = expit(dot.data)
            err_matrix = rating_matrix - dot

            E = original_E
            for i in range(20):

                new_V = V - alpha*(U.T*err_matrix + lambda_v*V)


                old_E_aux = E


                aux_dot = U.dot(new_V)
                aux_dot.data = expit(aux_dot.data)

                err_matrix_aux = rating_matrix - aux_dot

                aux_E = (1.0/2)*(np.sum(I.multiply(err_matrix_aux).data**2)) + (lambda_u/2.0) *norm(U,"fro")**2 + \
                (lambda_v/2.0)* norm(new_V,"fro")**2

                if old_E_aux - aux_E  > threshold:
                    V = new_V
                    old_E = E
                    E = aux_E
                    dot = aux_dot
                    err_matrix = err_matrix_aux
                    it+=1
                    if last_down - it > 5:
                        alpha*=10
                    print "Error %.2f for iteration %d"%(E,it)
                elif aux_E > old_E_aux:

                    alpha/=10
                    last_down = it
                    print "Changed Alpha to %.12f"%(alpha)
                    if alpha < 1E-8:
                        break
                else:
                    break
            dot = U.dot(V)
            dot.data = expit(dot.data)
            new_E = (1.0/2)*(np.sum(I.multiply(err_matrix).data**2)) + (lambda_u/2.0) *norm(U,"fro")**2 + \
                (lambda_v/2.0)* norm(V,"fro")**2

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
    dot.data = (dot.data*4.0) + 1
    return dot,it

def get_movie_rating_model_based(proc_matrix,rating_matrix,user,movie,k,method="dot",weighted = False,cached_kNN = {}):
    if movie >= rating_matrix.shape[1]:
        return 3.0

    if method.startswith("cos") and not movie in cached_kNN:
        movie_vector = csr_matrix(cosine_similarity(proc_matrix[movie,:], proc_matrix))
    else:
        movie_vector =  proc_matrix[movie,:]

    kNN = getkNN(movie,k ,movie_vector,cached_kNN)

    total_rating = 0.0
    rating_count = 0.0
    current = 0
    if len(kNN) == 0:
        return 3.0


    if weighted:
        cos_norm = sum([ kNN[x][0] for x in range((k if k < len(kNN) else len(kNN)))])
        if cos_norm == 0:
            return 3.0

    while rating_count < k and current < len(kNN):
        try:
            factor = 1.0
            if weighted:
                factor =  kNN[current][0]/cos_norm


            total_rating += ((rating_matrix[user,kNN[current][1]]) + 3) * factor

            rating_count+=1
        except IndexError:
            if weighted:
                total_rating+= 3.0/k
            else:
                total_rating+=3.0
            rating_count+=1
        current +=1

    if rating_count == 0:
        return 3.0
    if weighted:
        return total_rating
    return float(total_rating/rating_count)


def get_movie_rating_memory_based(proc_matrix,rating_matrix,user,movie,k,method="dot",weighted = False,cached_kNN = {},\
                                  cov = None):

    if method.startswith("cos") and not user in cached_kNN:
        user_vector = cosine_similarity(rating_matrix[user,:], rating_matrix)

    else:
        user_vector = proc_matrix[user,:]

    if cov is not None:
            user_vector= user_vector.multiply(cov[user,:])
    user_vector = csr_matrix(user_vector)

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
            cov_fact = 1.0
            if cov is not None:
                cov_fact = cov[user,kNN[current][1]]
            if weighted:
                factor =  kNN[current][0]/cos_norm
            total_rating += (rating_matrix[kNN[current][1],movie]*cov_fact + 3) * factor

            rating_count+=1
        except IndexError:
            if weighted:
                total_rating+= 3.0/k
            else:
                total_rating+=3.0
            rating_count+=1
        current +=1

    if rating_count == 0:
        return 3.0

    if weighted:
        return total_rating
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

     print "usage is python hw4.py training_file experiment_number test_file k metric is_weighted. No parameters will" \
              "compute the answers for question 1."
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


            rating_matrix_proc = rating_matrix.dot(rating_matrix.T)
            cov = None
            if int(experiment) == 3:
                cov = sparse_corrcoef(rating_matrix)
                cov[np.isnan(cov)] = 0


            for line in lines:

                current = line.split(",")
                user = int(current[0])
                movie = int(current[1])


                results.append(get_movie_rating_memory_based(rating_matrix_proc,rating_matrix,user,movie, \
                                                             k,weighted=w,cached_kNN=cache,cov = cov))


        elif int(experiment) == 2:
            rating_matrix_proc = rating_matrix.T.dot(rating_matrix)


            for line in lines:
                current = line.split(",")
                user = int(current[0])
                movie = int(current[1])

                results.append(get_movie_rating_model_based(rating_matrix_proc,rating_matrix,user,movie, \
                                                            k,weighted=w,cached_kNN=cache))
        elif int(experiment) == 4 or int(experiment) == 5:

            #lists used for cross validation
            alphas = [0.01]
            lambdas = [0.001]
            mv = rating_matrix.data.mean()


            stds = [0.2]
            for std in stds:
                for a in alphas:
                    for l1 in lambdas:
                        U = csr_matrix(np.random.normal(0.0,std,(rating_matrix.shape[0],k)))
                        V = csr_matrix(np.random.normal(0.0,std,(rating_matrix.shape[1],k))).T
                        s1 = time.time()
                        E,it = matrix_fact(rating_matrix.copy(), U.copy(), V.copy(), k,alpha=a,lambda_u=l1,lambda_v=l1)
                        results = []
                        for line in lines:
                            current = line.split(",")
                            user = int(current[0])
                            movie = int(current[1])
                            try:
                                results.append(E[user,movie])
                            except IndexError:
                                results.append(mv)
                        e1 = time.time()
                        with open("timings.txt","a") as timings:
                            timings.write(str(e1-s1) + " " + str(it)+ " " + str(experiment)+"_"+str(a)+"_"+str(l1)+"_" +
                                          str(l1) +"_"+str(k) + "\n")
                        with open("results_"+str(experiment)+"_"+str(k) +"_"+ str(std) + "_"+str(a)+"_"+str(l1)+"_" + str(l1)  + \
                                          ".txt","a") as result_file:
                            for r in results:
                                result_file.write(str(r)+ "\n")


        if int(experiment) != 4 and int(experiment) != 5:
            e = time.time()
            print "Total Time: %d"%(e-s)
            with open("timings.txt","a") as timings:
                timings.write(str(e-s) + " " + str(experiment)+"_"+method+"_"+str(k) +("_weighted\n" if w  else "\n"))
            with open("results_"+str(experiment)+"_"+method+"_"+str(k) +("_weighted" if w  else "")+".txt","a") as result_file:
                for r in results:
                    result_file.write(str(r)+ "\n")

if __name__ == "__main__":
    main()