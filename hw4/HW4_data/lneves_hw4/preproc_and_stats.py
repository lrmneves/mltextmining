__author__ = 'lrmneves'

from collections import defaultdict

#Variables for the statistics part of the report
user_dict = {}
movie_set = set()

rating_count_dict = defaultdict(int)
imputation_factor = 0.0
total_ratings =0
rating_count = 0

user_4321_ratings = defaultdict(int)
user_4321_movie_count = 0
user_4321_total_rating = 0.0

movie_3_ratings = defaultdict(int)
movie_3_user_count = 0
movie_3_total_rating = 0.0


with open("train.csv") as train_file:
    for line in train_file.readlines():
        current = line.split(",")
        #current[0] = movie, current[1] = user, current[2] = rating
        if current[1] not in user_dict:
            user_dict[current[1]] = []

        user_dict[current[1]].append(current[0]+"#"+str(float(current[2]) - imputation_factor))
        movie_set.add(current[0])


        rating_count_dict[current[2]]+=1

        rating_count+=1
        total_ratings+=float(current[2])
        if current[1] == "4321":
            user_4321_ratings[current[2]]+=1
            user_4321_movie_count+=1
            user_4321_total_rating+=float(current[2])

        if current[0] == "3":
            movie_3_ratings[current[2]]+=1
            movie_3_user_count+=1
            movie_3_total_rating+=float(current[2])

print "-- Global stats --"
print "Total # of Movies: %d"%(len(movie_set))
print "Total # of Users: %d"%(len(user_dict))
print "Total # of times any movie was rated 1: %d"%(rating_count_dict["1"])
print "Total # of times any movie was rated 3: %d"%(rating_count_dict["3"])
print "Total # of times any movie was rated 5: %d"%(rating_count_dict["5"])
print "Average Movie Rate: {0:.2f}".format(total_ratings/rating_count)

print "-- For user 4321 -- "
print "Total # of Movies: %d"%(user_4321_movie_count)
print "Total # of times any movie was rated 1: %d"%(user_4321_ratings["1"])
print "Total # of times any movie was rated 3: %d"%(user_4321_ratings["3"])
print "Total # of times any movie was rated 5: %d"%(user_4321_ratings["5"])
print "Average Movie Rate: {0:.2f}".format(user_4321_total_rating/user_4321_movie_count)

print "-- For Movie 3 -- "
print "Total # of Users: %d"%(movie_3_user_count)
print "Total # of times any movie was rated 1: %d"%(movie_3_ratings["1"])
print "Total # of times any movie was rated 3: %d"%(movie_3_ratings["3"])
print "Total # of times any movie was rated 5: %d"%(movie_3_ratings["5"])
print "Average Movie Rate: {0:.2f}".format(movie_3_total_rating/movie_3_user_count)

#Creates the new preprocessed data file

with open("preprocess_train_matrix_fact","a") as preprocess_file:
    for user in user_dict:
        preprocess_file.write(user + " " +",".join(user_dict[user])+"\n")