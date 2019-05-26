import sys
import pandas as pd
import numpy as np
import surprise as sp
from surprise import Reader, Dataset, SVD, evaluate, accuracy


def recommender(train_name, test_name):
    trainset_df = pd.read_csv(train_name, sep="\t", header=None, names=['user', 'item', 'rating', 'timestamp']) #Create Panda Dataframe from the text files
    testset_df = pd.read_csv(test_name, sep="\t", header=None, names=['user', 'item', 'rating', 'timestamp'])

    reader = Reader(rating_scale=(1, 5)) #make a Reader object to parse each line with the following structure: user ; item ; rating ; [timestamp]. Rating scales is between 1-5
    trainset = Dataset.load_from_df(trainset_df[['user', 'item', 'rating']], reader) #create a Dataset object from the dataframe, and took only the 'user', 'item', and 'rating' column (it must have only these 3 columns)
    trainset = trainset.build_full_trainset() #build

    testset = Dataset.load_from_df(testset_df[['user', 'item', 'rating']], reader)
    #testset = testset.construct_testset(testset.raw_ratings)
    testset = testset.build_full_trainset().build_testset()

    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)

    file_index = train_name[1]
    output_name = "u" + file_index + ".base_prediction.txt"
    with open(output_name, "w") as output:
        for user, item, _, est, _ in predictions:
            output.write("{0}\t{1}\t{2:.3f}\n".format(user,item,est))
    print(accuracy.rmse(predictions))

    #uid = str(1)  # raw user id (as in the ratings file). They are **strings**!
    #iid = str(10)  # raw item id (as in the ratings file). They are **strings**!

    # get a prediction for specific users and items.
    #pred = algo.predict(uid, iid, r_ui=2, verbose=True)

def main():
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    recommender(trainfile, testfile)

main()