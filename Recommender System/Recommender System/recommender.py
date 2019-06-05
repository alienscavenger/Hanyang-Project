import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import surprise.prediction_algorithms as spa
from surprise import Reader, Dataset, accuracy

START_TIME = time.time()
PREV_TIME = START_TIME
DEBUGFILE = False
VERBOSE = True
GRAPH = False
ZERO_INJECTION = False #tt is NOT used to predict absolute rating, but rather to find top-N item. So if you use this, accuracy will be LOWER
ZERO_INJECTION_THETA = 25 #set THETA percentage of the worst imputed data as 0 on the original dataset (uninteresting item)

def verbosePrint(text,printTime=True):
    if VERBOSE:
        if printTime:
            checkTime()
        print(text)

def checkTime(): #used on verbose mode
    global PREV_TIME
    curr = time.time()
    print("\n   (+%s)\n --- %s seconds ---" % (
        round(curr - PREV_TIME,2),
        round(curr - START_TIME,2)))
    PREV_TIME = curr
#end of checkTime()

def zeroInjection(trainset_df, names, train_name): #parameter is not passed by reference
    #pre-use preferrence matrix
    preuse_pref = trainset_df.copy()
    #preuse_pref.loc[preuse_pref['rating']>0, 'rating'] = 1 #set every nonzero value with 1. This line is replaced by 'aggfunc' on the pivot_table
    #reshape
    verbosePrint('Reshaping Pre-use matrix and imputing missing value with zeros')
    preuse_pref = preuse_pref.pivot_table(index = 'user', columns ='item', values = 'rating', aggfunc ='count').fillna(0)
    
    maxi_users = trainset_df.user.unique().max()
    maxi_items = trainset_df.item.unique().max()

    #add missing item to the dataset (not rated by any user)
    verbosePrint('Adding mising item to dataset')

    it = iter(preuse_pref.columns[:].values)
    curr = next(it)
    lis = []
    for i in range(1,maxi_items+1):
        if i!=curr:
            #print(i) #print the item which have no review
            preuse_pref.insert(i-1,i,float(0)) #insert at column i (0-based) with the name 'i', with default value for that column = 0
            lis.append(i)
        elif i!=maxi_items:
            curr = next(it)

    preuse_pref.reset_index(inplace = True) #move column name (item) to the first column (where the value would actually be the index), and the row name (user) to second column
    
    #unpivot
    verbosePrint('Reshaping back to previous shape, and sorting the column value')
    preuse_pref = preuse_pref.melt(id_vars=['user'],value_vars=preuse_pref.columns[1:].values,value_name='rating',var_name='item')
    preuse_pref = preuse_pref.sort_values(['user','item']).reset_index(drop = True) #sort by user and item value, then reset the index, and drop old index

    #make a Reader object to parse each line with the following structure: user ; item ; rating ; [timestamp]. Rating scales is between 1-5
    reader = Reader(rating_scale=(0, 1))
    
    #create a Dataset object from the dataframe, and took only the 'user', 'item', and 'rating' column (it must have only these 3 columns)
    #print(preuse_pref[preuse_pref['rating']<1][['user', 'item', 'rating']]) #row which have rating as 0
    verbosePrint('Creating Dataset trainset and testset from the Pre-use matrix')
    preuse_pref_trainset = Dataset.load_from_df(preuse_pref[['user', 'item', 'rating']], reader)
    preuse_pref_testset = Dataset.load_from_df(preuse_pref[preuse_pref['rating']<1][['user', 'item', 'rating']], reader)

    verbosePrint("Building full trainset and testset")
    preuse_pref_trainset = preuse_pref_trainset.build_full_trainset()
    preuse_pref_testset = preuse_pref_testset.build_full_trainset().build_testset()

    verbosePrint('Training the model ({:,} data, this may take a while)'.format(maxi_users*maxi_items))
    algo = SVD()
    algo.fit(preuse_pref_trainset)

    verbosePrint('Evaluating on the testset')
    predictions = algo.test(preuse_pref_testset) #Dataset (list of namedTuple)
    
    verbosePrint('(Debug) Outputting prediction (unrated item) to file')
    file_index = train_name[1]
    output_name = "u" + file_index + ".unrated.txt"
    with open(output_name, "w") as output:
        for user, item, _, est, _ in predictions:
            output.write("{0}\t{1}\t{2:.3f}\n".format(user,item,est))
    
    verbosePrint('Getting non-interesting item according to theta(%s%%)'%(str(ZERO_INJECTION_THETA)))
    predictions = [[user,item,est] for user, item, _, est, _ in predictions]
    predictions = pd.DataFrame(predictions, columns=names[:-1]).sort_values(['rating']).reset_index(drop = True) #Ascending (Now it is a Dataframe)

    cut_index = int(round((((ZERO_INJECTION_THETA)/float(100))*predictions.shape[0])+float(0.5))) #round up (1-based index)
    predictions = predictions[:cut_index]
    
    verbosePrint('(Debug) Outputting uninteresting item to file')
    file_index = train_name[1]
    output_name = "u" + file_index + ".uninteresting_item.txt"
    with open(output_name, "w") as output:
        for index, row in predictions.iterrows():
            output.write("{0}\t{1}\t{2:.3f}\n".format(int(row['user']),int(row['item']),row['rating']))

    predictions['rating'] = int(0) #set all current rating to 0

    verbosePrint('Appending non-interesting item to dataset')
    trainset_df = trainset_df.append(predictions, ignore_index=True, sort=False) #append predictions to old trainset, ignore the index on predictions, and don't sort the column after appending
    trainset_df = trainset_df.sort_values(['user','item']).reset_index(drop = True)
    
    verbosePrint('Finished zero-injecting dataset.')

    return trainset_df
#end of zeroInjection()

def recommender(train_name, test_name):
    names = ['user', 'item', 'rating', 'timestamp'] #name of the columns

    verbosePrint('Creating Panda Dataframe from file')
    trainset_df = pd.read_csv(train_name, sep="\t", header=None, names=names) #Create Panda Dataframe from the text files
    testset_df = pd.read_csv(test_name, sep="\t", header=None, names=names)
    
    n_data= trainset_df.user.shape[0]
    n_users = trainset_df.user.unique().shape[0]
    n_items = trainset_df.item.unique().shape[0]

    verbosePrint('\nNumber of data {}\nNumber of user: {}\nNumber of item: {}'.format(
        n_data,n_users,n_items),False)

    ##############################################################
    if ZERO_INJECTION:
        trainset_df = zeroInjection(trainset_df, names, train_name)
    ##############################################################
    
    if GRAPH:
        #plot graph 1
        plt.subplot(1,3,1)
        hist_rating_trainset = trainset_df['rating'].value_counts(sort = False).plot.bar(rot=0, title="Trainset rating count")
        hist_rating_trainset.set_xlabel("Rating")
        hist_rating_trainset.set_ylabel("Count")

        #plot graph 2
        plt.subplot(1,3,2)
        hist_rating_testset = testset_df['rating'].value_counts(sort = False).plot.bar(rot=0, title="Testset rating count",)
        hist_rating_testset.set_xlabel("Rating")
        hist_rating_testset.set_ylabel("Count")

    verbosePrint('Reading and building dataset')
    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(trainset_df[['user', 'item', 'rating']], reader)
    trainset = trainset.build_full_trainset() #build trainset (required by the Surprise library)

    testset = Dataset.load_from_df(testset_df[['user', 'item', 'rating']], reader)
    #testset = testset.construct_testset(testset.raw_ratings)
    testset = testset.build_full_trainset().build_testset()
    
    verbosePrint('Learning from trainset')

    algo = spa.KNNBaseline() #way faster, but more inaccurate
    algo = spa.SVD(n_epochs=400,lr_all=0.001,reg_all=0.1, verbose=False)
    algo.fit(trainset)

    verbosePrint('Evaluating testset')

    predictions = algo.test(testset) #predictions = list of namedtuple
    
    predictions_df = pd.DataFrame.from_records(
        predictions[:],
        columns=predictions[0]._fields)
    predictions_df = predictions_df.drop(['r_ui', 'details'], axis=1)
    predictions_df.columns = names[:-1]
    
    if GRAPH:
        plt.subplot(1,3,3)
        hist_rating_predictions = predictions_df['rating'].plot.hist(bins=50,rot=0, title="Predictions rating count",)
        hist_rating_predictions.set_xlabel("Rating")
        hist_rating_predictions.set_ylabel("Count")

    verbosePrint('Outputting prediction to file')

    file_index = train_name[1]
    output_name = "u" + file_index + ".base_prediction.txt"
    with open(output_name, "w") as output:
        for user, item, _, est, _ in predictions:
            output.write("{0}\t{1}\t{2:.3f}\n".format(user,item,est))

    verbosePrint('==Finished==')
    print(accuracy.rmse(predictions))

    if GRAPH:
        plt.show()

    '''
    # to predict one item
    uid = str(1)  # raw user id (as in the ratings file). They are **strings**!
    iid = str(1)  # raw item id (as in the ratings file). They are **strings**!

    # get a prediction for specific users and items.
    pred = algo.predict(uid, iid, r_ui=2, verbose=True)
    print(pred)
    
    predictions = algo.test(preuse_pref_testset)
    print(predictions)
    '''

def main():
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    if DEBUGFILE:
        trainfile = "u1_debug.base"
    print('Recommender System - (%s)'%(trainfile))
    
    global ZERO_INJECTION_THETA
    global ZERO_INJECTION
    recommender(trainfile,testfile) #start recommmendation process

main()