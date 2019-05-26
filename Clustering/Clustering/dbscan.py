import sys
import collections
import operator #itemgetter
import scipy.spatial as spatial
import numpy as np
import matplotlib.pyplot as plt

UNCHECKED = -1 #a constant to represent unchecked point
OUTLIER = -2 #a constant to represent the outlier point

# load the file, and convert to NUMPY array
def loadFile(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            coord = [float(i) for i in line.split()[1:3]] #only take the coordinates, because the object id is already sorted, meaning it's unnecessary to save
            dataset.append((coord))
    dataset = np.array(dataset) #create a numpy array from it
    return dataset

#main function
def find_cluster(maxcluster,eps,minpts,dataset,classifications,i,clusterid,neighbor_tree):
    seeds = neighbor_tree.query_ball_point(dataset[i],eps) #return the INDEX of all point within eps of dataset[i]
    seeds.remove(i) #remove item with value 'i' because query_ball_point finds all point that is within eps of dataset[i], including dataset[i] itself
    if len(seeds) < minpts:
        classifications[i] = OUTLIER #set as outlier if it didn't found enough neighbor points
        return False
    else:
        classifications[i] = clusterid #set current seed clusterid
        for seed_point in seeds:
            classifications[seed_point] = clusterid #set all seed points as current clusterid

        for seed_point in seeds:
            neighbor = neighbor_tree.query_ball_point(dataset[seed_point],eps)
            neighbor.remove(seed_point) #same reason as above
            if len(neighbor) >= minpts:
                for neighbor_point in neighbor:
                    if classifications[neighbor_point] == UNCHECKED:
                        classifications[neighbor_point] = clusterid #set neighbor point with the same cluster id
                        seeds.append(neighbor_point) #add this to the seed list to be checked
                    if classifications[neighbor_point] == OUTLIER:
                        #set this point as the border point, thus not adding it to the seed list because it is already known that there isn't enough neighbor points
                        classifications[neighbor_point] = clusterid 
        return True                    


def dbscans(maxcluster,eps,minpts,dataset,neighbor_tree):
    clusterid = 0 #clusterid also represent number of clusters found
    n_data = dataset.shape[0]
    classifications = [UNCHECKED] * n_data #classifications = array to represent the clusterid of each data
    for i in range(n_data):
        if classifications[i] == UNCHECKED: #only check for new cluster if a point is UNCHECKED
            if find_cluster(maxcluster,eps,minpts,dataset,classifications,i,clusterid,neighbor_tree): #if a new cluster is found
                clusterid = clusterid+1 #increase clusterid if a new cluster was found

    #cluster trimming if m>n
    if clusterid>maxcluster:
        cluster_counter = collections.Counter(classifications) #a dictionary to save the number of occurence of each key (point)
        sorted_counter = sorted(cluster_counter.items(), key=operator.itemgetter(1)) #sort by ascending value (count)
        diff = clusterid-maxcluster #clusterid is the number of clusters

        index = 0 #get the lowest value (count) first
        while(index<diff): #take 'diff'-much cluster
            key, value = sorted_counter[index]
            index = index+1 #increase the index

            if key==OUTLIER: #if an OUTLIER is on one of the lowest, then we ignore it, and check the next lowest count cluster. Don't forget to increase the diff
                diff=diff+1
                continue
            classifications[:] = [OUTLIER if id==key else id for id in classifications] #otherwise, set the classifications to OUTLIER (basically, we remove the cluster)
            clusterid = clusterid - 1 #reduce the number of cluster
        
        #fix the cluster id number, since some clusters are removed
        cluster_counter = collections.Counter(classifications)
        index = 0
        for key in sorted(cluster_counter.keys()): #sort by key
            if key == OUTLIER: #skip outlier
                continue
            classifications[:] = [index if id==key else id for id in classifications] #set all item with the same key on the 'classifications' array with the new id ('index')
            index = index+1
    return clusterid, classifications

#show cluster information
def showClusterInfo(n_clusters, classifications):  
    print('# of clusters= %d\n' %(n_clusters))
    cluster_counter = collections.Counter(classifications)
    total = 0
    count = 0
    outlier = -1
    for key in sorted(cluster_counter.keys()):
        value = cluster_counter[key]
        if key==OUTLIER:
            outlier = value
            total = total+value
        elif key==UNCHECKED: #this count should be 0 anyway
            continue
        else:
            print('Cluster %d : %d' % (key, value))
            count = count+1
            total = total+value
    print('Outliers : %d\n' % (outlier))
    print('\nTotal Data= ',total)

#print cluster information (object id) to file
def printClusterInfo(n_clusters, classifications,filename):
    cluster_counter = collections.Counter(classifications)
    for key in sorted(cluster_counter.keys()):
        if key==OUTLIER:
            continue
        else:
            outputname = filename.split('.')[0] + "_cluster_" + str(key) + ".txt"
            print(outputname)
            file = open(outputname,"w+")
            for i in range(len(classifications)):
                if classifications[i]==key:
                    file.write("%d\n"%(i))
            file.close()

#For ‘input1.txt’, n=8, Eps=15, MinPts=22
#For ‘input2.txt’, n=5, Eps=2, MinPts=7
#For ‘input3.txt’, n=4, Eps=5, MinPts=5
def main():
    print('==DBSCANS==')
    filename = sys.argv[1]
    maxcluster = int(sys.argv[2])
    eps = float(sys.argv[3])
    minpts = int(sys.argv[4])
    dataset = loadFile(filename)
    neighbor_tree = spatial.cKDTree(dataset) #KDTree to find quickly find nearest neighbor within specific range

    print('==start DBSCANS==')
    n_clusters, classifications = dbscans(maxcluster,eps,minpts,dataset,neighbor_tree) #main function
    showClusterInfo(n_clusters,classifications) #show general cluster information on console
    printClusterInfo(n_clusters,classifications,filename) #print cluster information to output file
    
main()