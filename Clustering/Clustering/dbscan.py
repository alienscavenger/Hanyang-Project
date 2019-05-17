import sys
import collections
import operator #itemgetter
import scipy.spatial as spatial
import numpy as np
import matplotlib.pyplot as plt

UNCHECKED = -1
OUTLIER = -2

# load the file, and convert to numpy array
def loadFile(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            coord = [float(i) for i in line.split()[1:3]] #only take the coordinates, because the object id is already sorted
            dataset.append((coord))
    dataset = np.array(dataset)
    return dataset

#main function
def find_cluster(maxcluster,eps,minpts,dataset,classifications,i,clusterid,neighbor_tree):
    seeds = neighbor_tree.query_ball_point(dataset[i],eps) #return the index of all point within eps of dataset[i]
    seeds.remove(i) #remove 'i' because query_ball_point finds all point that is within eps of dataset[i], including dataset[i] itself
    if len(seeds) < minpts:
        classifications[i] = OUTLIER #set as outlier if didn't foud enough neighbor point
        return False
    else:
        classifications[i] = clusterid #set current seed clusterid

        for seed_point in seeds:
            classifications[seed_point] = clusterid
            neighbor = neighbor_tree.query_ball_point(dataset[seed_point],eps)
            neighbor.remove(seed_point) #same reason as above
            if len(neighbor) >= minpts:
                for neighbor_point in neighbor:
                    if classifications[neighbor_point] == UNCHECKED:
                        classifications[neighbor_point] = clusterid #set neighbor point with the same cluster id
                        seeds.append(neighbor_point) #add this to the seed list (don't change order with the above line)
                    if classifications[neighbor_point] == OUTLIER:
                        classifications[neighbor_point] = clusterid #set this point as the border
        return True                    


def dbscans(maxcluster,eps,minpts,dataset,neighbor_tree):
    clusterid = 0
    n_data = dataset.shape[0]
    classifications = [UNCHECKED] * n_data
    for i in range(n_data):
        if classifications[i] == UNCHECKED:
            if find_cluster(maxcluster,eps,minpts,dataset,classifications,i,clusterid,neighbor_tree):
                clusterid = clusterid+1

    #cluster trimming if m>n
    if clusterid>maxcluster:
        cluster_counter = collections.Counter(classifications)
        sorted_counter = sorted(cluster_counter.items(), key=operator.itemgetter(1)) #sort by value ascending
        diff = clusterid-maxcluster #here, clusterid = number of clusters

        index = 0 #get the first lowest value
        while(index<diff): #take 'diff'-much cluster
            key, value = sorted_counter[index]
            index = index+1 #increase the index
            if key==OUTLIER: #if OUTLIER is on one of the lowest, don't forget to increase the diff
                diff=diff+1
                continue
            classifications[:] = [OUTLIER if id==key else id for id in classifications] #set the classifications to OUTLIER (basically remove the cluster)
            clusterid = clusterid - 1
        
        #fix the cluster id number, since some clusters are removed
        cluster_counter = collections.Counter(classifications)
        index = 0
        for key in sorted(cluster_counter.keys()): #sort by key value
            if key == OUTLIER: #skip outlier
                continue
            classifications[:] = [index if id==key else id for id in classifications]
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
    showClusterInfo(n_clusters,classifications)
    printClusterInfo(n_clusters,classifications,filename)
    
if __name__=='__main__':
    main()