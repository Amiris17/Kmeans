#Amiris Olivo
#CAPSTONE MACHINE LEARNING
#KMEANS

import pandas as pd
import math
import numpy as np
import statistics
import random
import matplotlib.pyplot as plt

data = pd.read_csv("breast-cancer-wisconsin.data", header=None)
df = pd.DataFrame(data)

# %%
# Cleaning data
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df[6] = df[6].astype(str).astype('int64')


# %%
# Normalization
def normalize_data(dataset):
    normalized_data = ((dataset - dataset.min()) / (dataset.max() - dataset.min()))
    normalized_data[0] = dataset[0]
    normalized_data[10] = dataset[10]
    return normalized_data


normalized_data = normalize_data(df)
normalized_data.head()
arraydata = np.array(normalized_data)

normalized_list_uncleaned = normalized_data.values.tolist()
random.seed(3)
random.shuffle(normalized_list_uncleaned)
normalized_list = []

for item in normalized_list_uncleaned:
    if (len(item) == 11):
        normalized_list.append(item)


def stratify(normalized_list):
    malignent_df = []
    benign_df = []
    # for idx,row in normalized_data.iterrows():
    #     print(row)
    #     # last_col=int(normalized_data.iloc[i][10].item())
    #     # if(last_col==2):
    #     #     benign_df.append(normalized_data[i])
    #     # else:
    #     #     malignent_df.append(normalized_data[i])
    # benign_df
    for row in normalized_list:
        if (row[10] == 2):
            benign_df.append(row)
        else:
            malignent_df.append(row)
    size_bengin = len(benign_df)
    size_malig = len(malignent_df)

    size_training_benign = round(.9 * size_bengin) #was .7
    size_training_malig = round(.9 * size_malig) # changed to .1 from .7

    size_val_benign = round(.2 * size_bengin)
    size_val_malig = round(.2 * size_malig)

    size_test_benign = round(.1 * size_bengin)
    size_test_malig = round(.1 * size_malig)

    train = []
    val = []
    test = []

    for i in range(size_training_benign):
        train.append(benign_df[i])
    for i in range(size_training_malig):
        train.append(malignent_df[i])



    for i in range(size_test_benign):
        test.append(benign_df[i])
    for i in range(size_test_malig):
        test.append(malignent_df[i])

    return train, test


#train, val, test = stratify(normalized_list)


def n_fold_cross_validation(data,folds):
    fold_size=int(len(data)/folds)
    data_split=[]
    for i in range(folds):
        one_fold=[]
        one_fold=(random.sample(data,fold_size))
        data_split.append(one_fold)
    return data_split

    #z=len(Fold_list)
    #z=random.randrange(len(fol))









def euclidean_distance(num1, num2):
    distance = 0
    # return np.sqrt(np.sum((num1-num2)**2))
    for i in range(len(num1)):
        try:
            distance += (num1[i] - num2[i]) ** 2
        except print(num1, num2):
            pass

    # print(distance)
    return np.sqrt(distance)


# %%
def empty_clusters(k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    return clusters


def k_means(k, train):
    random.seed(3)
    centroids = random.sample(train, k)
    c1 = []
    for cen in centroids:
        c1.append(cen[1:9])
    new_centroids = []
    clusters = empty_clusters(k)
    counter = 0
    # print("This is c1: ",c1,"This is centroids: "  ,centroids)
    while (counter < 10000):
        new_centroids = []
        # print(c1[0])
        clusters = empty_clusters(k)
        for row in train:
            distances = []
            for centroid in c1:
                distances.append(euclidean_distance(row[1:9], centroid))
            if (len(row) == 11):
                clusters[distances.index(min(distances))].append(row)
            else:
                print(row)
        for i in range(k):
            if (len(clusters[i]) != 0):
                temp = (np.mean(clusters[i], axis=0)).tolist()[1:9]
                # print(temp)
                new_centroids.append(temp)
            else:
                new_centroids.append(c1[i])
        counter += 1
        if (c1 == new_centroids):
            # print(c1[0],new_centroids[0])
            # print("New Centroids: ", new_centroids)
            break
        c1 = new_centroids
    # print(counter)
    return clusters, new_centroids


def optimal_k(train,val):
    accuracy_list = []
    for k in range(1, len(train)):
        clust, cent = (k_means(k, train))
        majority_cluster = {}
        majority_class = []
        for i in range(len(cent)):
            majority_list = []
            for row in clust[i]:
                majority_list.append(row[10])
            if (len(clust[i]) != 0):
                majority = statistics.mode(majority_list)
            else:
                majority = 4
            majority_class.append(majority)
        predicted = []
        actual = []
        for row in val:
            distances = []
            for c in cent:
                distances.append(euclidean_distance(row[1:9], c))
            predicted.append(majority_class[distances.index(min(distances))])
            actual.append(row[10])
        accuracy = 0
        for i in range(len(predicted)):
            if (predicted[i] == actual[i]):
                accuracy += 1

        accuracy = accuracy / len(predicted)
        print(accuracy,k)
        #print('ACCURACY ON TRAINING:',accuracy,k)

        accuracy_list.append(accuracy)
    #print(np.mean(accuracy_list))

    #print(max(accuracy_list), accuracy_list.index(max(accuracy_list)) + 1)
    return max(accuracy_list), accuracy_list.index(max(accuracy_list))+1
#variable=(optimal_k(train))

#fold_call=(optimal_k(Fold_res))
#fold2_call=(optimal_k)(Ten)
#print(fold_call)

def test_set(train,test,val):
    accuracy_list = []
    # for k in range(1,len(train)):
    f=optimal_k(train,val)
    majority_cluster = {}
    majority_class = []
    clust,cnt=k_means(f,train)
    for i in range(len(cent)):
        majority_list = []
        for row in clust[i]:
            majority_list.append(row[10])
        if (len(clust[i]) != 0):
            majority = statistics.mode(majority_list)
        else:
            majority = 4
        majority_class.append(majority)
    predicted = []
    actual = []
    for row in test:
        distances = []
        for c in cent:
            distances.append(euclidean_distance(row[1:9], c))
        predicted.append(majority_class[distances.index(min(distances))])
        actual.append(row[10])
    accuracy = 0
    for i in range(len(predicted)):
        if (predicted[i] == actual[i]):
            accuracy += 1
    accuracy = accuracy / len(predicted)
    # print(accuracy,k)
    accuracy_list.append(accuracy)
    #print(max(accuracy_list))
    print(accuracy_list.index(max(accuracy_list)))
    return max(accuracy_list)#accuracy_list.index(max(accuracy_list))
train,test=stratify(normalized_list)
f=n_fold_cross_validation(train,10) #10 sets
#print(len(f))
list_=[]
#print(len(f))
for i in f:

    val=i
    train=f
    train.remove(i)
    train=sum(train,[])
    print(test_set(train,test,val))
    list_.append(test_set(train,test,val))

accuracies=np.mean(list_)

print(accuracies)   #average accuracies
