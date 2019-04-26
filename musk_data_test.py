
# coding: utf-8

# In[20]:

from scipy.io import loadmat
import numpy as np
from sklearn import cross_validation
from sklearn import svm
import numpy as np
import time
import pandas as pd
import os
import datetime


def load_data(data):

    if data == 'musk1_scaled':
        filename_bag = 'data/musk1_scaled/Bag2_mus_escal.mat'
        filename_labels = 'data/musk1_scaled/bagI_mus_escal.mat'
        # print(1)
        X_g = loadmat('data/musk1_scaled/X_mus_escal.mat')
        # print(1)
    elif data == 'data_gauss':
        filename_bag = 'data/gauss_data/bag_g.mat'
        filename_labels = 'data/gauss_data/bagI_gauss.mat'
        X_g = loadmat('data/gauss_data/X_g.mat')
    elif data == 'musk1_original':
        filename_bag = 'data/musk1_unscaled/Bag2_musk_original.mat'
        filename_labels = 'data/musk1_unscaled/bagI_musk1_original.mat'
        X_g = loadmat('data/musk1_unscaled/X_musk_original.mat')
    else:
        file = data
        filename_bag = 'data/' + file + '/Bag2.mat'
        filename_labels = 'data/' + file + '/bagI.mat'
        X_g = loadmat('data/' + file + '/X.mat')
    bag_g = loadmat(filename_bag)
    labels = loadmat(filename_labels)
    try:
        Bag = bag_g['Bag2']
    except KeyError:
        Bag = bag_g['Bag']
    labels = labels['bagI']
    X = X_g['X']
    Bag = np.squeeze(Bag - 1)
    nrobags = max(Bag + 1)
    bags = []
    for i in range(0, nrobags):
        index = np.where(Bag == i)
        bag = X[index]
        bags.append(bag)
    return bags, labels, X


# In[21]:

def MIL2SIL(*args):
    # Parameters
    args = list(args)
    bags = args[0]
    labels = args[1]

    bagT = [np.asmatrix(bag) for bag in bags]
    baglabT = np.asmatrix(labels).reshape((-1, 1))

    X = np.vstack(bagT)
    Y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                   for bag, cls in zip(bagT, baglabT)])
    return X, Y


# In[22]:

'''
class：0 => non-musk，1 => musk
non-musk=negtive musk=positive
labels order: first 1  and then 0
'''


def bags_split(X, Y):
    judge = Y[0, 0]
    index = 0
    for i in range(Y.shape[0]):
        if Y[i, 0] != judge:
            index = i
            break
    splitX = np.split(X, [index], axis=0)
    negtive_instans = splitX[0]
    postive_instans = splitX[1]
    splitY = np.split(Y, [index], axis=0)
    negtive_labels = splitY[0]
    postive_labels = splitY[1]

    return postive_instans, postive_labels, negtive_instans, negtive_labels


def bags_split2(X, Y):
    Y = Y.reshape(-1)
    postive_instans = X[Y == 1]
    negtive_instans = X[Y == 0]
    postive_labels = Y[Y == 1]
    negtive_labels = Y[Y == 0]
    return postive_instans, postive_labels, negtive_instans, negtive_labels


# In[23]:

def pairwise_similarity(negative_samples, negative_labels, positive_samples,
                        positive_labels, C=1):
    '''
    Compute the Similarlity Score of a set of positive instances and negative
    instances.
    Inputs
    --------
    negative_samples : np.array with shape (N_neg, feature_dim)
        All the negative instances used for training
    negative_labels : np.array with shape (N_neg, )
        All the elements in negative_labels should be -1
    positive_samples : np.array with shape (N_pos, feature_dim)
        All the positive instances used for training
    positive_labels : np.array with shape (N_pos, )
        All the elements in positive_labels should be 1
    Return
    --------
    svm_scores : np.array with shape (N_pos, N_pos)
        The output of exemplar SVM scores for each instances which is used to
        compute similarity scores S(i, j) and ranking scores R(i)
    similarity_scores : np.array with shape (N_pos, N_pos)
        The similarity scores is symmetric matrix which is used for compute the
        S(i, j) for CSG's edges
    '''

    num_pos_instance = positive_labels.shape[0]
    svm_scores = []
    order_matrix = np.zeros((num_pos_instance, num_pos_instance))

    for i in range(num_pos_instance):
        data_samples = np.concatenate(
            (positive_samples[i].reshape(1, -1), negative_samples), axis=0)
        data_labels = np.concatenate(
            (positive_labels[i].reshape(1), negative_labels), axis=0)

        clf_linear = svm.LinearSVC(fit_intercept=False, C=C)
        clf_linear.fit(data_samples, data_labels)
#         print(clf_linear.coef_)

        scores_i = clf_linear.decision_function(positive_samples)
        svm_scores.append(scores_i.squeeze())

    svm_scores = np.stack(svm_scores, axis=0)
    index_order = np.argsort(svm_scores * -1, axis=1)

    for i in range(num_pos_instance):
        index = index_order[i]
        order_matrix[i][index] = np.arange(0, num_pos_instance) + 1
    similarity_scores = order_matrix * np.transpose(order_matrix)
    similarity_scores = 1 / similarity_scores
    similarity_scores[svm_scores <= 0] = 0
    similarity_scores[np.transpose(svm_scores) <= 0] = 0

    return (svm_scores, similarity_scores)


def update_consistency_score(edges, gamma, verbose=False):
    connection = edges > 0
    num_instance = edges.shape[0]
    cliques = [[i] for i in range(num_instance)]

    for i in range(num_instance):
        left_vertice = list(set(np.arange(num_instance)) - set(cliques[i]))
        while True:
            prev_num = len(cliques[i])
            idx = np.ix_(cliques[i], left_vertice)
            num_connection = (
                connection[idx].sum(
                    axis=0) > gamma *
                prev_num).sum()  # Equation 4

            # Judge Pi is empty or not
            if num_connection:
                temp = edges[idx].sum(axis=0)
                max_idx = np.argmax(temp)
                cliques[i].append(left_vertice[max_idx])
                left_vertice = list(
                    set(np.arange(num_instance)) - set(cliques[i]))
            else:
                break
    if verbose:
        print(cliques)
    # All of the maximal γ-quasi-clique are established
    maximal_nums = np.array([len(clique) for clique in cliques])
    connection_matrix = np.zeros_like(edges)

    for i in range(num_instance):
        connection_matrix[i][cliques[i]] = 1

    consistency_score = np.zeros_like(edges)
    for i in range(num_instance):
        for j in range(num_instance):
            temp_idx = connection_matrix[:, [i, j]].sum(axis=1) == 2
            consistency_score[i, j] = np.max(
                maximal_nums[temp_idx]) if temp_idx.sum() else 0

    return consistency_score


def calculate_final_score(similarity_scores, gamma=0.9):
    prev_edges = similarity_scores
    while True:
        consistency_score = update_consistency_score(prev_edges, gamma)
#         alpha = np.maximum(10, np.log((consistency_score + 1e-8) / (similarity_scores + 1e-8)))
        alpha = np.power(10, np.floor(
            np.log((consistency_score + 1e-8) / (similarity_scores + 1e-8))))
        curr_edges = consistency_score + similarity_scores * alpha
        if (prev_edges != curr_edges).sum():
            prev_edges = curr_edges
        else:
            break

    return prev_edges / np.linalg.norm(prev_edges)


def ranking_instance(svm_scores, edges, dampening=0.8, num_iter=20):
    diag_score = np.diag(svm_scores)
    rank_score = np.random.randn(*diag_score.shape)
#     print(diag_score)
    for i in range(num_iter):
        rank_score = (1 - dampening) * diag_score + \
            dampening * np.dot(edges, rank_score)
    return rank_score


# In[24]:

def find_positive_instance(
        pos_features,
        neg_features,
        C=1,
        gamma=0.9,
        dampening=0.8,
        num_iter=100):
    positive_labels = np.ones(pos_features.shape[0])
    negative_labels = np.ones(neg_features.shape[0]) * -1

    #print('Computing Pairwise Similarity')
    start_time = time.time()

    svm_scores, similarity_scores = pairwise_similarity(
        neg_features, negative_labels, pos_features, positive_labels, C)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    #print("Pairwise Similarity time (h:m:s): {}".format(elapsed))

    #print('Updating Graph')
    start_time = time.time()

    edges = calculate_final_score(similarity_scores, gamma)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    #print("Updating Graph time (h:m:s): {}".format(elapsed))

    #print('Computing Ranking Scores')
    start_time = time.time()

    ranking_scores = ranking_instance(svm_scores, edges, dampening, num_iter)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    #print("Ranking Scores time (h:m:s): {}".format(elapsed))

    return ranking_scores


# In[25]:

def predict(test_bags, _type, classifier):
    """
    @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
                  object containing m instances with k features
    """
    bag_modified_test = None

    if _type == 'average':
        bag_mean_test = np.asarray([np.mean(bag, axis=0) for bag in test_bags])
        bag_modified_test = bag_mean_test
    elif _type == 'extreme':
        bag_max_test = np.asarray([np.amax(bag, axis=0) for bag in test_bags])
        bag_min_test = np.asarray([np.amin(bag, axis=0) for bag in test_bags])
        bag_extreme_test = np.concatenate((bag_max_test, bag_min_test), axis=1)
        bag_modified_test = bag_extreme_test
    elif _type == 'max':
        bag_max_test = np.asarray([np.amax(bag, axis=0) for bag in test_bags])
        bag_modified_test = bag_max_test
    elif _type == 'min':
        bag_min_test = np.asarray([np.amin(bag, axis=0) for bag in test_bags])
        bag_modified_test = bag_min_test
    else:
        print('No exist')
    predictions = classifier.predict(bag_modified_test)
    return predictions


# In[26]:

def main(data_name, c_value, cliqu):
    bags, labels, X = load_data(data_name)
    seed = 66
    #seed = 70
    # Split Data
    #seed= 90
    train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(
        bags, labels, test_size=0.1, random_state=seed)

    X, Y = MIL2SIL(train_bags, train_labels)
    X, Y = X.getA(), Y.getA()

    postive_instans, postive_labels, negtive_instans, negtive_labels = bags_split2(
        X, Y)

    ranking_scores = find_positive_instance(
        postive_instans, negtive_instans, C=c_value)

    # cliqu=0.9
    find_finalP = postive_instans[(ranking_scores > cliqu)]
    find_finalP_label = postive_labels[(ranking_scores > cliqu)]
    final_train = np.concatenate((find_finalP, negtive_instans), axis=0)
    final_label = np.concatenate((find_finalP_label, negtive_labels), axis=0)
    if np.sum(final_label == 0) < 1:
        #print('parameter fail')
        return 0
    if np.sum(final_label == 1) < 1:
        #print('parameter fail')
        return 0
    clf_linear = svm.LinearSVC(fit_intercept=False, C=c_value)
    clf_linear.fit(final_train, final_label)

    test_bags, test_labels = MIL2SIL(test_bags, test_labels)
    test_labels = test_labels.getA().reshape(-1)

    predict = clf_linear.predict(test_bags)
    acc = (predict == test_labels).sum() / (test_labels.shape[0])
    # print('acc:'+str(acc))
    return acc


# In[ ]:

names = [
    'musk1_scaled',
    'musk2_scaled',
    'tiger_scaled',
    'fox_scaled',
    'elephant_scaled']
c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cliques = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for name in names:
    info = []
    scores = []
    i = 0
    for c_value in c_values:
        for clique in cliques:
            f = open("out1.txt", "a+")
            print(i)
            info.append({'name': name, 'c_value': c_value, 'clique': clique})
            acc = main(name, c_value, clique)
            scores.append(acc)
            print(info[i], file=f)
            print(acc, file=f)
            i += 1
            f.close()
    index = scores.index(max(scores))
    print('best one in' + name)
    print(info[index])
    print(scores[index])


# In[3]:

for i in range(12):
    f = open('out.txt', 'a+')
    print(i, file=f)
    f.close()


# In[ ]:
