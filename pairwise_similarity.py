#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import svm
import numpy as np


def pairwise_similarity(negative_samples, negative_labels, positive_samples,
                        positive_labels):
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

        clf_linear = svm.LinearSVC(fit_intercept=False)
        clf_linear.fit(data_samples, data_labels)

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

if __name__ == '__main__':
    negtive_samples = np.random.randn(100, 4096)
    positive_samples = np.random.randn(5, 4096)
    negtive_labels = np.ones((100, )) * -1
    positive_labels = np.ones((5, ))

    (svm_scores, similarity_scores) = pairwise_similarity(
        negtive_samples, negtive_labels, positive_samples, positive_labels)

    print(svm_scores)
    print(similarity_scores)
