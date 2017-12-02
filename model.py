#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 10:23 PM
# @Author: Shen Sijie
# @File: model.py
# @Project: ZHWordSegmentation

import gzip, pickle
import random
import copy


class Perceptron(object):
    """
    A perceptron model used for Chinese word segmentation
    """

    def __init__(self, dimension):
        """
        Initialize the model
        :param dimension: dimension of vocabulary
        """
        self.dimension = dimension
        self.theta = [random.randint(-5, 5) for i in range(dimension)]
        self.theta[0] = 0
        self.theta_sum = [0] * dimension
        self.last_update = [0] * dimension
        self.total_step = 0

    def get_score(self, features):
        """
        Get scores of input features
        :param features: features of a character, should be a tuple contains indexes of features in vocabulary
        :return: score of *features*
        """
        return sum([self.theta[i] for i in features])

    def update(self, feature_list, label):
        """
        Update arguments of model
        :param feature_list: features of a character, should be a tuple like (features_0_labeled, features_1_labeled)
        :param label: correct label of this character
        :return: None
        """
        # Compute scores
        score_true = self.get_score(feature_list[label])
        score_false = self.get_score(feature_list[1 - label])

        # Update theta
        self.total_step += 1
        if score_false >= score_true:
            for i in feature_list[label]:
                self.theta_sum[i] += self.theta[i] * (self.total_step - self.last_update[i]) + 1
                self.theta[i] += 1
                self.last_update[i] = self.total_step
            for i in feature_list[1 - label]:
                self.theta_sum[i] += self.theta[i] * (self.total_step - self.last_update[i]) - 1
                self.theta[i] -= 1
                self.last_update[i] = self.total_step

    def predict(self, feature_list):
        """
        Get prediction of input features
        :param feature_list: features of a character, should be a tuple like (features_0_labeled, features_1_labeled)
        :return: prediction of *feature_list*
        """
        # Compute scores
        score0 = self.get_score(feature_list[0])
        score1 = self.get_score(feature_list[1])

        if score0 > score1:
            return 0
        else:
            return 1

    def save(self, path, average=True):
        """
        Save the model arguments
        :param path: save path
        :param average: use average perceptron
        :return: None
        """
        if average:
            # Make average
            for i in range(self.dimension):
                if self.last_update[i] != self.total_step:
                    self.theta_sum[i] += self.theta[i] * (self.total_step - self.last_update[i])
                self.theta[i] = self.theta_sum[i] / self.total_step

        file = gzip.open(path, 'wb')
        pickle.dump(self.theta, file)
        file.close()

        print('Model saved at path', path)

    def load(self, path):
        """
        Load arguments from a saved model
        :param path: save path
        :return: None
        """
        file = gzip.open(path, 'rb')
        self.theta = pickle.load(file)
        file.close()

        print('Model loaded from saved model', path)
