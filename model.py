#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 10:23 PM
# @Author: Shen Sijie
# @File: model.py
# @Project: ZHWordSegmentation

import gzip
import pickle


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
        self.theta = [0] * dimension
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


class StructuredPerceptron(object):
    """
    A structured perceptron model used for Chinese word segmentation
    """

    def __init__(self, dimension):
        """
        Initialize the model
        :param dimension: dimension of vocabulary
        """
        self.dimension = dimension
        self.theta = [0] * dimension
        self.theta_sum = [0] * dimension

        self.transitions = [[0, 0], [0, 0]]
        self.transitions_sum = [[0, 0], [0, 0]]

        self.last_update = [0] * dimension
        self.transitions_last_update = [[0, 0], [0, 0]]
        self.total_step = 0

    def get_score(self, features):
        """
        Get scores of input features
        :param features: features of a character, should be a tuple contains indexes of features in vocabulary
        :return: score of *features*
        """
        return sum([self.theta[i] for i in features])

    def predict(self, sentence_features):
        """
        Get prediction of input sentence features, using Viterbi Algorithm
        :param sentence_features: features of input sentence
        :return: a list of 0/1 predictions
        """
        # Get emissions
        emissions = list()
        for feature_list in sentence_features:
            emissions.append([self.get_score(feature_list[0]), self.get_score(feature_list[1])])

        # Viterbi forward
        alphas = list()
        pointers = list()
        alphas.append([emissions[0][0], emissions[0][1]])
        pointers.append([-1, -1])
        for i in range(1, len(sentence_features)):
            score00 = alphas[i - 1][0] + self.transitions[0][0] + emissions[i][0]
            score10 = alphas[i - 1][1] + self.transitions[1][0] + emissions[i][0]

            score01 = alphas[i - 1][0] + self.transitions[0][1] + emissions[i][1]
            score11 = alphas[i - 1][1] + self.transitions[1][1] + emissions[i][1]

            alphas.append([max([score00, score10]), max([score01, score11])])
            pointers.append([0 if score00 > score10 else 1, 0 if score01 > score11 else 1])

        # Viterbi backward
        tags = [0 if alphas[-1][0] > alphas[-1][1] else 1]
        for i in range(len(sentence_features) - 1, 0, -1):
            tags.append(pointers[i][tags[-1]])
        tags.reverse()

        return tags

    def update(self, sentence_features, sentence_labels):
        """
        Update arguments of model
        :param sentence_features: features of input sentence
        :param sentence_labels: labels of input sentence
        :return: None
        """
        pred = self.predict(sentence_features)

        self.total_step += 1
        if pred == sentence_labels:
            return

        if len(pred) != len(sentence_labels):
            print('Vector dimension not compatible')

        # Update arguments
        for i in range(len(sentence_labels)):
            if pred[i] != sentence_labels[i]:
                right = sentence_labels[i]
                wrong = pred[i]

                # Update arguments of right features
                for feature in sentence_features[i][right]:
                    self.theta_sum[feature] += self.theta[feature] * (self.total_step - self.last_update[feature]) + 1
                    self.theta[feature] += 1
                    self.last_update[feature] = self.total_step

                # Update arguments of right transitions
                prev = pred[i - 1] if i > 0 else 1
                self.transitions_sum[prev][right] += self.transitions[prev][right] * (
                        self.total_step - self.transitions_last_update[prev][right]) + 1
                self.transitions[prev][right] += 1
                self.transitions_last_update[prev][right] = self.total_step

                # Update arguments of wrong features
                for feature in sentence_features[i][wrong]:
                    self.theta_sum[feature] += self.theta[feature] * (self.total_step - self.last_update[feature]) - 1
                    self.theta[feature] -= 1
                    self.last_update[feature] = self.total_step

                # Update arguments of wrong transitions
                prev = pred[i - 1] if i > 0 else 1
                self.transitions_sum[prev][wrong] += self.transitions[prev][wrong] * (
                        self.total_step - self.transitions_last_update[prev][wrong]) - 1
                self.transitions[prev][wrong] -= 1
                self.transitions_last_update[prev][wrong] = self.total_step

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
            for i in range(2):
                for j in range(2):
                    self.transitions_sum[i][j] += self.transitions[i][j] * (
                            self.total_step - self.transitions_last_update[i][j])
                self.transitions[i][j] = self.transitions_sum[i][j] / self.total_step

        file = gzip.open(path, 'wb')
        pickle.dump((self.theta, self.transitions), file)
        file.close()

        print('Model saved at path', path)

    def load(self, path):
        """
        Load arguments from a saved model
        :param path: save path
        :return: None
        """
        file = gzip.open(path, 'rb')
        self.theta, self.transitions = pickle.load(file)
        file.close()

        print('Model loaded from saved model', path)
