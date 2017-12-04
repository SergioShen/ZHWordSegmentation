#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 8:32 PM
# @Author: Shen Sijie
# @File: dataset.py
# @Project: ZHWordSegmentation

from constant import *
from vocab import Vocab


class Dataset(object):
    """
    A dataset with formatted contents, used for train and test
    """

    def __init__(self, name):
        """
        Initialize the dataset: generate/load vocabulary, generate features
        :param name: name of dataset, should be 'train' or 'test' or 'keyboard'
        """
        # Open data file
        if name == 'train':
            self.data_file = open(TRAIN_DATA, 'r')
        elif name == 'test':
            self.data_file = open(TEST_DATA, 'r')

        # Generate vocabulary, features, labels and words
        self.vocab = Vocab([UNKNOWN])

        # Load vocabulary if vocab exists
        self.vocab_loaded = False
        if os.path.exists(VOCAB_PATH):
            self.vocab_loaded = True
            vocab_file = open(VOCAB_PATH, 'r')
            for line in vocab_file:
                self.vocab.add(line.strip())
            vocab_file.close()

        # Return if keyboard test
        if name == 'keyboard':
            return

        self.features = list()
        self.labels = list()
        self.words = list()

        for line in self.data_file:
            sentence_features = list()  # features of this sentence
            sentence_label = list()  # labels of this sentence

            self.words.append(line.split())  # separated words of this sentence
            text = '^' + line.strip() + '$'  # add begin and end mark
            length = len(text)
            for i in range(length):
                # Ignore empty chars and specific chars
                if text[i] == ' ' or text[i] == '^' or text[i] == '$':
                    continue

                # Get label of this character
                if text[i + 1] in SPACE or text[i + 1] == '$':
                    label = 1
                else:
                    label = 0

                # Get prev and next character
                prev = i - 1
                next = i + 1
                while text[prev] in SPACE:
                    prev -= 1
                while text[next] in SPACE:
                    next += 1

                # Add label
                sentence_label.append(label)

                # Add feature
                if not self.vocab_loaded:
                    # Vocab is not loaded, we need to add features into vocabulary and get indexes
                    sentence_features.append(((
                                                  # Uni-gram, with label 0
                                                  self.vocab.add('_'.join((str(1), text[prev], str(0)))),
                                                  self.vocab.add(('_'.join((str(2), text[i], str(0))))),
                                                  self.vocab.add(('_'.join((str(3), text[next], str(0))))),

                                                  # Bi-gram, with label 0
                                                  self.vocab.add('_'.join((str(4), text[prev], text[i], str(0)))),
                                                  self.vocab.add('_'.join((str(5), text[i], text[next], str(0)))),
                                                  self.vocab.add('_'.join((str(6), text[prev], text[next], str(0)))),

                                                  # Tri-gram, with label 0
                                                  self.vocab.add(
                                                      '_'.join((str(7), text[prev], text[i], text[next], str(0))))
                                              ), (
                                                  # Uni-gram, with label 1
                                                  self.vocab.add('_'.join((str(1), text[prev], str(1)))),
                                                  self.vocab.add(('_'.join((str(2), text[i], str(1))))),
                                                  self.vocab.add(('_'.join((str(3), text[next], str(1))))),

                                                  # Bi-gram, with label 1
                                                  self.vocab.add('_'.join((str(4), text[prev], text[i], str(1)))),
                                                  self.vocab.add('_'.join((str(5), text[i], text[next], str(1)))),
                                                  self.vocab.add('_'.join((str(6), text[prev], text[next], str(1)))),

                                                  # Tri-gram, with label 1
                                                  self.vocab.add(
                                                      '_'.join((str(7), text[prev], text[i], text[next], str(1))))
                                              )))
                else:
                    # Vocab loaded, we only need to get indexes from vocabulary
                    sentence_features.append(((
                                                  # Uni-gram, with label 0
                                                  self.vocab.get_index('_'.join((str(1), text[prev], str(0)))),
                                                  self.vocab.get_index(('_'.join((str(2), text[i], str(0))))),
                                                  self.vocab.get_index(('_'.join((str(3), text[next], str(0))))),

                                                  # Bi-gram, with label 0
                                                  self.vocab.get_index('_'.join((str(4), text[prev], text[i], str(0)))),
                                                  self.vocab.get_index('_'.join((str(5), text[i], text[next], str(0)))),
                                                  self.vocab.get_index(
                                                      '_'.join((str(6), text[prev], text[next], str(0)))),

                                                  # Tri-gram, with label 0
                                                  self.vocab.get_index(
                                                      '_'.join((str(7), text[prev], text[i], text[next], str(0))))
                                              ), (
                                                  # Uni-gram, with label 1
                                                  self.vocab.get_index('_'.join((str(1), text[prev], str(1)))),
                                                  self.vocab.get_index(('_'.join((str(2), text[i], str(1))))),
                                                  self.vocab.get_index(('_'.join((str(3), text[next], str(1))))),

                                                  # Bi-gram, with label 1
                                                  self.vocab.get_index('_'.join((str(4), text[prev], text[i], str(1)))),
                                                  self.vocab.get_index('_'.join((str(5), text[i], text[next], str(1)))),
                                                  self.vocab.get_index(
                                                      '_'.join((str(6), text[prev], text[next], str(1)))),

                                                  # Tri-gram, with label 1
                                                  self.vocab.get_index(
                                                      '_'.join((str(7), text[prev], text[i], text[next], str(1))))
                                              )))
            self.labels.append(sentence_label)
            self.features.append(sentence_features)

    def show_vocab(self):
        """
        Print vocabulary
        :return: None
        """
        for i in range(self.vocab.size()):
            print('%4d' % i, '==>', self.vocab.get_word(i))

    def save_vocab(self, path):
        """
        Save vocabulary into a file
        :param path: save path
        :return: None
        """
        file = open(path, 'w')
        for i in range(self.vocab.size()):
            file.write(self.vocab.get_word(i) + '\n')
        file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item], self.words[item]

    def generate_features(self, text):
        """
        Generate features for a input sentence
        :param text: input sentence
        :return: list of features of *test*
        """
        text = '^' + text + '$'

        features = list()
        for i in range(1, len(text) - 1):
            prev = i - 1
            next = i + 1
            features.append(((
                                 self.vocab.get_index('_'.join((str(1), text[prev], str(0)))),
                                 self.vocab.get_index(('_'.join((str(2), text[i], str(0))))),
                                 self.vocab.get_index(('_'.join((str(3), text[next], str(0))))),

                                 # Bi-gram, with label 0
                                 self.vocab.get_index('_'.join((str(4), text[prev], text[i], str(0)))),
                                 self.vocab.get_index('_'.join((str(5), text[i], text[next], str(0)))),
                                 self.vocab.get_index(
                                     '_'.join((str(6), text[prev], text[next], str(0)))),

                                 # Tri-gram, with label 0
                                 self.vocab.get_index(
                                     '_'.join((str(7), text[prev], text[i], text[next], str(0))))
                             ), (
                                 # Uni-gram, with label 1
                                 self.vocab.get_index('_'.join((str(1), text[prev], str(1)))),
                                 self.vocab.get_index(('_'.join((str(2), text[i], str(1))))),
                                 self.vocab.get_index(('_'.join((str(3), text[next], str(1))))),

                                 # Bi-gram, with label 1
                                 self.vocab.get_index('_'.join((str(4), text[prev], text[i], str(1)))),
                                 self.vocab.get_index('_'.join((str(5), text[i], text[next], str(1)))),
                                 self.vocab.get_index(
                                     '_'.join((str(6), text[prev], text[next], str(1)))),

                                 # Tri-gram, with label 1
                                 self.vocab.get_index(
                                     '_'.join((str(7), text[prev], text[i], text[next], str(1))))
                             )))
        return features
