#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 8:43 PM
# @Author: Shen Sijie
# @File: vocab.py
# @Project: ZHWordSegmentation

from constant import *

import copy


class Vocab(object):
    """
    A vocabulary linking words with indexes
    """

    def __init__(self, data=None):
        """
        Initialize the vocab
        :param data: reserved words whose index is fixed
        """
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.reserved = []

        if data is not None:
            self.reserved = copy.deepcopy(data)
            for item in data:
                self.add(item)

    def size(self):
        """
        Get the size of the vocabulary
        :return: size of the vocabulary
        """
        return len(self.idxToLabel)

    def get_index(self, key, default=0):
        """
        Get index from word
        :param key: string of a word
        :param default: return value when *key* is not in vocabulary, default value is 0
        :return: index of *key* in vocabulary
        """
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def get_word(self, idx, default=UNKNOWN):
        """
        Get word from index
        :param idx: index of a word
        :param default: return value when *idx* is out of range, default value is *UNKNOWN*
        :return: word string whose index in vocabulary is *idx*
        """
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    def add(self, word):
        """
        Add a word into vocabulary
        :param word: string of word
        :return: index of *word*
        """
        if word in self.labelToIdx:
            idx = self.labelToIdx[word]
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = word
            self.labelToIdx[word] = idx
        return idx

    def get_words(self):
        """
        Get all words in the vocabulary
        :return: a list of all words in the vocabulary
        """
        return self.labelToIdx.keys()

    def sort(self):
        """
        Sort the vocabulary, but with reserved words unchanged
        :return: None
        """
        not_reserved = list(filter(lambda key: key not in self.reserved, self.labelToIdx.keys()))
        not_reserved.sort()
        self.idxToLabel.clear()
        self.labelToIdx.clear()

        for item in self.reserved:
            self.add(item)
        for item in not_reserved:
            self.add(item)
