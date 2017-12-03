#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 11:37 PM
# @Author: Shen Sijie
# @File: main.py
# @Project: ZHWordSegmentation

from constant import *
from model import Perceptron
from dataset import Dataset

USE_MODEL = AVERAGE_PERCEPTRON_MODEL


def train(model_name):
    """
    Train the model with train dataset
    :param model_name: model to be trained
    :return: None
    """
    print('--------', 'Generating dataset', '--------')
    train_dataset = Dataset('train')
    if not os.path.exists(VOCAB_PATH):
        train_dataset.save_vocab(VOCAB_PATH)
    model = Perceptron(train_dataset.vocab.size())

    print('--------', 'Training begins', '--------')
    for e in range(EPOCH):
        for i in range(len(train_dataset)):
            features, labels, words = train_dataset[i]
            for j in range(len(features)):
                model.update(features[j], labels[j])

            if i % 1000 == 0:
                print('Epoch %d, step %d' % (e, i))

    model.save(os.path.join(MODEL_SAVE_PATH, model_name))


def test(model_name):
    """
    Test the saved model with test dataset
    :param model_name: model to be used
    :return: None
    """
    print('--------', 'Generating dataset', '--------')
    test_dataset = Dataset('test')
    model = Perceptron(test_dataset.vocab.size())
    model.load(os.path.join(MODEL_SAVE_PATH, model_name))

    print('--------', 'Testing begins', '--------')
    output_file = open(TEST_OUTPUT, 'w')
    for i in range(len(test_dataset)):
        features, labels, words = test_dataset[i]
        sentence = ''.join(words)
        for j in range(len(features)):
            pred = model.predict(features[j])
            output_file.write(sentence[j])
            if pred == 1:
                output_file.write('  ')
        output_file.write('\n')


def keyboard_test(model_name):
    """
    Get a string from terminal and print segmented sentences
    :param model_name: model to be used
    :return: None
    """
    print('--------', 'Generating dataset', '--------')
    test_dataset = Dataset('keyboard')
    model = Perceptron(test_dataset.vocab.size())
    model.load(os.path.join(MODEL_SAVE_PATH, model_name))

    print('现在可以开始输入了！')
    while True:
        text = input()
        text = text.strip()
        features = test_dataset.generate_features(text)
        for i in range(len(features)):
            pred = model.predict(features[i])
            print(text[i], end='')
            if pred == 1:
                print('  ', end='')
        print('')


if __name__ == '__main__':
    train(USE_MODEL)
    test(USE_MODEL)
    keyboard_test(USE_MODEL)
