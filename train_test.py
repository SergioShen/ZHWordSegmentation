#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 11:37 PM
# @Author: Shen Sijie
# @File: main.py
# @Project: ZHWordSegmentation

from constant import *
from model import Perceptron, StructuredPerceptron
from dataset import Dataset


def train(model_name, average):
    """
    Train the model with train dataset
    :param model_name: model to be trained
    :param average: use average model or not
    :return: None
    """
    print('--------', 'Generating train dataset', '--------')
    train_dataset = Dataset('train')
    if not os.path.exists(VOCAB_PATH):
        train_dataset.save_vocab(VOCAB_PATH)
    model = Perceptron(train_dataset.vocab.size())

    print('--------', 'Training begins', '--------')
    for e in range(EPOCH):
        print('Epoch', e, '  ', end='', flush=True)
        for i in range(len(train_dataset)):
            features, labels, words = train_dataset[i]
            for j in range(len(features)):
                model.update(features[j], labels[j])

            if i % 2000 == 0:
                print('.', end='', flush=True)
        print('')
    print('--------', 'Training finished', '--------')

    model.save(os.path.join(MODEL_SAVE_PATH, model_name), average)


def test(model_name, output_file_path):
    """
    Test the saved model with test dataset
    :param model_name: model to be used
    :param output_file_path: test result output path
    :return: None
    """
    print('--------', 'Generating test dataset', '--------')
    test_dataset = Dataset('test')
    model = Perceptron(test_dataset.vocab.size())
    model.load(os.path.join(MODEL_SAVE_PATH, model_name))

    print('--------', 'Testing begins', '--------')
    output_file = open(output_file_path, 'w')
    for i in range(len(test_dataset)):
        features, labels, words = test_dataset[i]
        sentence = ''.join(words)
        for j in range(len(features)):
            pred = model.predict(features[j])
            output_file.write(sentence[j])
            if pred == 1:
                output_file.write('  ')
        output_file.write('\n')
    print('--------', 'Testing finished', '--------')
    print('Result saved at path', output_file_path)


def keyboard_test(model_name):
    """
    Get a string from terminal and print segmented sentences
    :param model_name: model to be used
    :return: None
    """
    print('--------', 'Loading vocabulary', '--------')
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


def structured_train(model_name, average):
    """
    Train the model with train dataset
    :param model_name: model to be trained
    :param average: use average model or not
    :return: None
    """
    print('--------', 'Generating train dataset', '--------')
    train_dataset = Dataset('train')
    if not os.path.exists(VOCAB_PATH):
        train_dataset.save_vocab(VOCAB_PATH)
    model = StructuredPerceptron(train_dataset.vocab.size())

    print('--------', 'Training begins', '--------')
    for e in range(EPOCH):
        print('Epoch', e, '  ', end='', flush=True)
        for i in range(len(train_dataset)):
            features, labels, words = train_dataset[i]

            if len(features) > 0:  # There exists empty sentence in the dataset
                model.update(features, labels)

            if i % 2000 == 0:
                print('.', end='', flush=True)
        print('')
    print('--------', 'Training finished', '--------')

    model.save(os.path.join(MODEL_SAVE_PATH, model_name), average)


def structured_test(model_name, output_file_path):
    """
    Test the saved model with test dataset
    :param model_name: model to be used
    :param output_file_path: test result output path
    :return: None
    """
    print('--------', 'Generating test dataset', '--------')
    test_dataset = Dataset('test')
    model = StructuredPerceptron(test_dataset.vocab.size())
    model.load(os.path.join(MODEL_SAVE_PATH, model_name))

    print('--------', 'Testing begins', '--------')
    output_file = open(output_file_path, 'w')
    for i in range(len(test_dataset)):
        features, labels, words = test_dataset[i]
        sentence = ''.join(words)
        pred = model.predict(features)
        for j in range(len(features)):
            output_file.write(sentence[j])
            if pred[j] == 1:
                output_file.write('  ')
        output_file.write('\n')
    print('--------', 'Testing finished', '--------')
    print('Result saved at path', output_file_path)


def structured_keyboard_test(model_name):
    """
    Get a string from terminal and print segmented sentences
    :param model_name: model to be used
    :return: None
    """
    print('--------', 'Loading vocabulary', '--------')
    test_dataset = Dataset('keyboard')
    model = StructuredPerceptron(test_dataset.vocab.size())
    model.load(os.path.join(MODEL_SAVE_PATH, model_name))

    print('现在可以开始输入了！')
    while True:
        text = input()
        text = text.strip()
        features = test_dataset.generate_features(text)
        pred = model.predict(features)
        for i in range(len(features)):
            print(text[i], end='')
            if pred[i] == 1:
                print('  ', end='')
        print('')
