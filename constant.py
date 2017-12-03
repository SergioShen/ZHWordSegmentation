#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/1/17 8:36 PM
# @Author: Shen Sijie
# @File: constant.py
# @Project: ZHWordSegmentation

import os

# DEBUG flag
DEBUG = True

# Arguments of model
EPOCH = 10

# Constants
SPACE = [' ', '　']
UNKNOWN = '<unknown>'

# Path of everything
PROJECT_PATH = './'

DATASET_PATH = os.path.join(PROJECT_PATH, 'data_set')
TRAIN_DATA = os.path.join(DATASET_PATH, 'train.txt')
TEST_DATA = os.path.join(DATASET_PATH, 'test.txt')

RESULT_PATH = os.path.join(PROJECT_PATH, 'result')
TEST_ANSWER = os.path.join(RESULT_PATH, 'test.answer.txt')
TEST_OUTPUT = os.path.join(RESULT_PATH, "test.output.txt")
VOCAB_PATH = os.path.join(RESULT_PATH, 'vocab.txt')

MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, 'saved_model')
PERCEPTRON_MODEL = 'perceptron.model'
AVERAGE_PERCEPTRON_MODEL = 'perceptron.average.model'
STRUCTURED_PERCEPTRON_MODEL = 'perceptron.structured.model'

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)
