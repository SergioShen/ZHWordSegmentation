#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 12/4/17 2:52 AM
# @Author: Shen Sijie
# @File: main.py
# @Project: ZHWordSegmentation

from train_test import *
from optparse import OptionParser

# Parse command line arguments
parser = OptionParser(usage='Usage: python %prog [-s] [-a] [-k] [-o <filename>]')
parser.add_option('-s', '--structured',
                  action='store_true',
                  dest='structured',
                  help='Use structured perceptron'
                  )
parser.add_option('-a', '--average',
                  action='store_true',
                  dest='average',
                  help='Use average arguments'
                  )
parser.add_option('-k', '--keyboard',
                  action='store_true',
                  dest='keyboard',
                  help='Run keyboard test'
                  )
parser.add_option('-o', '--output',
                  action='store',
                  dest='outputfile',
                  type='string',
                  default=TEST_OUTPUT,
                  help='Output test result into OUTPUTFILE')

(options, args) = parser.parse_args()

# Run the program
if options.structured:
    # Use structured model
    if options.average:
        USE_MODEL = AVERAGE_STRUCTURED_PERCEPTRON_MODEL
    else:
        USE_MODEL = STRUCTURED_PERCEPTRON_MODEL
    print('Model name:', USE_MODEL)

    if options.keyboard:
        # Begin keyboard test
        structured_keyboard_test(USE_MODEL)
    else:
        # Begin train and test
        structured_train(USE_MODEL, options.average)
        structured_test(USE_MODEL, options.outputfile)

else:
    if options.average:
        # Use unstructured model
        USE_MODEL = AVERAGE_PERCEPTRON_MODEL
    else:
        USE_MODEL = PERCEPTRON_MODEL
    print('Model name:', USE_MODEL)

    if options.keyboard:
        # Begin keyboard test
        keyboard_test(USE_MODEL)
    else:
        # Begin train and test
        train(USE_MODEL, options.average)
        test(USE_MODEL, options.outputfile)
