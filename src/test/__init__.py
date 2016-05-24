'''
Created on May 23, 2016

@author: Gudmundur Heimisson
'''
import logging
import sys
from unittest import TestCase

class LoggingTest(TestCase):

    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
