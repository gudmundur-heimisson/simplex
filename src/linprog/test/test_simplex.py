'''
Created on May 22, 2016

@author: Gudmundur Heimisson
'''

import numpy as np
import logging
import sys
from linprog.test import LoggingTest
from linprog import Tableau
from linprog import Simplex


class SimplexTest(LoggingTest):

    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    def test_simplex(self):
        # From HW2 #
        M1 = Tableau(np.array([0., -1, -2, -1, 0, 0,
                               4, 1, 1, 1, 1, 0,
                               0, 1, -1, 0, 0, 1]).reshape(3, 6))
        # From HW3 #
        M2 = Tableau(np.array([0., -1, 1, 2, -1, 0, 0,
                               4, 2, 3, 0, -3, 1, 0,
                               -2, -1, 0, -1, 0, 0, 1,
                               1, 1, 2, 0, -2, 0, 0]).reshape(4, 7))
        M3 = Tableau(np.array([-6., -1, 1, -1, 0,
                               8, 2, -2, -6, 0,
                               6, 2, 3, -2, 1]).reshape(3, 5))
        # From HW4 #
        M4 = Tableau(np.array([0., -1, 1, -1, 2, 0, 0,
                               4, 2, 3, -3, 0, 1, 0,
                               2, 1, 0, 0, 1, 0, -1,
                               1, 1, 2, -2, 0, 0, 0]).reshape(4, 7))
        M5 = Tableau(np.array([3., 1, 0, 0, 1, 0,
                               -1, 1, 1, 0, -1, 0,
                               -4, -1, 0, 1, -1, 0,
                               1, 1, 0, 0, 0, 1]).reshape(4, 6))
        # From HW5 #
        M6 = Tableau(np.array([0., 0, 2, 0, 0,
                               -3, -1, -1, 1, 0,
                               -1, 0, -1, 0, 1]).reshape(3, 5))
        # From HW6 #
        M7 = Tableau(np.array([0., 0, -1, -1, -1, 0,
                               10, 1, 1, 1, 1, 0,
                               5, 0, 1, 1, 1 / 6, 1]).reshape(3, 6))
        M8 = Tableau(np.array([0., 10, 5, 0, 0, 0,
                               1, 1, 1, -1, 0, 0,
                               1, 1, 1, 0, -1, 0,
                               1, 1, 1 / 6, 0, 0, -1]).reshape(4, 6))
        # From HW7 #
        M9 = Tableau(np.array([395., 0, 0, 0, 7, 3, 1, 1,
                               -5, 0, 1, 0, 7, 0, -1, 2,
                               55, 1, 0, 0, -4, 1, 1, -2,
                               30, 0, 0, 1, 1, -1, 0, 1]).reshape(4, 8))
        M10 = Tableau(np.array([0., -6, -5, -3, -7, 0, 0, 0,
                                50, 1, 1, 0, 3, 1, 0, 0,
                                150, 2, 1, 2, 1, 0, 1, 0,
                                80, 1, 1, 1, 4, 0, 0, 1,
                                9, 0, 0, 0, 0, 0, 0, 1]).reshape(5, 8))
        M11 = Tableau(np.array([0., -6, -5, -3, -7, 0, 0, 0,
                                50, 1, 1, 0, 3, 1, 0, 0,
                                165, 2, 1, 2, 1, 0, 1, 0,
                                80, 1, 1, 1, 4, 0, 0, 1]).reshape(4, 8))
        tableaus = [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11]
        for i, M in enumerate(tableaus):
            print("Tableau:", M, sep="\n")
            s = Simplex(M)
            for i, t in enumerate(s):
                print(i, t.z, sep='= ')
            print(M)
            self.assertTrue(M.optimal or M.infeasible)
            print(s.iters, " iterations to solve.")
            print("Optimal!" if M.optimal else "Infeasible!")
            print('{:=^75}'.format(''))
