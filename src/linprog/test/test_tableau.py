'''
Created on May 21, 2016

@author: Gudmundur Heimisson
'''

import numpy as np
from linprog.test import LoggingTest
from linprog import Tableau


def error(X, Y):
    return np.sum(np.abs(X) - np.abs(Y))


class TableauTest(LoggingTest):

    def test_basis_pivot(self):
        T = Tableau(np.array([[0., -6., -5., -3., -7., 0., 0., 0.],
                              [50., 1., 1., 0., 3., 1., 0., 0.],
                              [150., 2., 1., 2., 1., 0., 1., 0.],
                              [80., 1., 1., 1., 4., 0., 0., 1.],
                              [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T2 = Tableau(np.array([[480., 0., 1., 3., 17., 0., 0., 6.],
                               [-30., 0., 0., -1., -1., 1., 0., -1.],
                               [-10., 0., -1., 0., -7., 0., 1., -2.],
                               [80., 1., 1., 1., 4., 0., 0., 1.],
                               [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T3 = Tableau(np.array([[426., 0., 1., 3., 17., 0., 0., 0.],
                               [-21., 0., 0., -1., -1., 1., 0., 0.],
                               [8., 0., -1., 0., -7., 0., 1., 0.],
                               [71., 1., 1., 1., 4., 0., 0., 0.],
                               [9., 0., 0., 0., 0., 0., 0., 1.]]))
        self.assertEqual(error(T.basis, [5, 6, 0, 0]), 0)
        T.basis_pivot()
        self.assertAlmostEqual(error(T.M, T2.M), 0)
        self.assertEqual(error(T.basis, [5, 6, 1, 0]), 0)
        T.basis_pivot()
        self.assertAlmostEqual(error(T.M, T3.M), 0)
        self.assertEqual(error(T.basis, [5, 6, 1, 7]), 0)

    def test_pivot(self):
        T = Tableau(np.array([[0., -6., -5., -3., -7., 0., 0., 0.],
                              [50., 1., 1., 0., 3., 1., 0., 0.],
                              [150., 2., 1., 2., 1., 0., 1., 0.],
                              [80., 1., 1., 1., 4., 0., 0., 1.],
                              [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T2 = Tableau(np.array([[300., 0., 1., -3., 11., 6., 0., 0.],
                               [50., 1., 1., 0., 3., 1., 0., 0.],
                               [50., 0., -1., 2., -5., -2., 1., 0.],
                               [30., 0., 0., 1., 1., -1., 0., 1.],
                               [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T3 = Tableau(np.array([[375., 0., -0.5, 0., 3.5, 3., 1.5, 0.],
                               [50., 1., 1., 0., 3., 1., 0., 0.],
                               [25., 0., -0.5, 1., -2.5, -1., 0.5, 0.],
                               [5., 0., 0.5, 0., 3.5, 0., -0.5, 1.],
                               [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T.pivot(1, 1)
        self.assertAlmostEqual(error(T.M, T2.M), 0)
        T.pivot(2, 3)
        self.assertAlmostEqual(error(T.M, T3.M), 0)

    def test_dual_simplex_pivot(self):
        T = Tableau(np.array([[426., 0., 1., 3., 17., 0., 0., 0.],
                              [-21., 0., 0., -1., -1., 1., 0., 0.],
                              [8., 0., -1., 0., -7., 0., 1., 0.],
                              [71., 1., 1., 1., 4., 0., 0., 0.],
                              [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T2 = Tableau(np.array([[69., 0., 1., -14., 0., 17., 0., 0.],
                               [21., 0., 0., 1., 1., -1., 0., 0.],
                               [155., 0., -1., 7., 0., -7., 1., 0.],
                               [-13., 1., 1., -3., 0., 4., 0., 0.],
                               [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T.dual_simplex_pivot()
        self.assertAlmostEqual(error(T.M, T2.M), 0)

    def test_subproblem_pivot(self):
        T = Tableau(np.array([[69., 0., 1., -14., 0., 17., 0., 0.],
                              [21., 0., 0., 1., 1., -1., 0., 0.],
                              [155., 0., -1., 7., 0., -7., 1., 0.],
                              [-13., 1., 1., -3., 0., 4., 0., 0.],
                              [9., 0., 0., 0., 0., 0., 0., 1.]]))
        T2 = Tableau(np.array([[363., 0., 1., 0., 14., 3., 0., 0.],
                               [21., 0., 0., 1., 1., -1., 0., 0.],
                               [8., 0., -1., 0., -7., 0., 1., 0.],
                               [50., 1., 1., 0., 3., 1., 0., 0.],
                               [9., 0., 0., 0., 0., 0., 0., 1.]]))
        pivot = T.get_subproblem_pivot()
        self.assertTupleEqual(pivot, (1, 3))
        T.subproblem_pivot()
        self.assertAlmostEqual(error(T.M, T2.M), 0)
        self.assertTrue(T.optimal)
