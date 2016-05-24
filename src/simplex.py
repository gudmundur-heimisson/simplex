'''
Created on May 22, 2016

@author: Gudmundur Heimisson
'''


class MaxIterationsReachedError(Exception):
    pass


class Simplex:

    def __init__(self, tableau, max_iters=10000):
        self.max_iters = max_iters
        self.iters = 0
        self.tableau = tableau

    def _do_until(self, do, until):
        while self.iters <= self.max_iters:
            if not until():
                self.iters += 1
                yield do()
            else:
                raise StopIteration()
        else:
            raise MaxIterationsReachedError()

    def __iter__(self):
        tableau = self.tableau
        if not tableau.canonical:
            yield from self._phase0()
        # After phase 0, tableau is either canonical or infeasible
        if tableau.infeasible:
            raise StopIteration()
        else:
            done = lambda: tableau.optimal or tableau.unbounded
            yield from self._do_until(tableau.simplex_pivot, done)

    def _phase0(self):
        '''
        Pivots by dual simplex until the tableau is canonical or infeasible
        '''
        tableau = self.tableau
        # Acquire a basis
        have_basis = lambda: all(tableau.basis) or tableau.infeasible
        yield from self._do_until(tableau.basis_pivot, have_basis)
        # Get to canonical form
        canonical_pivot = lambda: (tableau.dual_simplex_pivot()
                                   if all(tableau.c >= 0)
                                   else tableau.subproblem_pivot())
        is_canonical = lambda: tableau.canonical or tableau.infeasible
        yield from self._do_until(canonical_pivot, is_canonical)

    def solve(self):
        '''
        Solves the LP.
        Returns false if the LP is infeasible or unbounded.
        Returns true if the LP is successfully solved.
        If iterations exceeds max_iters, throws a MaxIterationsReachedError.
        '''
        for _ in self:
            pass
        return self.tableau.optimal
