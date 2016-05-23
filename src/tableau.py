'''
Created on May 21, 2016

@author: Gudmundur Heimisson
'''
import numpy as np


class PivotException(Exception):
    pass


class InvalidBranchException(Exception):
    pass


class Tableau:

    def __init__(self, array):
        self._canonical = None
        self._optimal = None
        self._infeasible = None
        self._unbounded = None
        self._basis = None
        self._vars = None
        self._array = array

    def __str__(self):
        return str(self._array)

    def __repr__(self):
        return repr(self._array)

    @property
    def n(self):
        return self._array.shape[0]

    @property
    def m(self):
        return self._array.shape[1]

    @property
    def A(self):
        return self._array[1:self.n, 1:self.m]

    @property
    def b(self):
        return self._array[1:self.n, 0]

    @property
    def c(self):
        return self._array[0, 1:self.m]

    @property
    def z(self):
        return -self._array[0, 0]

    @property
    def M(self):
        return self._array

    @property
    def x(self):
        if self._vars is None and self.canonical:
            # Initialize to zeros
            self._vars = np.zeros(self.m - 1)
            for col_index, b in zip(self.basis_cols, self.b):
                self._vars[col_index - 1] = b
        return self._vars

    @property
    def canonical(self):
        if self._canonical is None:
            # Check for identity columns
            if (all(self.basis)
                    and all(b >= 0 for b in self.b)):
                self._canonical = True
            else:
                self._canonical = False
        return self._canonical

    @property
    def optimal(self):
        if self._optimal is None:
            self._optimal = self.canonical and all(c >= 0 for c in self.c)
        return self._optimal

    @property
    def infeasible(self):
        if self._infeasible is None:
            for i, b in enumerate(self.b):
                if ((b != 0 and all(a == 0 for a in self.A[i, :]))
                        or (b < 0 and all(a >= 0 for a in self.A[i, :]))
                        or (b > 0 and all(a <= 0 for a in self.A[i, :]))):
                    self._infeasible = True
                    break
            else:
                self._infeasible = False
        return self._infeasible

    @property
    def unbounded(self):
        if self._unbounded is None:
            if not self._canonical:
                self._unbounded = False
            else:
                self._unbounded = any(c < 0 and
                                      all(a <= 0 for a in self.A[:, i])
                                      for i, c in enumerate(self.c))
        return self._unbounded

    @property
    def basis(self):
        if self._basis is None:
            cols = np.zeros(self.n - 1, dtype='int_')
            for col_index in range(1, self.m):
                is_basic, x_index = self._is_basic_col(col_index)
                if is_basic and not cols[x_index - 1]:
                    cols[x_index - 1] = col_index
            self._basis = cols
        return self._basis

    def _is_basic_col(self, col_index):
        '''
        Checks whether the corresponding column is an identity column
        (i.e., the corresponding variable is basic).
        Returns a tuple.
        If the column is an identity column,
        returns true and the index of the 1 in the column.
        If the column is not an identity column, returns false and None
        '''
        one_index = None
        for row_index, val in enumerate(self._array[:, col_index]):
            if val != 0 and val != 1:
                return False, None
            elif val == 1:
                if one_index is not None:
                    return False, None
                else:
                    one_index = row_index
        if one_index is None:  # All elements are 0
            return False, None
        return True, one_index

    def _del_empty_rcs(self):
        arr = self._array
        rows = np.abs(arr).sum(0) != 0
        cols = np.abs(arr).sum(1) != 0
        self._array = arr[np.ix_(rows, cols)]

    def pivot(self, row, column):
        r, c = row, column
        arr = self._array
        if r <= 0 or c <= 0:
            raise PivotException('Invalid pivot! Must pivot in A!')
        pivot = arr[r, c]
        if pivot == 0:
            raise PivotException('Pivot must be non-zero!')
        if pivot != 1:
            arr[r, :] /= pivot
        for i in range(self.n):
            if i == r:
                continue
            a_ic = arr[i, c]
            if a_ic != 0:
                arr[i, :] -= a_ic * arr[r, :]
        self._canonical = None
        self._optimal = None
        self._infeasible = None
        self._unbounded = None
        self._basis = None
        self._vars = None
        return self

    def get_simplex_pivot(self):
        if not self.canonical:
            raise PivotException("Must be in canonical form.")
        if self.optimal or self.unbounded:
            return None
        # Find column with most negative c
        j, _ = min(enumerate(self.c), key=lambda t: t[1])
        # Find minimum ratio row
        i, _ = min(((i, b / a)
                    for i, (a, b) in enumerate(zip(self.A[:, j], self.b))
                    if a > 0),
                   key=lambda x: x[1])
        return i + 1, j + 1

    def get_dual_simplex_pivot(self):
        if not all(c >= 0 for c in self.c):
            raise PivotException("Must have c >= 0 to dual simplex pivot")
        if any(col is None for col in self.basis):
            raise PivotException('Must have full set of basis columns')
        if self.optimal or self.infeasible:
            return None
        # Get row with most negative b
        i, _ = min(((i, b) for i, b in enumerate(self.b) if b < 0),
                   key=lambda t: t[1])
        # Find maximum ratio column
        j, _ = min(((j, c / a)
                    for j, (a, c) in enumerate(zip(self.A[i, :], self.c))
                    if a < 0),
                   key=lambda t: t[1])
        return i + 1, j + 1

    def get_basis_pivot(self):
        # Find out which basis rows are missing, and which cols do not
        # already contain a basis
        cols = list(range(1, self.m))
        for basis_index, col_index in enumerate(self.basis):
            if col_index:
                cols.remove(col_index)
            else:
                row = basis_index + 1
                break
        else:
            # Already have basis
            return None
        for c in cols:
            if self._array[row, c] != 0:
                return row, c
        else:
            raise PivotException('Impossible to establish basis.')

    def get_subproblem_pivot(self):
        if np.sum(self.basis == 0) > 0:
            raise PivotException("Must have full basis")
        if self.canonical or self.infeasible:
            return None
        arr = self._array[1:, :]
        b_negs = self.b < 0
        i_b = next(i for i,b in enumerate(b_negs) if b)
        # Form a subproblem with one of the b's
        sub_c = arr[i_b, :].reshape(1, self.m)
        sub_bA = arr[~b_negs, :]
        sub_M = Tableau(np.r_[sub_c, sub_bA])
        pivot = sub_M.get_simplex_pivot()
        if not pivot:
            # Subproblem is unbounded or optimal
            if sub_M.optimal:
                # Original tableau is infeasible
                self._infeasible = True
                return None
            else:
                r = i_b
                c = next(j for j, a in enumerate(self.A[r, :]) if a < 0)
                pivot = r, c
        else:
            sub_r, c = pivot
            r = next(i for i,b in enumerate(~b_negs) if b and i == sub_r)
            pivot = r, c
        return pivot

    def simplex_pivot(self):
        pivot = self.get_simplex_pivot()
        if not pivot:
            return self  # Can't simplex pivot
        return self.pivot(*pivot)

    def dual_simplex_pivot(self):
        pivot = self.get_dual_simplex_pivot()
        if not pivot:
            return self
        return self.pivot(*pivot)

    def basis_pivot(self):
        pivot = self.get_basis_pivot()
        if not pivot:
            return self
        return self.pivot(*pivot)

    def subproblem_pivot(self):
        pivot = self.get_subproblem_pivot()
        if not pivot:
            return self
        return self.pivot(*pivot)
