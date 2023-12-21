import numpy as np
from sympy import symbols

class Solver():
    def __init__(self, A=None, b=None):
        self.A = A
        self.b = b
        self.C = None
        self.B = None
        self.solution = None
        
    def _system_integer_solver(self, A: list, b: list):
        A = np.array(A, dtype=object)
        b = np.array(b)
        equations, variables = A.shape
        if equations > variables:
            raise ValueError('System is poorly conditioned. Equations > Variables unsupported')
        C = np.eye(variables, dtype=object)
        B = np.eye(equations, dtype=object)
        var_id = {}
        for v in range(variables):
            var_id[v] = v

        def row_step(A, diag, var_id):
            col = 0
            gcd = np.gcd.reduce(A[diag:, diag + col])
            while gcd != 1:
                col += 1
                if col + diag == variables:
                    raise ValueError('Poorly conditioned')
                gcd = np.gcd.reduce(A[diag:, diag + col])
            column = A[diag:, diag + col].copy()
            column_change_index = col
            C_column_change = np.eye(variables - diag, dtype=object)
            C_column_change[0][0] = 0
            C_column_change[column_change_index][column_change_index] = 0
            C_column_change[0][column_change_index] = 1
            C_column_change[column_change_index][0] = 1
            
            C_column_change
            C = np.eye(variables, dtype=object)
            C[diag:, diag:] = C_column_change
            var_id[diag] = col + diag
            var_id[col + diag] = diag
            B_i = np.eye(equations - diag, dtype=object)
            while np.count_nonzero(column) != 1:
                B_step = []
                pivot_index = np.argmin(np.abs(column[np.nonzero(column)]))
                pivot = column[np.nonzero(column)][pivot_index]
                index = np.where(column == pivot)[0][0]
                for i in range(equations - diag):
                    transform = [0] * (equations - diag)
                    transform[i] = 1
                    if i != index:
                        scale = (column[i]//pivot)
                        transform[index] = -scale
                        column[i] -= scale * pivot
                    B_step.append(transform)
                B_i = B_step @ B_i
            row_change_index = np.nonzero(column)[0][0]
            B_row_change = np.eye(equations - diag, dtype=object)
            B_row_change[0][0] = 0
            B_row_change[row_change_index][row_change_index] = 0
            B_row_change[0][row_change_index] = 1
            B_row_change[row_change_index][0] = 1
            B_i = B_row_change @ B_i

            B = np.eye(equations, dtype=object)
            B[diag:, diag:] = B_i
            return A, B, C, var_id

        def column_step(A, diag):
            row = A[diag, diag:].copy()
            C_i = np.eye(variables - diag, dtype=object)

            while np.count_nonzero(row) != 1:
                C_step = []
                pivot_index = np.argmin(np.abs(row[np.nonzero(row)]))
                pivot = row[np.nonzero(row)][pivot_index]
                index = np.where(row == pivot)[0][0]
                for i in range(variables - diag):
                    transform = [0] * (variables - diag)
                    transform[i] = 1
                    if i != index:
                        scale = (row[i]//pivot)
                        transform[index] = -scale
                        row[i] -= scale * pivot
                    C_step.append(transform)
                C_step = np.array(C_step, dtype=object)
                C_step = C_step.T
                C_i = C_i @ C_step
            column_change_index = np.nonzero(row)[0][0]
            C_column_change = np.eye(variables - diag, dtype=object)
            C_column_change[0][0] = 0
            C_column_change[column_change_index][column_change_index] = 0
            C_column_change[0][column_change_index] = 1
            C_column_change[column_change_index][0] = 1
            
            C_i = C_i @ C_column_change
            
            C = np.eye(variables, dtype=object)
            C[diag:, diag:] = C_i
            return C


        for k in range(min(A.shape) - 1):
            A, B_i, C_i, var_id = row_step(A, k, var_id)
            B = B_i @ B
            C = C_i @ C # THIS WAS SO TRICKY AND COUNTER INTUITIVE (LOOK BELOW IN WHAT ORDER I MULTIPLY C_i) BUT IT GAVE ME 99% TEST PASSED AFTER THIS!
            A = B_i @ A
            
            C_i = column_step(A, k)
            C = C @ C_i
            A = A @ C_i
    
        if variables != equations and equations != 1:
            C_final = column_step(A, k+1)
            C = C @ C_final
            A = A @ C_final
        
        d = np.diag(A)
        if np.count_nonzero(d) != len(d):
            raise ValueError('No integer solution')
        b = B @ b
        x = [0] * equations
        for n in range(equations):
            flt = b[n] / d[n]
            num = b[n] // d[n]
            if flt != num:
                raise ValueError('No integer solution')
            x[n] = num
        
        self.B = B
        self.C = C
        for s in range(variables - equations):
            x = np.concatenate((x, np.array([symbols(f'x_{s}')])))
        solution = C @ x
        permuted_solution = []
        for m in range(variables):
            permuted_solution.append(solution[var_id[m]])
        self.solution = permuted_solution
        return permuted_solution

    def solve(self, *args, **kwargs):
        if self.solution is None:
            if self.A is None or self.b is None:
                self.A = kwargs.get('A', None)
                self.b = kwargs.get('b', None)
            self._system_integer_solver(self.A, self.b) 
        point = np.array(kwargs.get('point', None))
        solution = self.solution
        for l in range(len(point)):
            solution = [s.subs(f'x_{l}', point[l]) for s in solution]
        return solution