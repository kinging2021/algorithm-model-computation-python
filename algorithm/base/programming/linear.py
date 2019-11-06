from scipy import optimize as op


def linear_programming(obj, left=None, right=None, left_eq=None, right_eq=None, bounds=None):
    return op.linprog(obj, left, right, left_eq, right_eq, bounds)
