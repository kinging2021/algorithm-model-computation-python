import numpy as np
from algorithm.base.programming.linear import linear_programming


class ElecCurvePrice:
    def __init__(self, curve, price, deal_tag, datetime_list=None):
        self.curve = np.array(curve)
        self.price = price
        assert (deal_tag == "B" or deal_tag == "S")
        self.tag = deal_tag
        if datetime_list is not None:
            assert (len(datetime_list) == len(self.curve))
        self.datetime_list = datetime_list

    def __len__(self):
        return len(self.curve)


def e_curve_price_arbitrage(curve_price_list):
    min_len, max_len = np.inf, 0
    for c_p in curve_price_list:
        if len(c_p) < min_len:
            min_len = len(c_p)
        if len(c_p) > max_len:
            max_len = len(c_p)
    if min_len != max_len:
        print("the length of electronic curve must be same")
        return
    obj = []
    constraint = []
    for c_p in curve_price_list:
        coeff = 1
        if c_p.tag == "S":
            coeff = -1
        obj.append(coeff * c_p.price)
        constraint.append(-1 * coeff * c_p.curve)

    A_ub = np.array(constraint).T
    B_ub = np.array([0] * A_ub.shape[0])

    bounds = ((0, 100),) * len(curve_price_list)

    result = linear_programming(obj, A_ub, B_ub, bounds=bounds)

    if result.fun < -1 * 1E-3:
        return True, result
    else:
        return False, result

    return


def get_e_price_bounds(curve_price_list, curve):
    # 买入不能套利时，上下界
    # 卖出不能套利时，上下界
    # TODO
    return
