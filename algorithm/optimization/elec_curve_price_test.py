from .elec_curve_price import ElecCurvePrice, e_curve_price_arbitrage


def test_whether_exists_arbitrage():
    cp1 = ElecCurvePrice([1, 1, 1], 2, "B")
    cp2 = ElecCurvePrice([1.1, 1.1, 1.1], 2.1, "S")
    cp3 = ElecCurvePrice([1.1, 1.1, 1.1], 2.3, "S")
    ret1 = e_curve_price_arbitrage([cp1, cp2])
    ret2 = e_curve_price_arbitrage([cp1, cp3])
    assert (ret1[0] == False)
    assert (ret2[0] == True)
    return
