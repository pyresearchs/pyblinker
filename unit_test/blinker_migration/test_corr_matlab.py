import numpy as np
from pyblinker.fitutils.forking import corr as corr_matlab
def test_corr_matlab():
    x = [15.399296760559082,
         26.770189285278320,
         40.020221710205080,
         54.111049652099610,
         67.944847106933600,
         80.329727172851560]

    y = [14.410156,
         27.617781,
         40.825409,
         54.033035,
         67.240662,
         80.448288]

    expected_coef = 0.999531686306000

    coef, pval = corr_matlab(x, y)
    print("Computed coef:", coef[0, 0])
    print("Expected coef:", expected_coef)

    np.testing.assert_almost_equal(coef[0, 0], expected_coef, decimal=6)
    print("Test passed. Coefficient:", coef[0, 0])

if __name__ == '__main__':
    test_corr_matlab()
