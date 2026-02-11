import unittest
import logging
import numpy as np
from pyblinker.fitutils.forking import (
    corr as corr_matlab,
    polyval as polyval_matlab,
    polyfit as polyfit_matlab,
    get_intersection,
)

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMatlabForking(unittest.TestCase):
    def test_polyfit_matlab(self):
        logger.info("Testing polyfit_matlab function")
        n = 1
        x = np.array([43, 44, 45, 46, 47, 48])
        y = np.array([15.399296760559082, 26.770189285278320, 40.020221710205080,
                      54.111049652099610, 67.944847106933600, 80.329727172851560])

        expected_p = np.array([24.709207534790040,47.429222106933594])
        expected_S = {
            'R': np.array([[0, -2.449489742783178],
                           [2.236067977499790, 0]]),
            'df': 4,
            'normr': 1.691495418548584,
            'rsquared': 0.999063611030579
        }
        expected_mu = np.array([45.5, 1.870828693386971])

        p, S, mu = polyfit_matlab(x, y, n)

        np.testing.assert_allclose(p, expected_p, rtol=1e-6)
        np.testing.assert_allclose(mu, expected_mu, rtol=1e-6)
        np.testing.assert_allclose(S['normr'], expected_S['normr'], rtol=1e-6)
        np.testing.assert_allclose(S['rsquared'], expected_S['rsquared'], rtol=1e-6)
        np.testing.assert_equal(S['df'], expected_S['df'])

        logger.info(f"Computed p: {p}, Expected p: {expected_p}")
        logger.info("Test polyfit_matlab passed.")
    def test_polyval_matlab(self):
        logger.info("Testing polyval_matlab function")
        p = np.array([24.709207534790040, 47.429222106933594])
        x = np.array([43, 44, 45, 46, 47, 48])
        mu = [45.5, 1.870828693386971]
        S = {
            'R': [[0, -2.449489742783178],
                  [2.236067977499790, 0]],  # Singular matrix
            'df': 4,
            'normr': 1.691495418548584,
            'rsquared': 0.999063611030579
        }

        y_expected = np.array([
            14.410156250000000,
            27.617780685424805,
            40.825408935546875,
            54.033035278320310,
            67.240661621093750,
            80.448287963867190
        ])

        y, delta = polyval_matlab(p, x, S=S, mu=mu)

        np.testing.assert_allclose(y, y_expected, rtol=1e-6, atol=1e-6)
        logger.info(f"Computed y: {y}, Expected y: {y_expected}")

        if delta is not None:
            logger.info(f"Computed delta: {delta}")
        else:
            logger.info("Delta computation skipped due to singular matrix.")

        logger.info("Test polyval_matlab passed.")

    def test_corr_matlab(self):
            logger.info("Testing corr_matlab function")
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
            logger.info(f"Computed coef: {coef[0, 0]}, Expected coef: {expected_coef}")

            np.testing.assert_almost_equal(coef[0, 0], expected_coef, decimal=6)
            logger.info("Test corr_matlab passed.")
    #
    def test_get_intersection(self):
            logger.info("Testing get_intersection function")
            p = [24.709207534790040, 47.429222106933594]
            q = [-23.652940750122070, 46.986415863037110]
            u = [45.500000000000000, 1.870828693386971]
            v = [59.500000000000000, 4.183300132670378]

            expected_leftXIntercept = 41.90895080566406
            expected_rightXIntercept = 67.81009674072266
            expected_xIntersect = 49.67326354980469
            expected_yIntersect = 102.5481115

            x_intersect, y_intersect, left_x_intercept, right_x_intercept = get_intersection(p, q, u, v)

            self.assertAlmostEqual(x_intersect, expected_xIntersect, places=3)
            self.assertAlmostEqual(y_intersect, expected_yIntersect, places=3)
            self.assertAlmostEqual(left_x_intercept, expected_leftXIntercept, places=3)
            self.assertAlmostEqual(right_x_intercept, expected_rightXIntercept, places=3)


            logger.info("Test get_intersection passed.")
    def test_lines_intersection(self):
        xLeft = np.array([43, 44, 45, 46, 47, 48])
        yLeft = np.array([15.399296760559082, 26.770189285278320, 40.020221710205080,
                  54.111049652099610, 67.944847106933600, 80.329727172851560])

        xRight = np.array([53, 54, 55, 56, 57, 58, 59, 60,
                           61, 62, 63, 64, 65, 66])
        yRight=np.array([81.576363,76.221619,72.655701,69.304802,64.227875,
                         57.023861,49.117550,42.331360,37.262333,32.985798,28.144085,
                         22.177107,15.572393,9.2089672])

        n=1
        pLeft, SLeft, muLeft = polyfit_matlab(xLeft, yLeft, n)
        yPred, delta = polyval_matlab(pLeft, xLeft, S=SLeft, mu=muLeft)
        leftR2, _ = corr_matlab(yLeft, yPred)



        pRight, SRight, muRight = polyfit_matlab(xRight, yRight, 1)
        yPredRight, delta = polyval_matlab(pRight, xRight, S=SRight, mu=muRight)
        rightR2, _ = corr_matlab(yRight, yPredRight)

        x_intersect, y_intersect, left_x_intercept, right_x_intercept = get_intersection(pLeft, pRight, muLeft, muRight)

        expected_leftXIntercept = 41.90895080566406
        expected_rightXIntercept = 67.81009674072266
        expected_xIntersect = 49.67326354980469
        expected_yIntersect = 102.5481115
        expected_leftR2=0.99953169
        # expected_pval_left=0
        expected_rightR2=0.99763674
        # expected_pval_right=0

        self.assertAlmostEqual(leftR2[0, 0], expected_leftR2, places=3)
        # self.assertAlmostEqual(pval_left, expected_pval_left, places=3)
        self.assertAlmostEqual(rightR2[0, 0],  expected_rightR2, places=3)
        # self.assertAlmostEqual(pval_right,  expected_pval_right, places=3)


        self.assertAlmostEqual(x_intersect, expected_xIntersect, places=3)
        self.assertAlmostEqual(y_intersect, expected_yIntersect, places=3)
        self.assertAlmostEqual(left_x_intercept, expected_leftXIntercept, places=3)
        self.assertAlmostEqual(right_x_intercept, expected_rightXIntercept, places=3)
        logger.info("Test get_intersection passed.")



if __name__ == '__main__':
    unittest.main()
