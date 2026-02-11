import numpy as np
from pyblinker.fitutils.forking import polyval as polyval_matlab
# Unit test remains the same
def test_polyval_matlab():
    # Input variables
    p = np.array([24.709207534790040, 47.429222106933594])
    x = np.array([43, 44, 45, 46, 47, 48])
    mu = [45.5, 1.870828693386971]
    S = {
        'R': [[0, -2.449489742783178], [2.236067977499790, 0]],  # Singular matrix
        'df': 4,
        'normr': 1.691495418548584,
        'rsquared': 0.999063611030579
    }

    # Expected outputs
    y_expected = np.array([
        14.410156250000000,
        27.617780685424805,
        40.825408935546875,
        54.033035278320310,
        67.240661621093750,
        80.448287963867190
    ])

    # Run function
    y, delta = polyval_matlab(p, x, S=S, mu=mu)

    # Check outputs
    np.testing.assert_allclose(y, y_expected, rtol=1e-6, atol=1e-6)
    print("Test passed: y matches expected y")
    print("Computed y:", y)
    print("Expected y:", y_expected)
    if delta is not None:
        print("Computed delta:", delta)
    else:
        print("Delta computation skipped due to singular matrix.")

# Run the unit test
test_polyval_matlab()
