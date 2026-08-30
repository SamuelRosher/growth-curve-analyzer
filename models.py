import numpy as np
def gompertz(t, K, mu_max, lam):
    """
    Gompertz function.

    Parameters:
    t : array of time points (hours)
    K : carrying capacity (Maximum OD the culture reaches)
    mu_max : maximum specific growth rate (per hour)
    1am : lag phase time (hours)

    Returns:
    Predicted OD values at each time point
    """
    exponent = (mu_max * np.e / K) * (lam - t) + 1
    return K * np.exp(-np.exp(exponent))
"""
gompertz function is a sigmoid function that describes growth as a function of time.
 It is often used to model microbial growth, tumor growth, and other biological processes. 
 The parameters K, mu_max, and 1am control the shape of the curve, with K representing the maximum population size, 
 mu_max representing the maximum growth rate, and 1am representing the lag phase duration."""
def logistic(t, K, mu_max, t_mid):
    return K / (1 + np.exp(-mu_max * (t - t_mid)))
""" t_mid is the time point in which growth is teh fastest (the inflecion point of the curve)"""

