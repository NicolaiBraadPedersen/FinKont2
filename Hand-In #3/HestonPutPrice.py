import numpy as np
from scipy.integrate import quad

def bs_fourier(spot, timetoexp, strike, r, divyield, sigma):
    X = np.log(spot / strike) + (r - divyield) * timetoexp

    def integrand(k):
        return np.real(
            np.exp((-1j * k + 0.5) * X - 0.5 * (k**2 + 0.25) * sigma**2 * timetoexp)
            / (k**2 + 0.25)
        )

    integral, _ = quad(integrand, -np.inf, np.inf)
    return np.exp(-divyield * timetoexp) * spot - strike * np.exp(-r * timetoexp) * integral / (2 * np.pi)

def heston_fourier(spot, timetoexp, strike, r, divyield, V, theta, kappa, epsilon, rho, greek=1):
    X = np.log(spot / strike) + (r - divyield) * timetoexp
    kappahat = kappa - 0.5 * rho * epsilon
    xiDummy = kappahat**2 + 0.25 * epsilon**2

    def integrand(k):
        xi = np.sqrt(k**2 * epsilon**2 * (1 - rho**2) + 2j * k * epsilon * rho * kappahat + xiDummy)
        Psi_P = -(1j * k * rho * epsilon + kappahat) + xi
        Psi_M =  (1j * k * rho * epsilon + kappahat) + xi
        alpha = -kappa * theta * (Psi_P * timetoexp + 2 * np.log(
            (Psi_M + Psi_P * np.exp(-xi * timetoexp)) / (2 * xi)
        )) / epsilon**2
        beta = -(1 - np.exp(-xi * timetoexp)) / (Psi_M + Psi_P * np.exp(-xi * timetoexp))
        numerator = np.exp((-1j * k + 0.5) * X + alpha + (k**2 + 0.25) * beta * V)

        if greek == 1: return np.real(numerator / (k**2 + 0.25))
        if greek == 2: return np.real((0.5 - 1j * k) * numerator / (spot * (k**2 + 0.25)))
        if greek == 3: return -np.real(numerator / spot**2)
        if greek == 4: return np.real(numerator * beta)

    integral, _ = quad(integrand, -100, 100, limit=200)

    if greek == 1: return np.exp(-divyield * timetoexp) * spot - strike * np.exp(-r * timetoexp) * integral / (2 * np.pi)
    if greek == 2: return np.exp(-divyield * timetoexp) - strike * np.exp(-r * timetoexp) * integral / (2 * np.pi)
    if greek == 3: return -strike * np.exp(-r * timetoexp) * integral / (2 * np.pi)
    if greek == 4: return -strike * np.exp(-r * timetoexp) * integral / (2 * np.pi)

def andreasen_fourier(spot, timetoexp, strike, Z, lam, beta, epsilon):
    X = np.log(spot / strike)

    def integrand(k):
        neweps = lam * epsilon
        xi = np.sqrt(k**2 * neweps**2 + beta**2 + 0.25 * neweps**2)
        Psi_P = -beta + xi
        Psi_M =  beta + xi
        A = -beta * (Psi_P * timetoexp + 2 * np.log(
            (Psi_M + Psi_P * np.exp(-xi * timetoexp)) / (2 * xi)
        )) / epsilon**2
        B = (1 - np.exp(-xi * timetoexp)) / (Psi_M + Psi_P * np.exp(-xi * timetoexp))
        return np.real(np.exp((-1j * k + 0.5) * X + A - (k**2 + 0.25) * B * lam**2 * Z) / (k**2 + 0.25))

    integral, _ = quad(integrand, -np.inf, np.inf)
    return spot - strike * integral / (2 * np.pi)

if __name__ == '__main__':
    # note tha the below only holds since the put-call parity reduces to put=call, due to the specific parameter values!
    print('a=0.5:'+f'{heston_fourier(1, 30, np.exp(0.02 * 30), 0.02, 0, 0.2 ** 2*0.5**2, 0.2 ** 2 * 0.5 ** 2, 2, 1 * 0.5, -0.5)}')
    print('a=1:'+f'{heston_fourier(1, 30, np.exp(0.02 * 30), 0.02, 0, 0.2 ** 2, 0.2 ** 2, 2, 1, -0.5)}')