import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

def call_price(s,t,T,K,r,sigma):

    d1 = 1/(sigma*np.sqrt(T-t)) * ( np.log(s/K) + (r+0.5*sigma**2)*(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    price = s * norm.cdf(d1) - np.exp(-r*(T-t))*K*norm.cdf(d2)
    return price

def put_price(s,t,T,K,r,sigma):
    price = K*np.exp(-r*(T-t)) + call_price(s,t,T,K,r,sigma) - s
    return price

def call_delta(s,t,T,K,r,sigma):
    d1 = 1 / (sigma * np.sqrt(T - t)) * (np.log(s / K) + (r + 0.5 * sigma ** 2) * (T - t))
    delta = norm.cdf(d1)
    return delta

def put_delta(s,t,T,K,r,sigma):
    delta = call_delta(s,t,T,K,r,sigma) - 1
    return delta

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


if __name__ == '__main__':
    print(f'Price 1 = {put_price(1,0,30,np.exp(0.02*30),0.02,0.2*0.5):2f}')
    print(f'Price 2 = {put_price(1*np.exp((-0.5*0.02+0.5*0.07)*30),0,30,np.exp(0.02*30),0.02,0.2*0.5):2f}')
    print(f'Price 3 = {put_price(1*np.exp(((0.5*0.02+(0.5*(0.5-1)*0.2**2)/2-0.02))*30),0,30,np.exp(0.02*30),0.02,0.2*0.5):2f}')
    print(f'Price 4 = {0.5*put_price(1,0,30,np.exp(0.02*30),0.02,0.2):2f}')