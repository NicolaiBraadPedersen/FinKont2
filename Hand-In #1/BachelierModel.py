import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.stats import norm

# below is a personal style for plotting, will only work if the whole repo is cloned
from utils.plotting import use_earthy_style
use_earthy_style()

class Bachelier():
    def __init__(self):
        self.r = 0
        self.S0 = 100
        self.capT = 0.25
        self.true_vol = 15

        self.has_results = False

    def Bach_price(self, s, t, K):
        tau = self.capT - t
        sigma = self.true_vol
        d1 = (s-K) / (sigma*np.sqrt(tau))

        return (s-K) * norm.cdf(d1) + sigma * np.sqrt(tau) * norm.pdf(d1)

    def BS_price(self, s, t, K, imp_vol):
        tau = self.capT - t
        sigma = imp_vol

        d1 = 1/(sigma*np.sqrt(tau)) * ( np.log(s/K) + (self.r + 1/2 * sigma**2) * tau )
        d2 = d1 - sigma*np.sqrt(tau)

        return s * norm.cdf(d1) -np.exp(-self.r * tau) * K * norm.cdf(d2)

    def price_diff(self, imp_vol, K):
        s = self.S0
        t = 0

        return self.Bach_price(s = s, t = t, K = K) - self.BS_price(s = s, t = t, K = K, imp_vol = imp_vol)

    def implied_volatility(self, K, normalized):
        results = []

        for k in K:
            sol = root(
                self.price_diff,
                20,
                args=(k,)
            )
            if sol.success:
                if normalized:
                    iv = sol.x[0] * self.S0
                else :
                    iv = sol.x[0]
            else:
                iv = np.nan

            results.append(iv)

        self.results = np.array([results , K])
        self.has_results = True

        return results

    def plot_implied_volatility(self, normalized):
        if not self.has_results:
            print('No results are available for plotting. Run self.implied_volatility!')
            return

        if normalized:
            plt.hlines(self.true_vol, xmin = np.min(self.results[1, :]), xmax = np.max(self.results[1, :]), linestyles = 'dashed', label='True Volatility', color = "#724E25")


        plt.scatter(self.results[1, :], self.results[0, :], s=10, label = 'Implied Volatility')

        plt.xlabel("Strike")
        if normalized:
            plt.ylabel("Implied Volatility * S_0")
        else:
            plt.ylabel("Implied Volatility")
        plt.title(f"IV for Bachelier model | S_0 = {self.S0}")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    bach = Bachelier()
    Strikes = np.arange(80, 120 + 1)

    bach.implied_volatility(K = Strikes, normalized=False)
    bach.plot_implied_volatility(normalized=False)

    bach.implied_volatility(K=Strikes, normalized=True)
    bach.plot_implied_volatility(normalized=True)

    bach.S0 = 50
    bach.implied_volatility(K=Strikes, normalized=False)
    bach.plot_implied_volatility(normalized=False)

    bach.implied_volatility(K=Strikes, normalized=True)
    bach.plot_implied_volatility(normalized=True)