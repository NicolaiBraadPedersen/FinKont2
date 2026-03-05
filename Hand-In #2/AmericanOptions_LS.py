import numpy as np
from scipy.optimize import minimize

class AmericanOptions:
    def __init__(self):
        self.S = np.array([
                [1.00, 1.09, 1.08, 1.34],
                [1.00, 1.16, 1.26, 1.54],
                [1.00, 1.22, 1.07, 1.03],
                [1.00, 0.93, 0.97, 0.92],
                [1.00, 1.11, 1.56, 1.52],
                [1.00, 0.76, 0.77, 0.90],
                [1.00, 0.92, 0.84, 1.01],
                [1.00, 0.88, 1.22, 1.34]
            ])
        self.S0 = 1
        self.r = 0.06
        self.K = 1.1
        self.T = 3
        self.N = 50

    def volatility_ml(self):
        est_s = self.estimates_s()

        def likelyhood(sigma):
            mu = self.r - sigma**2 * 0.5
            return np.sum(np.log(sigma**2)*0.5 + 1/ (2 * sigma**2) * (est_s - mu) ** 2)

        var_est = minimize(likelyhood, 1).x

        var_sd = 1/(2*var_est * np.sqrt(len(est_s)/(2*var_est**4) + len(est_s)/(4*var_est**2)))

        return var_est, var_sd


    def volatility_rv(self):
        est_s = self.estimates_s()

        var_est = np.std(est_s, ddof=1)
        var_sd = var_est/np.sqrt(2*len(est_s)-1)
        return var_est, var_sd

    def estimates_s(self):
        est = np.log(self.S[:,1:]/self.S[:,:-1])
        out= []
        for row in est:
            for val in row:
                out.append(val)
        return out

    def binomial_put_price(self,sigma, T = 3, N=1000, price_type='American'):
        K = self.K
        S = self.S0
        dt = 1 / N
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        R = np.exp(self.r * dt)
        p = (R - d) / (u - d)

        X = np.array([max(0, K - (S * (d ** k * u ** (T * N - k)))) for k in range(T * N + 1)])

        XX = X.copy()

        X = np.delete(X, -1)
        XX = np.delete(XX, 0)

        Y = np.array([max(0, K - (S * (d ** k * u ** (T * N - k)))) for k in range(T * N)])

        for i in range(T * N, 0, -1):
            ev = Y
            bdv = R ** -1 * (p * X + (1 - p) * XX)

            if (price_type == 'Bermudian') and (i%N != 0):
                val = bdv
            elif price_type == 'European':
                val = bdv
            else:
                val = np.maximum(bdv, ev)

            X_temp = X.copy()
            X = np.delete(val.copy(), -1)
            XX = np.delete(val.copy(), 0)
            Y = np.delete(X_temp, 0)
        return (val[0])

    @staticmethod
    def lag_pol(X, amount = 2):
        exp_neg_half_X = np.exp(-X / 2)
        L_0 = exp_neg_half_X
        L_1 = exp_neg_half_X * (1 - X)
        L_2 = exp_neg_half_X * (1 - 2 * X + X ** 2 / 2)
        L_3 = exp_neg_half_X * (1 - 3 * X + 3 * X ** 2 / 2 - X ** 3 / 6)
        L_4 = exp_neg_half_X * (1 - 4 * X + 3 * X ** 2 - 2 * X ** 3 / 3 + X ** 4 / 24)
        L_5 = exp_neg_half_X * (1 - 5 * X + 5 * X ** 2 - 5 * X ** 3 / 3 + 5 * X ** 4 / 24 - X ** 5 / 120)

        bases = {
            2: (L_0, L_1, L_2),
            3: (L_0, L_1, L_2, L_3),
            4: (L_0, L_1, L_2, L_3, L_4),
            5: (L_0, L_1, L_2, L_3, L_4, L_5),
        }

        return np.column_stack(bases[amount])

    def lsm_put_price(self, amount = 2):
        r = self.r
        data = self.S
        T = self.T
        N = self.N

        CFM = np.maximum(self.K - data, 0)
        CFM[:, 0] = 0
        A_CFM = CFM.copy()

        disc = np.exp(-r*(1/N))

        for i in range(T*N-1, 1, -1):
            ITM = CFM[:, i - 1] > 0
            if ITM.sum() != 0:
                continue
            Y = disc * A_CFM[ITM, i]
            X = data[ITM, i - 1]
            Z = self.lag_pol(X, amount = amount)

            Q = self.lag_pol(data[:, i - 1])
            b = np.linalg.solve(Z.T @ Z, Z.T @ Y)
            E = Q @ b

            CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
            CFM[CFM[:, i - 1] > 0, i:] = 0
            A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

        disc_vec = np.vander([disc], N * T + 1, increasing=True).T

        disc_CashFlows = CFM @ disc_vec

        price = np.mean(disc_CashFlows)

        return price

    def BM(self, sigma, T=3, Omega=10000):
        N = self.N
        dt = 1 / N
        Z = np.random.normal(0, np.sqrt(dt), size=(Omega, T * N))
        x = np.zeros((Omega, T * N + 1))
        x[:, 0] = self.S0
        x[:, 1:] = self.S0 * np.exp(np.cumsum((self.r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1))
        return x

    def sim_exp_10000(self):
        prices = []
        self.N = 1
        for i in range(10000):
            np.random.seed(i)
            self.S = self.BM(sigma=0.15,T=3,Omega=8)
            price = self.lsm_put_price(amount=2, price_type='Bermudian')
            prices.append(price)

        return prices

if __name__ == '__main__':
    amr = AmericanOptions()
    # res = amr.volatility_ml()
    # res2 = amr.volatility_rv()
    print(amr.binomial_put_price(sigma=0.15))
    print(amr.binomial_put_price(sigma=0.15, price_type='Bermudian'))
    print(amr.binomial_put_price(sigma=0.15, price_type='European'))
    #amr.S = amr.BM(sigma = 0.1484)
    #print(amr.lsm_put_price())
    #b = amr.sim_exp_10000()

    # print(res,res2)