import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# below is a personal style for plotting, will only work if the whole repo is cloned
from utils.plotting import use_earthy_style
use_earthy_style()

class AmericanOptions:
    def __init__(self):
        self.filepath = ''
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

    def binomial_put_price(self, sigma, T=3, N=1000, price_type='American'):
        K = self.K
        S = self.S0
        dt = 1 / N
        total_steps = int(T * N)

        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        R = np.exp(self.r * dt)
        q = (R - d) / (u - d)

        st_prices = S * (u ** np.arange(total_steps + 1)) * (d ** np.arange(total_steps, -1, -1))
        values = np.maximum(0, K - st_prices)

        for i in range(total_steps - 1, -1, -1):
            continuation_val = (q * values[1:] + (1 - q) * values[:-1]) / R

            current_s = S * (u ** np.arange(i + 1)) * (d ** np.arange(i, -1, -1))
            exercise_val = np.maximum(0, K - current_s)

            if price_type == 'European':
                values = continuation_val
            elif price_type == 'American':
                values = np.maximum(continuation_val, exercise_val)
            elif price_type == 'Bermudian':
                if (i + 1) % N == 0:
                    values = np.maximum(continuation_val, exercise_val)
                else:
                    values = continuation_val

        return values[0]

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
            1: (L_0,L_1),
            2: (L_0, L_1, L_2),
            3: (L_0, L_1, L_2, L_3),
            4: (L_0, L_1, L_2, L_3, L_4),
            5: (L_0, L_1, L_2, L_3, L_4, L_5),
        }

        return np.column_stack(bases[amount])

    def lsm_put_price(self, poly_amount = 2, OLS_form = 'Matrix', B = np.array([[]])):
        r = self.r
        data = self.S
        T = self.T
        N = self.N

        CFM = np.maximum(self.K - data, 0)
        CFM[:, 0] = 0
        A_CFM = CFM.copy()

        disc = np.exp(-r*(1/N))

        B_reg = np.zeros((poly_amount + 1,T*N))

        for i in range(T*N, 0, -1):
            ITM = CFM[:, i - 1] > 0
            if ITM.sum() == 0:
                continue
            Y = disc * A_CFM[ITM, i]
            X = data[ITM, i - 1]
            Z = self.lag_pol(X, amount=poly_amount)

            Q = self.lag_pol(data[:, i - 1], amount=poly_amount)

            if B.any():
                b = B[:,i-1]
            elif OLS_form == 'NumericRegression':
                b, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
                B_reg[:,i-1] = b
            elif OLS_form == 'Matrix':
                if np.linalg.matrix_rank(Z.T @ Z) == np.shape(Z.T @ Z)[0]:
                    b = np.linalg.solve(Z.T @ Z, Z.T @ Y)
                else:
                    b, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
                B_reg[:, i - 1] = b
            E = Q @ b

            CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
            CFM[CFM[:, i - 1] > 0, i:] = 0
            A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

        disc_vec = np.vander([disc], N * T + 1, increasing=True).T

        disc_CashFlows = CFM @ disc_vec

        price = np.mean(disc_CashFlows)

        return price, B_reg

    def BM(self, sigma, T=3, Omega=10000):
        N = self.N
        dt = 1 / N
        Z = np.random.normal(0, np.sqrt(dt), size=(Omega, T * N))
        x = np.zeros((Omega, T * N + 1))
        x[:, 0] = self.S0
        x[:, 1:] = self.S0 * np.exp(np.cumsum((self.r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1))
        return x

    def plot_hedge_error(self, data, insample = True):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(data, bins=30, density=True, alpha=0.4, edgecolor="white", label="Data")

        # KDE smoothing line
        kde_x = np.linspace(data.min() - 3 * std, data.max() + 3 * std, 300)
        kde_y = stats.gaussian_kde(data)(kde_x)
        ax.plot(kde_x, kde_y, linewidth=2.5, label="KDE")

        # Gaussian line
        gauss_y = stats.norm.pdf(kde_x, mean, std)
        ax.plot(kde_x, gauss_y, linewidth=2.5, linestyle="-.", color = '#36454F', label="Gaussian Fit")

        # Mean line
        ax.axvline(mean, color='black', linewidth=2, linestyle="--", label=f"Mean = {mean:.4f}")
        ax.plot([], [], color='black', linewidth=0, label=f"Std = {std:.4f}")

        if insample:
            ax.set_title('LSM Put Price  | In sample ', fontsize=15, fontweight="bold", pad=15)
        else:
            ax.set_title('LSM Put Price | Out of sample ', fontsize=15, fontweight="bold", pad=15)

        ax.set_xlabel("Error", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(left=-0,right=0.2)
        ax.set_ylim(bottom=0,top=14)

        plt.tight_layout()
        plt.legend()
        plt.savefig(self.filepath + rf'\price_estimate_insample_{insample}.png')
        plt.show()
        pass

    def sim_exp_pol_and_regtype(self):
        results = {}
        for form in ['Matrix', 'NumericRegression']:
            for poly_amount in [2, 3, 4, 5]:
                prices = []
                for i in range(10):
                    np.random.seed(i)
                    #self.S = self.BM(sigma=0.15,T=3,Omega=10000)
                    price, _ = self.lsm_put_price(poly_amount=poly_amount,OLS_form=form)
                    prices.append(price)

                results[(form, poly_amount)] = {
                    'mean': np.mean(prices),
                    'std': np.std(prices),
                    'prices': prices
                }

        return results

    def sim_exp_10000(self):
        prices = []
        self.N = 1
        for i in range(10000):
            np.random.seed(i)
            self.S = self.BM(sigma=0.15,T=3,Omega=8)

            price, _ = self.lsm_put_price(poly_amount=2, OLS_form = 'NumericRegression')
            prices.append(price)

            if (i + 1) % 1000 == 0:
                print(f'{(i + 1) / 100} pct done')
        return prices

    def sim_exp_10000_same_b(self):
        prices = []
        self.N = 1
        for i in range(10000):
            self.N = 1

            if i == 0:
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
                price, B_obs = self.lsm_put_price(poly_amount=2)
            else:
                np.random.seed(i)
                self.S = self.BM(sigma=0.15, T=3, Omega=8)
                price, _ = self.lsm_put_price(poly_amount=2, OLS_form = 'NumericRegression', B = B_obs)
            prices.append(price)

            if (i+1) % 1000 == 0:
                print(f'{(i + 1) / 100} pct done')
        return prices

if __name__ == '__main__':
    amr = AmericanOptions()
    amr.filepath = r'C:\Users\nicol\OneDrive - University of Copenhagen\Desktop\4 år\FinKont2\HandIn2'


    ## 2.a ##
    res, res2 = amr.volatility_ml(), amr.volatility_rv()

    ## 2.b ##
    res3 = []
    for sigma in [0.1484, 0.15, 0.1519]:
        for type in ['American', 'Bermudian', 'European']:
            price = amr.binomial_put_price(sigma=sigma, price_type = type)
            res3.append(price)
    prices_0 = np.array(res3)

    ## 2.c ##
    amr.S = np.array([
                [1.00, 1.09, 1.08, 1.34],
                [1.00, 1.16, 1.26, 1.54],
                [1.00, 1.22, 1.07, 1.03],
                [1.00, 0.93, 0.97, 0.92],
                [1.00, 1.11, 1.56, 1.52],
                [1.00, 0.76, 0.77, 0.90],
                [1.00, 0.92, 0.84, 1.01],
                [1.00, 0.88, 1.22, 1.34]
            ])
    amr.N = 1
    res4 = amr.sim_exp_pol_and_regtype()
    means = {form: {poly: res4[(form, poly)]['mean'] for poly in [2, 3, 4, 5]}
             for form in ['Matrix', 'NumericRegression']}

    stds = {form: {poly: res4[(form, poly)]['std'] for poly in [2, 3, 4, 5]}
            for form in ['Matrix', 'NumericRegression']}

    df_means = pd.DataFrame(means).T
    df_stds = pd.DataFrame(stds).T

    ## 2.d ##
    res5 = amr.sim_exp_10000()
    amr.plot_hedge_error(res5, insample = True)

    ## 2.e ##
    res6 = amr.sim_exp_10000_same_b()
    amr.plot_hedge_error(res6, insample = False)