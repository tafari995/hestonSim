
import numpy as np
import pandas as pd 
from scipy import special

import matplotlib.pyplot as plt

class HesSim:
    def __init__(self,filepath,rho,kappa,theta,sigma,T):
        self.data=filepath  #wants raw string
        self.rho=rho
        self.kappa=kappa
        self.theta=theta
        self.sigma=sigma
        self.length=T
        
    def rS0V_getFromData(self,ohlc,window):
        dt=pd.read_csv(self.data, parse_dates=True)
        dt.dropna()
        price_stream=dt[ohlc]
        sampl=price_stream[(price_stream.shape[0]-self.length)-window:]
        r=(1/252.0)*price_stream.diff().mean()
        S0=price_stream[price_stream.shape[0]-self.length]
        vol_sampl=(1/sampl**2)*sampl.rolling(window=window).std() #feel free to play with this 1/S0**2 constant
        
        return (r,S0,vol_sampl)
        
    
    def characteristic_function(self, a:float, V_u:float, V_t:float, t:int, u:int) -> float:
        d = 4 * self.kappa * self.theta / (self.sigma**2)
        gamma_a = np.sqrt(self.kappa**2 - 2 * self.sigma**2 * 1j * a)
        
        term1 = np.exp(-0.5 * (gamma_a - self.kappa) * (t - u))
        term2 = (1 - np.exp(-self.kappa * (t - u))) / (1 - np.exp(-gamma_a * (t - u)))
        
        exp_term = np.exp((V_u + V_t) / 2 * (
            (1 + np.exp(-self.kappa * (t - u))) / (1 - np.exp(-self.kappa * (t - u))) -
            gamma_a * (1 + np.exp(-gamma_a * (t - u))) / (1 - np.exp(-gamma_a * (t - u)))
        ))
        
        bessel_num = special.iv(0.5 * d - 1, np.sqrt(V_u * V_t * 4 * gamma_a * np.exp(-0.5 * gamma_a * (t - u))) / (self.sigma**2 * (1 - np.exp(-gamma_a * (t - u)))))
        bessel_den = special.iv(0.5 * d - 1, np.sqrt(V_u * V_t * 4 * np.exp(-0.5 * self.kappa * (t - u))) / (self.sigma**2 * (1 - np.exp(-self.kappa * (t - u)))))
        
        return term1 * term2 * exp_term * (bessel_num / bessel_den)
    
    def integrand(self,j:float,h:float,x:float,V_u:float,V_t:float,t:int,u:int) -> float:
        ch_func=self.characteristic_function(j * h, V_u, V_t, t, u)
        return np.sin(j * h * x) / j * np.real(ch_func)
        
    #Pr(V(u,t)<x) approximation with trapezoidal rule
    def probability_distribution(self,x:float, V_u:float, V_t:float, t:int, u:int, h:float, N:int) -> float:   
        terms = np.array([self.integrand(j,h,x,V_u,V_t,t,u) for j in range(1, N+1)])
        return h * x / np.pi + 2 * h / np.pi * np.sum(terms)
        

    def inverse_cdf(self,U:float,V_u:float, V_t:float, t:int, u:int) -> float:
        # Initial guess using normal approximation
        mean = self.theta * (t - u) + (V_u - self.theta) * (1 - np.exp(-self.kappa * (t - u))) / self.kappa
        var = self.sigma**2 * self.theta * (t - u) / (2 * self.kappa) + \
              self.sigma**2 * (V_u - self.theta) * (1 - np.exp(-self.kappa * (t - u))) / (self.kappa**2) + \
              self.sigma**2 * (V_u - self.theta)**2 * (1 - np.exp(-2 * self.kappa * (t - u))) / (4 * self.kappa**3)
        x = max(0.01 * mean, np.random.normal(mean, np.sqrt(var)))
        
        # Newton's method
        h, N = 5e-3, 20  
        for _ in range(5):  # Max iterations
            F = self.probability_distribution(x, V_u, V_t, t, u, h, N)
            if abs(F - U) < 1e-5:
                return x
            dF = (self.probability_distribution(x + 1e-5, V_u, V_t, t, u, h, N) - F) / 1e-5
            d2F = (self.probability_distribution(x + 2e-5, V_u, V_t, t, u, h, N) - 
                   2 * F + 
                   self.probability_distribution(x - 1e-5, V_u, V_t, t, u, h, N)) / (1e-5**2)
            x = x - dF / d2F * (1 - np.sqrt(1 - 2 * (F - U) * d2F / dF**2))
        
        # Fallback to bisection if Newton's method fails
        a, b = 0, 10 * mean
        while b - a > 1e-5:
            x = (a + b) / 2
            if self.probability_distribution(x, V_u, V_t, t, u, h, N) < U:
                a = x
            else:
                b = x
        return x
        
    def simulate_integral(self, V_u:float, V_t:float, t:int, u:int, m:int):
        rng=np.random.default_rng()
        if m==1:
            U=rng.uniform()
            integral_sample=self.inverse_cdf(U,V_u, V_t, t, u)
            return integral_sample    
        U = rng.uniform(size=m)
        integral_samples = np.array([self.inverse_cdf(u,V_u, V_t, t, u) for u in U])    
        return integral_samples
        
    def runsim(self,ohlc,window):
        r, S0, V_sample =self.rS0V_getFromData(ohlc,window)
        V_sample=V_sample[V_sample.shape[0]-self.length:].to_numpy()
        S_sample=np.zeros(shape=self.length,dtype=float)
        integral_Vds=np.zeros(shape=self.length,dtype=float)
        integral_rootVdW=np.zeros(shape=self.length,dtype=float)
        
        S_sample[0]+=S0    
        deg_free=(4.0*self.theta*self.kappa)/(self.sigma**2)
        rng=np.random.default_rng()
        for j in range(1,self.length):
            V_u, V_t = V_sample[j-1], V_sample[j]
            integral_Vds[j]+=self.simulate_integral(V_u, V_t, j, j-1,1)
            integral_rootVdW[j]+=(1/self.sigma)*(V_t-V_u-self.kappa*self.theta+self.kappa*integral_Vds[j])
            mu_u_t=np.log(S_sample[j-1])+r-(0.5*integral_Vds[j])+self.rho*integral_rootVdW[j] 
            sigma_u_t=np.sqrt((1-self.rho**2)*integral_Vds[j])                
            Z=rng.standard_normal()
            S_sample[j]+=np.exp(mu_u_t+sigma_u_t*Z)           
        return (S_sample,V_sample)


filepath=r"/home/tafari/Documents/Python_Projects/stock_repo/stock_trader/stock_data/SPX.csv" #your filepath to csv data goes in quotes

#I found these parameters good for simulating S&P500 data
rho=5.5e-3
kappa=2.27
theta=4.0e-5
sigma=0.204
#you completely control the length of the simulation :)
T=150

#uncomment to plot simulated stock prices, it looks very realistic!
'''
simulation1=HesSim(filepath,rho,kappa,theta,sigma,T)
simulated_prices=simulation1.runsim('Open',simulation1.length//3)[0]
plt.plot(simulated_prices, label='Simulated Stock')
plt.legend()
plt.show()
'''







