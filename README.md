# hestonSim

Simulating stock data with Heston model

I followed the operations paper 'Exact Simulation of Stochastic Volatility and other Jump Diffusion Processes' by Broadie-Kaya published 2006 (http://www.columbia.edu/~mnb2/broadie/Assets/broadie_kaya_exact_sim_or_2006.pdf). I don't make efficient use of intermediate data structures like numpy arrays used in simulating the various integrals, if you wanted to simulate  >1000 rows of data, I recommend optimizing code so that it is more memory efficient. 

T is the variable telling you how many rows of price data to base the simulation on. You can simulate whichever price data column you like, just change the ohlc variable. The window variable will change how far the volatility samples look back in the data frame, so small window values will give simulations close to your actual stock, and large window values will imitate the stock less closely. I saved the project as a class so that interested parties could easily do little tests of my code in a terminal.

I use numpy, scipy, and pandas, and in the commented code at the very end I use matplotlib to visualize the trendline.
