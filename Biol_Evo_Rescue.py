import pandas as pd
import numpy as np
import scipy
from itertools import product
def Biological_MultipleRun(b_A, b_M=0.075, d=0.1, c =1e-3,  mu=1e-4, iteration=1000,T_max=1e3, init_pop_size=1000):
    #　This function simulates the population dynamics of two phenotypes A and M under biological evolution using the Gillespie algorithm.
    #　b_A  is the birth rate of phenotype A, 
    #　b_M  is the birth rate of phenotype M
    # iteration is the number of simulation runs to estiamte the extinction probability
    #　d    is the natural death rate (common for both phenotypes)
    #　mu   is the mutation rate (probability of changing phenotype at birth)
    #　c    is the death rate through the competition
    #　T_max is the maximum simulation time 
    
    extinction_count=0    
    fraction_A=[]
    rescue_time=[]
    for i in range(iteration):
        #Set initial condition
        N_A = 1
        N_M = init_pop_size-N_A
        flag=0
        t = 0
        times = [t]
        pop_A = [N_A]
        pop_M = [N_M]
        pop_total = [N_A + N_M]
        save_time=np.nan


        # Gillespie algotithm loop
        # The Gillespie algorithm is a stochastic simulation algorithm used to model the time evolution of systems
        # with discrete events, such as birth and death processes in populations.
        while t < T_max and (N_A + N_M) > 0:
            total_N = N_A + N_M
            birth_A = b_A * N_A 
            birth_M = b_M * N_M 
            death_A = d * N_A
            death_M = d * N_M
            competition_A = c * N_A* total_N
            competition_M = c * N_M * total_N

            rates = [birth_A, birth_M, death_A, death_M, competition_A, competition_M]
            rate_sum = sum(rates)

            if rate_sum == 0:
                break
            # waiting time until the next event
            dt = np.random.exponential(1 / rate_sum)
            t += dt

            # event selection
            r = np.random.uniform(0, rate_sum)
            if r < rates[0]:  # birth of A
                if np.random.rand() < mu:
                    N_M += 1  # mutation to M
                else:
                    N_A += 1
            elif r < rates[0] + rates[1]:  # birth of M
                if np.random.rand() < mu:
                    N_A += 1  # mutation to A
                else:
                    N_M += 1
            elif r < rates[0] + rates[1] + rates[2]:  # natural death of A
                N_A -= 1
            elif r < rates[0] + rates[1] + rates[2] + rates[3]:  # natural death of M
                N_M -= 1
            elif r < rates[0] + rates[1] + rates[2] + rates[3] + rates[4]:  # competition death of A
                N_A -= 1
            else:  # competition death of M
                N_M -= 1
            if N_A>(b_A - d) /c * 0.1 and flag==0:
                flag=1
                save_time=t

        if N_A+N_M==0:
             extinction_count+=1
        else:
             # save the final fraction of A in the population and the time to rescue (time until A reaches a certain threshold, e.g., 10% of the carrying capacity)
             fraction_A.append(N_A/(N_A+N_M))
        if flag==1:
             rescue_time.append(save_time)

    mean_fraction = np.mean(fraction_A) if fraction_A else 0
    mean_rescue_time = np.nanmean(rescue_time) if rescue_time else 0
    var_fraction = np.var(fraction_A) if fraction_A else 0
    var_rescue_time = np.nanvar(rescue_time) if rescue_time else 0
    return extinction_count, mean_fraction, mean_rescue_time, var_fraction, var_rescue_time 
if __name__ == "__main__":
    # eco-evolutionary dynamics with genetic changes
    # parameter lists
    b_A_list = [0.11, 0.12,  0.15, 0.2, 0.25, 0.3]  # d < b_A
    mu_list = [0, 1e-5, 1e-4, 1e-3]
    c_list  = [3e-4, 1e-4, 3e-3, 1e-3]
    init_pop_size_list = [1e1, 1e2, 1e3]
    param_product = product(b_A_list, mu_list,    c_list,    init_pop_size_list)
    df_params = pd.DataFrame(param_product, columns=["b_A", "mu", "c", "init_pop_size"])
    for i in range(len(df_params)):
        b_A = df_params.loc[i, "b_A"]
        mu = df_params.loc[i, "mu"]
        c = df_params.loc[i, "c"]
        init_pop_size = int(df_params.loc[i, "init_pop_size"])
        extinction, mean_fraction, mean_rescue_time, var_fraction, var_rescue_time = Biological_MultipleRun(b_A=b_A, b_M=b_M, d=0.1, c=c, mu=mu, iteration=iteration, T_max=T_max, init_pop_size=init_pop_size)
        df_params.loc[i, "extinction"] = extinction
        df_params.loc[i, "rescue"] = iteration-extinction
        df_params.loc[i, "mean_fraction"] = mean_fraction
        df_params.loc[i, "mean_rescue_time"] = mean_rescue_time
        df_params.loc[i, "var_fraction"] = var_fraction
        df_params.loc[i, "var_rescue_time"] = var_rescue_time
    df_params.to_csv('Biological_evolutionary_rescue.csv', index=False)