import pandas as pd
import numpy as np
import scipy
from itertools import product
def SocialLearning(X, Y, learning_bias=0, parm=1):
    # This function simulates the social learning process between two individuals X and Y.
    # learning_bias: 0: content bias, 1: conformity bias, and 2: anticonformity bias
    # parm: strength of social learning bias; default value gives the unbiased case
    
    # Eitehr X or Y can be zero, which means that cultural evolution cannot change the phenotype
    if X==0:
        return 0
    elif Y==0:
        return 1
    
    if learning_bias == 0:  # content bias
        return (1+parm)*X/((1+parm)*X+Y)
    elif learning_bias == 1:  # conformity bias
        return (X**(1+parm))/(X**(1+parm) + Y**(1+parm))
    elif learning_bias == 2:  # anticonformity bias
        return (X**(-parm))/(X**(-parm) + Y**(-parm))
    else:
        raise ValueError("Invalid learning bias value. Use 0, 1, or 2.")    

def Cultural_MultipleRun(b_A, b_M=0.075, d=0.1, c =1e-3,  mu=1e-4, s=0.1, 
                         learning_bias=0, parm=1, iteration=1000,T_max=1e3, init_pop_size=1000):
    #　This function simulates the population dynamics of two phenotypes A and M under biological evolution using the Gillespie algorithm.
    #　b_A  is the birth rate of phenotype A, 
    #　b_M  is the birth rate of phenotype M
    # iteration is the number of simulation runs to estiamte the extinction probability
    #　d    is the natural death rate (common for both phenotypes)
    # c    is the death rate by competition common for both phenotypes)
    #　mu   is the innovation rate ( changing to the other phenotype through try-and-error
    #　s.   is the rate of of social learning
    #  learning_bias is the social bias 0: content bias, 1: conformity bias, and 2: anticonformity bias
    #  parm is the strength of social learning bias   
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
            birth_tot = b_A * N_A + b_M * N_M 
            death_A = d * N_A
            death_M = d * N_M
            competition_A = c * N_A* total_N
            competition_M = c * N_M * total_N
            ind_learning_A = mu * N_A
            ind_learning_M = mu * N_M
            soc_learning_A = s * N_A
            soc_learning_M = s * N_M

            rates = [birth_tot, death_A, death_M, competition_A, competition_M,
                    ind_learning_A, ind_learning_M, soc_learning_A, soc_learning_M]
            rate_sum = sum(rates)

            if rate_sum == 0:
                break
            # waiting time until the next event
            dt = np.random.exponential(1 / rate_sum)
            t += dt

            # event selection
            # note that all newborn individuals are assumed M under the cultural evolution
            r = np.random.uniform(0, rate_sum)
            if r < rates[0]:  # birth of M from A or M 
                    N_M += 1
            elif r < rates[0] + rates[1]:  # natural death of A
                N_A -= 1
            elif r < rates[0] + rates[1] + rates[2]:  # natural death of M
                N_M -= 1
            elif r < rates[0] + rates[1] + rates[2] + rates[3]:  # competition death of A
                N_A -= 1
            elif r < rates[0] + rates[1] + rates[2] + rates[3] + rates[4]:  # competition death of M
                N_M -= 1
            elif r < rates[0] + rates[1] + rates[2] + rates[3] + rates[4] + rates[5]:  # individual learning from A to M
                N_A -= 1
                N_M += 1
            elif r < rates[0] + rates[1] + rates[2] + rates[3] + rates[4] + rates[5] + rates[6]:  # individual learning from M to A
                N_M -= 1
                N_A += 1
            # write about the social learning from here
            #-----------------------
            elif r < rates[0] + rates[1] + rates[2] + rates[3] + rates[4] + rates[5] + rates[6] + rates[7]:  # social learning of A
                if SocialLearning(N_A, N_M, learning_bias, parm) < np.random.rand():
                    # individual with phenotype A changes to M through social learning
                    N_A -= 1
                    N_M += 1
                #otherwise, the individual with A socially learns A; nothing happens
            else:  # social learning of M
                if SocialLearning(N_A, N_M, learning_bias, parm) > np.random.rand():
                    # individual with phenotype M changes to A through social learning
                    N_M -= 1
                    N_A += 1
                 #otherwise, the individual with M socially learns M; nothing happens   
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
    # Cultural evolution model only with individaul learning
    # parameter lists
    b_A_list = [0.11, 0.12,  0.15, 0.2, 0.25, 0.3]  # d < b_A
    mu_list = [0, 1e-5, 1e-4, 1e-3, 1e-1] # individual learning rate
    c_list  = [3e-4, 1e-4, 3e-3, 1e-3]
    init_pop_size_list = [1e1, 1e2, 1e3]
    param_product = product(b_A_list, mu_list,    c_list,    init_pop_size_list)
    df_params = pd.DataFrame(param_product, columns=["b_A", "mu", "c", "init_pop_size"])
    for i in range(len(df_params)):
        b_A = df_params.loc[i, "b_A"]
        mu = df_params.loc[i, "mu"]
        c = df_params.loc[i, "c"]
        init_pop_size = int(df_params.loc[i, "init_pop_size"])
        extinction, mean_fraction, mean_rescue_time, var_fraction, var_rescue_time = Cultural_MultipleRun(b_A=b_A, b_M=b_M, d=0.1, c=c,
                                                                                                          s=0, learning_bias=0, parm=0,
                                                                                                          mu=mu, iteration=iteration, T_max=T_max, init_pop_size=init_pop_size)
        df_params.loc[i, "extinction"] = extinction
        df_params.loc[i, "rescue"] = iteration-extinction
        df_params.loc[i, "mean_fraction"] = mean_fraction
        df_params.loc[i, "mean_rescue_time"] = mean_rescue_time
        df_params.loc[i, "var_fraction"] = var_fraction
        df_params.loc[i, "var_rescue_time"] = var_rescue_time
    df_params.to_csv('Cultural_evolutionary_rescue_individual.csv', index=False)