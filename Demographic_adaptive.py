from itertools import product
import math
import numpy as np

def Baseline_Model2(b_A=0.11, b_M=0.075, d=0.1, c =1e-3,  mu=0, iteration=1000,
                     init_fraction=0.1, T_max=1e3,  trajectory=False):
    # This function simulates the population dynamics of two phenotypes A and M under biological evolution using the Gillespie algorithm.
    # b_A  is the birth rate of phenotype A,
    # b_M  is the birth rate of phenotype M
    # iteration is the number of simulation runs to estimate the extinction probability
    # d    is the death rate (common for both phenotypes)
    # mu   is the mutation rate (probability of changing phenotype at birth)
    # K    is the environmental carrying capacity
    # T_max is the maximum simulation time

    extinction_count = 0
    fraction_A = []
    rescue_time = []
    if trajectory:
        all_records = []
    for i in range(iteration):
        # Set initial condition
        K = (b_A-d)/c
        N_A = math.ceil(init_fraction * K)
        N_M = 0
        flag = 0
        t = 0
        times = [t]
        pop_A = [N_A]
        pop_M = [N_M]
        pop_total = [N_A + N_M]

        # Gillespie algorithm loop
        # The Gillespie algorithm is a stochastic simulation algorithm used to model the time evolution of systems
        # with discrete events, such as birth and death processes in populations.
        if trajectory:
                all_records.append({
                    "traj_id": i,           
                    "time": t,            
                     "pop_total": N_A+N_M })
        while t < T_max and (N_A + N_M) > 0:    
            total_N = N_A + N_M
            birth_A = b_A * N_A
            birth_M = b_M * N_M
            death_A = d * N_A
            death_M = d * N_M
            competition_A = c * N_A * total_N
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
            if trajectory:
                all_records.append({
                    "traj_id": i,
                    "time": t,
                    "pop_total": N_A + N_M
                })

        if N_A + N_M == 0:
            extinction_count += 1
    if trajectory:
        df_traj = pd.DataFrame(all_records)
       #fig, ax = plt.subplots(figsize=(6, 4))
       # for traj_id, df_one in df_traj.groupby("traj_id"):
       #     ax.plot(
        #        df_one["time"],
        #       df_one["pop_total"],
        #       color="grey",
        #       alpha=0.01,      
        #        linewidth=0.5)
        #    ax.set_xlabel("Time")
        #    ax.set_ylabel("Population size")
        return extinction, df_traj
    else:
        return extinction_count
if __name__ == "__main__":
    # Run the population dynamics of adaptive individuals without evolution
    # parameter lists   
    iteration=1000
    init_fraction= [0.1, 1]
    c_list  = [1e-4, 3e-4, 1e-3, 3e-3]
    b_A_list = [0.11, 0.12,  0.15, 0.2, 0.25, 0.3]
    param_product = product(b_A_list,   c_list, init_fraction)
    df_params = pd.DataFrame(param_product, columns=["b_A", "c", "init_fraction"])
    for i in range(len(df_params)):
        b_A=df_params.loc[i, "b_A"]
        c = df_params.loc[i, "c"]
        init_fraction=df_params.loc[i, "init_fraction"]
        extinction= Baseline_Model2(b_A=b_A,b_M=0.075, d=0.1, c=c, iteration=iteration, 
                                                init_fraction=init_fraction, T_max=1e3,  trajectory=False)
        df_params.loc[i, "extinction"] = extinction
        df_params.loc[i, "rescue"] = iteration-extinction
    df_params.to_csv('BaselineModel_adaptive.csv', index=False)