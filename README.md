# Biological and Cultural and Evolutionary Rescue models

This repository provides the source code for "Comparison of evolutionary rescue via biological and cultural evolution." 

The manuscript shows that transmission bias alters the probability of cultural evolutionary rescue. In particular, anticonformity-biased transmission sometimes more effectively rescues populations than biological evolution through the faster establishment of initially rare mutants.

Python and key library versions are as follows:

. Python version: 3.11.12 (main, Apr  8 2025, 14:15:29)
. pandas 2.2.3
. numpy 2.2.6
. scipy 1.15.3
. statsmodels 0.14.4
. sklearn 1.9.0

1. Simulation codes and associated results csv files,
To implement the eco-evolutionary dynamics, run the following scripts:

Demographic_maladaptive.py: This script runs the population dynamics of maladaptive individuals without any evolutionary processes. The results are saved as "BaselinModel.csv" and "BaselineMpodel_Trajectories.csv."

Demographic_adaptive.py: This script runs the population dynamics of adaptive individuals without any evolutionary processes. The results are saved as "BaselineModel_adaptive.csv."

Bio_Evo_Rescue.py: This scripts simualtes the eco-evolutionary dynamics under biological evolution. The results are saved as "Biological_evolutionary_rescue.csv."

Cul_Evo_Rescue_individual.py: This script simulates eco-evolutionary dynamics in which individual learning alone can affect individuals' traits.  The results are saved as "Cultural_evolutionary_rescue_individual.csv."

Cul_Evo_Rescue_parallel.py: This script runs cultural evolution and population dynamics models in parallel under content-, conformity-, and anticonformity-biased social learning. The results are saved as "Cultural_evolutionary_rescue_content.csv", "Cultural_evolutionary_rescue_conformity.csv", and "Cultural_evolutionary_rescu_anticonformity.csv"

2. Statistical analyses and visualisation
   Analysis.ipynb provides code to reproduce the figures and tables in the associated manuscript.
   

