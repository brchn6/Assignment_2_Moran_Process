(sexy_yeast_env) 12:39:26 💜 barc@cn441:~ > python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --help
usage: main.py [-h] [--population_size POPULATION_SIZE] [--initial_a_count INITIAL_A_COUNT]
               [--selection_coefficient SELECTION_COEFFICIENT] [--num_simulations NUM_SIMULATIONS]
               [--show_trajectories SHOW_TRAJECTORIES] [--results_path RESULTS_PATH]
               [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--parameter_sweep]
               [--mutation_rate MUTATION_RATE] [--env_fluctuation] [--env_amplitude ENV_AMPLITUDE]
               [--env_period ENV_PERIOD] [--env_type {sine,random,step}] [--run_extension_analysis]

Moran Process Simulation with Extensions

options:
  -h, --help            show this help message and exit
  --population_size POPULATION_SIZE
                        Size of the population
  --initial_a_count INITIAL_A_COUNT
                        Initial number of type A individuals
  --selection_coefficient SELECTION_COEFFICIENT
                        Selection coefficient for type A
  --num_simulations NUM_SIMULATIONS
                        Number of independent simulation runs
  --show_trajectories SHOW_TRAJECTORIES
                        Number of trajectories to plot
  --results_path RESULTS_PATH
                        Path to save results
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level
  --parameter_sweep     Run parameter sweep
  --mutation_rate MUTATION_RATE
                        Mutation rate (probability of type change during reproduction)
  --env_fluctuation     Enable environmental fluctuations
  --env_amplitude ENV_AMPLITUDE
                        Amplitude of environmental fluctuations
  --env_period ENV_PERIOD
                        Period of environmental fluctuations
  --env_type {sine,random,step}
                        Type of environmental fluctuation
  --run_extension_analysis
                        Run detailed analysis of extensions

                        

bsub -q short -R rusage[mem=50GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --population_size 1000 --initial_a_count 1  --selection_coefficient 0.01 --num_simulations 1000 --results_path  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results
bsub -q short -R rusage[mem=50GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --population_size 1000 --initial_a_count 50  --selection_coefficient 0.1 --num_simulations 1000 --results_path  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results
bsub -q short -R rusage[mem=50GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --parameter_sweep --num_simulations 1000 --results_path  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results
bsub -q gsla-cpu -n 4 -R span[hosts=1] -R rusage[mem=5GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --parameter_sweep --num_simulations 1 --results_path  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results
bsub -q gsla-cpu -n 4 -R span[hosts=1] -R rusage[mem=7GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --parameter_sweep --num_simulations 1000 --results_path  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results

bsub -q short -R rusage[mem=50GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --population_size 1000 --initial_a_count 1  --selection_coefficient '-0.1' --num_simulations 1000 --results_path  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results


python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py --population_size 100 --selection_coefficient 0.1 --mutation_rate 0.01 --results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/results 