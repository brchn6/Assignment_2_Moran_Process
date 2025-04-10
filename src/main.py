"""
this script simulates the Moran process for a population of individuals
with two types (A and B) and calculates the fixation probability of type A.

path : /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_2_Moran_Process/src/main.py

Author: BC 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import logging
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm


#######################################################################
# Setup logging
#######################################################################
def init_log(results_path, log_level="INFO"):
    """Initialize logging with specified log level"""
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Initialize logging
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    
    # Add file handler
    path = os.path.join(results_path, 'moran_process_simulation.log')
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(level)
    logging.getLogger().addHandler(fh)

    logging.info("Logging initialized successfully.")
    logging.info(f"Results will be saved to: {results_path}")

    return logging, results_path

#######################################################################
# Time tracking decorator
#######################################################################
def timing_decorator(func):
    """Decorator to measure and log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # If the result is already a tuple, we need to handle it properly
        if isinstance(result, tuple):
            return result  # Return the original tuple without wrapping it
        else:
            return result, elapsed_time  # Wrap single result with elapsed time
    return wrapper

#######################################################################
# Classes
#######################################################################
class Population:
    def __init__(self, size, initial_a_count, selection_coefficient):
        """
        Initialize a population for Moran Process simulation
        
        Parameters:
        - size: Total population size
        - initial_a_count: Initial number of type A individuals
        - selection_coefficient: Selection advantage of type A over type B
        """
        self.size = size
        self.selection_coefficient = selection_coefficient
        
        # Initialize population state: 1 for type A, 0 for type B
        self.population = np.zeros(size, dtype=int)
        self.population[:initial_a_count] = 1
        np.random.shuffle(self.population)
        
        # Track current count of type A
        self.type_a_count = initial_a_count
        
    def get_a_frequency(self):
        """Return the current frequency of type A in the population"""
        return self.type_a_count / self.size
    
    def step(self):
        """
        Perform one step of the Moran process:
        1. Select an individual for reproduction based on fitness
        2. Create offspring
        3. Select random individual for death
        4. Replace dead with offspring
        
        Returns:
        - True if population changed, False otherwise
        """
        # Calculate fitness-weighted selection probabilities
        fitness = np.ones(self.size)
        fitness[self.population == 1] = 1 + self.selection_coefficient
        fitness_sum = np.sum(fitness)
        selection_probs = fitness / fitness_sum
        
        # Select individual for reproduction
        reproducing_idx = np.random.choice(self.size, p=selection_probs)
        reproducing_type = self.population[reproducing_idx]
        
        # Select random individual for death
        dying_idx = np.random.choice(self.size)
        dying_type = self.population[dying_idx]
        
        # Update population if there's a change
        if reproducing_type != dying_type:
            self.population[dying_idx] = reproducing_type
            self.type_a_count += (1 if reproducing_type == 1 else -1)
            return True
        
        return False
    
    def is_fixed(self):
        """Return True if the population is fixed (all type A or all type B)"""
        return self.type_a_count == 0 or self.type_a_count == self.size
    
    def is_type_a_fixed(self):
        """Return True if type A has fixed in the population"""
        return self.type_a_count == self.size

#######################################################################
# Simulation functions
#######################################################################
@timing_decorator
def run_single_simulation(population_size, initial_a_count, selection_coefficient, max_steps=None):
    """
    Run a single simulation of the Moran process until fixation
    
    Parameters:
    - population_size: Size of the population
    - initial_a_count: Initial number of type A individuals
    - selection_coefficient: Selective advantage of type A
    - max_steps: Maximum number of steps (default: None for unlimited)
    
    Returns:
    - is_a_fixed: Whether type A fixed in the population
    - steps: Number of steps until fixation
    - trajectory: List of type A frequencies at each step
    """
    pop = Population(population_size, initial_a_count, selection_coefficient)
    
    trajectory = [pop.get_a_frequency()]
    steps = 0
    
    while not pop.is_fixed():
        pop.step()
        steps += 1
        trajectory.append(pop.get_a_frequency())
        
        if max_steps is not None and steps >= max_steps:
            break
    
    return pop.is_type_a_fixed(), steps, trajectory

@timing_decorator
def calculate_fixation_probability(population_size, initial_a_count, selection_coefficient, num_simulations=100):
    """
    Calculate the empirical fixation probability of type A by running multiple simulations
    
    Parameters:
    - population_size: Size of the population
    - initial_a_count: Initial number of type A individuals
    - selection_coefficient: Selective advantage of type A
    - num_simulations: Number of independent simulation runs
    
    Returns:
    - fixation_probability: Empirical probability of type A fixation
    - mean_fixation_time: Mean time to fixation across simulations
    """
    fixed_count = 0
    fixation_times = []
    
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        is_a_fixed, steps, _ = run_single_simulation(
            population_size, initial_a_count, selection_coefficient
        )
        
        if is_a_fixed:
            fixed_count += 1
        
        fixation_times.append(steps)
    
    fixation_probability = fixed_count / num_simulations
    mean_fixation_time = np.mean(fixation_times)
    
    return fixation_probability, mean_fixation_time

def theoretical_fixation_probability(population_size, initial_a_count, selection_coefficient):
    """
    Calculate the theoretical fixation probability for type A
    """
    if selection_coefficient == 0:  # Neutral selection
        return initial_a_count / population_size
    
    s = selection_coefficient
    N = population_size
    i = initial_a_count
    
    # For any selection coefficient and initial count
    # The general formula works for both advantageous and deleterious mutations
    if s != 0:
        r = 1 + s  # relative fitness
        numerator = 1 - (1/r)**i
        denominator = 1 - (1/r)**N
        return numerator / denominator
    else:
        return i / N

@timing_decorator
def run_parameter_sweep(results_path, args):
    """
    Run simulations with multiple parameter combinations and analyze the impact of each parameter
    
    Parameters:
    - results_path: Path to save results
    - args: Command line arguments
    """
    logging.info("Starting parameter sweep analysis...")
    
    # Define parameter ranges to test
    population_sizes = [10, 50, 100, 500, 1000]
    selection_coefficients = [-0.5, -0.1, 0, 0.1, 0.5, 1.0]
    initial_a_counts = [1, 5, 10, 25, 50]  # As absolute counts
    initial_a_frequencies = []  # Will be calculated based on population size
    
    # Track results
    results = []
    
    # Total number of parameter combinations
    total_combinations = len(population_sizes) * len(selection_coefficients) * len(initial_a_counts)
    logging.info(f"Will test {total_combinations} parameter combinations")
    
    # Progress counter
    current_combination = 0
    
    # Create results DataFrame structure
    for pop_size in population_sizes:
        for sel_coef in selection_coefficients:
            for init_a in initial_a_counts:
                # Skip invalid combinations (initial_a > population_size)
                if init_a > pop_size:
                    continue
                
                # Calculate initial frequency
                init_freq = init_a / pop_size
                
                # Update progress
                current_combination += 1
                logging.info(f"Testing combination {current_combination}/{total_combinations}: "
                           f"N={pop_size}, s={sel_coef}, initial_a={init_a} ({init_freq:.2f})")
                
                # Calculate theoretical fixation probability
                theo_prob = theoretical_fixation_probability(pop_size, init_a, sel_coef)
                
                # Run fewer simulations for large populations to save time
                n_sims = min(args.num_simulations, 
                             max(50, int(args.num_simulations / (pop_size / 100))))
                
                # Run simulations
                try:
                    start_time = time.time()
                    fix_prob, mean_time = calculate_fixation_probability(
                        pop_size, init_a, sel_coef, n_sims
                    )
                    total_time = time.time() - start_time
                    
                    # Record results
                    results.append({
                        'population_size': pop_size,
                        'selection_coefficient': sel_coef,
                        'initial_a_count': init_a,
                        'initial_a_frequency': init_freq,
                        'theoretical_fixation_prob': theo_prob,
                        'empirical_fixation_prob': fix_prob,
                        'mean_fixation_time': mean_time,
                        'simulation_time': total_time,
                        'num_simulations': n_sims
                    })
                    
                    logging.info(f"  Results: Fixation Probability = {fix_prob:.4f}, "
                               f"Mean Fixation Time = {mean_time:.1f}, "
                               f"Simulation Time = {total_time:.1f}s")
                    
                except Exception as e:
                    logging.error(f"Error in simulation with N={pop_size}, s={sel_coef}, initial_a={init_a}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    csv_path = os.path.join(results_path, "parameter_sweep_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Raw results saved to {csv_path}")
    
    # Analyze and visualize results
    analyze_parameter_sweep(results_df, results_path)
    
    return results_df

def analyze_parameter_sweep(results_df, results_path):
    """
    Analyze the parameter sweep results to understand the impact of each parameter

    Parameters:
    - results_df: DataFrame containing simulation results
    - results_path: Path to save analysis results
    """
    logging.info("Analyzing parameter sweep results...")

    # 1. Create summary table
    summary = results_df.groupby(['population_size', 'selection_coefficient']).agg({
        'empirical_fixation_prob': ['mean', 'std'],
        'mean_fixation_time': ['mean', 'std'],
        'simulation_time': ['mean', 'sum']
    }).reset_index()

    # Save summary table
    summary_path = os.path.join(results_path, "parameter_sweep_summary.csv")
    summary.to_csv(summary_path)
    logging.info(f"Summary table saved to {summary_path}")

    # 2. Visualization of parameter effects
    try:
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        viz_path = os.path.join(results_path, "visualizations")
        os.makedirs(viz_path, exist_ok=True)

        # 2.1 Effect of selection coefficient on fixation probability
        plt.figure()
        for pop_size in results_df['population_size'].unique():
            subset = results_df[results_df['population_size'] == pop_size]
            sns.lineplot(data=subset, x='selection_coefficient', y='empirical_fixation_prob', 
                         marker='o', label=f"N={pop_size}")
        plt.xlabel("Selection Coefficient (s)")
        plt.ylabel("Fixation Probability")
        plt.title("Effect of Selection Coefficient on Fixation Probability")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "selection_effect_on_fixation.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.2 Effect of population size on fixation probability
        plt.figure()
        for s in results_df['selection_coefficient'].unique():
            subset = results_df[results_df['selection_coefficient'] == s]
            sns.lineplot(data=subset, x='population_size', y='empirical_fixation_prob', 
                         marker='o', label=f"s={s}")
        plt.xscale('log')
        plt.xlabel("Population Size (log scale)")
        plt.ylabel("Fixation Probability")
        plt.title("Effect of Population Size on Fixation Probability")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "population_size_effect_on_fixation.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.3 Effect of initial frequency on fixation probability
        plt.figure()
        for s in results_df['selection_coefficient'].unique():
            subset = results_df[results_df['selection_coefficient'] == s]
            sns.lineplot(data=subset, x='initial_a_frequency', y='empirical_fixation_prob', 
                         marker='o', label=f"s={s}")
        plt.xlabel("Initial Frequency of Type A")
        plt.ylabel("Fixation Probability")
        plt.title("Effect of Initial Frequency on Fixation Probability")
        plt.grid(True, alpha=0.3)
        x = np.linspace(0, 1, 100)
        plt.plot(x, x, 'k--', alpha=0.7, label="Neutral Drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "initial_frequency_effect_on_fixation.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.4 Effect on fixation time
        plt.figure()
        for s in results_df['selection_coefficient'].unique():
            subset = results_df[results_df['selection_coefficient'] == s]
            sns.lineplot(data=subset, x='population_size', y='mean_fixation_time', 
                         marker='o', label=f"s={s}")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Population Size (log scale)")
        plt.ylabel("Mean Fixation Time (log scale)")
        plt.title("Effect of Population Size and Selection Coefficient on Fixation Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "parameter_effects_on_fixation_time.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.5 Computational Performance
        plt.figure()
        sns.lineplot(data=results_df, x='population_size', y='simulation_time', marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Population Size (log scale)")
        plt.ylabel("Simulation Time (seconds, log scale)")
        plt.title("Computational Performance: Effect of Population Size on Simulation Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "computational_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.6 Heatmap of fixation probabilities
        heatmap_data = results_df.pivot_table(
            values='empirical_fixation_prob',
            index='population_size',
            columns='selection_coefficient'
        )
        plt.figure()
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", 
                    cbar_kws={'label': 'Fixation Probability'})
        plt.title("Fixation Probability Heatmap")
        plt.xlabel("Selection Coefficient (s)")
        plt.ylabel("Population Size (N)")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "fixation_probability_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.7 Correlation matrix
        corr_data = results_df[['population_size', 'selection_coefficient', 'initial_a_frequency',
                                'empirical_fixation_prob', 'mean_fixation_time', 'simulation_time']]
        corr_matrix = corr_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    cbar_kws={'label': 'Correlation Coefficient'})
        plt.title("Correlation Between Parameters and Outcomes")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, "parameter_correlation_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Visualizations saved to {viz_path}")

    except ImportError as e:
        logging.warning(f"Could not create visualizations: {e}")
    except Exception as e:
        logging.error(f"Error in visualization: {e}")
        import traceback
        logging.error(traceback.format_exc())

    # 3. Generate summary report
    create_parameter_sweep_report(results_df, results_path)

def create_parameter_sweep_report(results_df, results_path):
    """
    Create a text report summarizing the parameter sweep findings
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - results_path: Path to save the report
    """
    report_path = os.path.join(results_path, "parameter_sweep_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Moran Process: Parameter Sweep Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        # Write introduction
        f.write("This report summarizes the effects of different parameters on the Moran Process simulation.\n\n")
        
        # Summary statistics
        f.write("-"*80 + "\n")
        f.write("Summary Statistics:\n")
        f.write("-"*80 + "\n")
        
        f.write(f"Total parameter combinations tested: {len(results_df)}\n")
        f.write(f"Population sizes tested: {sorted(results_df['population_size'].unique())}\n")
        f.write(f"Selection coefficients tested: {sorted(results_df['selection_coefficient'].unique())}\n")
        f.write(f"Initial type A counts tested: {sorted(results_df['initial_a_count'].unique())}\n\n")
        
        # Effect of selection coefficient
        f.write("-"*80 + "\n")
        f.write("Effect of Selection Coefficient:\n")
        f.write("-"*80 + "\n")
        
        # Group by selection coefficient and calculate mean fixation probability
        s_effect = results_df.groupby('selection_coefficient')['empirical_fixation_prob'].mean().reset_index()
        for _, row in s_effect.iterrows():
            f.write(f"s = {row['selection_coefficient']}: Mean fixation probability = {row['empirical_fixation_prob']:.4f}\n")
        
        # Explanation of the trend
        f.write("\nAnalysis: ")
        if s_effect['selection_coefficient'].corr(s_effect['empirical_fixation_prob']) > 0:
            f.write("As expected, higher selection coefficients lead to higher fixation probabilities. ")
            f.write("This confirms that stronger selection pressure increases the chance of advantageous alleles fixing.\n\n")
        else:
            f.write("The relationship between selection coefficient and fixation probability is not as expected. ")
            f.write("This may be due to stochastic effects or limited sample size.\n\n")
        
        # Effect of population size
        f.write("-"*80 + "\n")
        f.write("Effect of Population Size:\n")
        f.write("-"*80 + "\n")
        
        # Group by population size and selection coefficient
        pop_effect = results_df.groupby(['population_size', 'selection_coefficient'])['empirical_fixation_prob'].mean().reset_index()
        
        # For each selection coefficient, report trend with population size
        for s in sorted(pop_effect['selection_coefficient'].unique()):
            s_data = pop_effect[pop_effect['selection_coefficient'] == s]
            f.write(f"For s = {s}:\n")
            
            # Calculate correlation
            corr = s_data['population_size'].corr(s_data['empirical_fixation_prob'])
            
            # Report trend
            f.write(f"  Correlation between population size and fixation probability: {corr:.4f}\n")
            
            if s > 0:
                if corr > 0:
                    f.write("  As expected for s > 0, larger populations show higher fixation probabilities.\n")
                    f.write("  This is because selection becomes more effective relative to drift in larger populations.\n")
                else:
                    f.write("  Unexpected trend: For s > 0, larger populations should show higher fixation probabilities.\n")
            elif s < 0:
                if corr < 0:
                    f.write("  As expected for s < 0, larger populations show lower fixation probabilities.\n")
                    f.write("  This is because selection more effectively removes deleterious mutations in larger populations.\n")
                else:
                    f.write("  Unexpected trend: For s < 0, larger populations should show lower fixation probabilities.\n")
            else:  # s == 0
                if abs(corr) < 0.1:
                    f.write("  As expected for s = 0 (neutral evolution), population size has little effect on fixation probability.\n")
                    f.write("  Under neutrality, fixation probability should approximately equal initial frequency.\n")
                else:
                    f.write("  Unexpected trend: For s = 0, population size should have little effect on fixation probability.\n")
            
            f.write("\n")
        
        # Effect on fixation time
        f.write("-"*80 + "\n")
        f.write("Effect on Fixation Time:\n")
        f.write("-"*80 + "\n")
        
        # Group by population size and calculate mean fixation time
        time_effect = results_df.groupby('population_size')['mean_fixation_time'].mean().reset_index()
        
        # Report relationship
        corr_time = time_effect['population_size'].corr(time_effect['mean_fixation_time'])
        f.write(f"Correlation between population size and fixation time: {corr_time:.4f}\n\n")
        
        if corr_time > 0:
            f.write("As expected, larger populations take longer to reach fixation.\n")
            f.write("This is because more replacement events are needed for an allele to sweep through a larger population.\n\n")
        else:
            f.write("Unexpected trend: Larger populations should take longer to reach fixation.\n\n")
        
        # Computational performance
        f.write("-"*80 + "\n")
        f.write("Computational Performance:\n")
        f.write("-"*80 + "\n")
        
        # Group by population size and calculate mean simulation time
        perf_data = results_df.groupby('population_size')['simulation_time'].mean().reset_index()
        for _, row in perf_data.iterrows():
            f.write(f"Population size N = {row['population_size']}: Mean simulation time = {row['simulation_time']:.2f} seconds\n")
        
        # Calculate and report scaling factor
        if len(perf_data) > 1:
            log_sizes = np.log(perf_data['population_size'])
            log_times = np.log(perf_data['simulation_time'])
            slope, _, _, _, _ = stats.linregress(log_sizes, log_times)
            
            f.write(f"\nComputational scaling: Simulation time scales approximately as O(N^{slope:.2f})\n")
            f.write("This means doubling the population size increases simulation time ")
            f.write(f"by a factor of approximately {2**slope:.2f}.\n\n")
        
        # Overall conclusions
        f.write("="*80 + "\n")
        f.write("Overall Conclusions:\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Selection Coefficient (s): The most important parameter determining fixation probability.\n")
        f.write("   Higher s values consistently lead to higher fixation probabilities for advantageous alleles.\n\n")
        
        f.write("2. Population Size (N): Affects the balance between selection and drift.\n")
        f.write("   For advantageous alleles (s > 0), larger N increases fixation probability.\n")
        f.write("   For deleterious alleles (s < 0), larger N decreases fixation probability.\n")
        f.write("   For neutral alleles (s = 0), N has little effect on fixation probability.\n\n")
        
        f.write("3. Initial Frequency: Strongly correlated with fixation probability.\n")
        f.write("   Higher initial frequencies lead to higher fixation probabilities, as expected.\n\n")
        
        f.write("4. Fixation Time: Scales approximately linearly with population size.\n")
        f.write("   Selection coefficient affects fixation time: stronger selection leads to faster fixation.\n\n")
        
        f.write("5. Computational Performance: Simulation time scales with population size.\n")
        f.write("   Consider performance tradeoffs when running simulations with very large populations.\n\n")
    
    logging.info(f"Parameter sweep report created at {report_path}")
    return report_path

def run_single_parameter_set(results_path, args):
    """Run simulation with a single set of parameters"""
    # Log simulation parameters
    logging.info(f"Starting Moran process simulation with parameters:")
    logging.info(f"Population size: {args.population_size}")
    logging.info(f"Initial type A count: {args.initial_a_count}")
    logging.info(f"Selection coefficient: {args.selection_coefficient}")
    logging.info(f"Number of simulations: {args.num_simulations}")
    
    # Calculate theoretical fixation probability
    theoretical_prob = theoretical_fixation_probability(
        args.population_size, args.initial_a_count, args.selection_coefficient
    )
    logging.info(f"Theoretical fixation probability: {theoretical_prob:.6f}")
    
    # Run multiple simulations to calculate empirical fixation probability
    logging.info("Calculating empirical fixation probability...")
    
    fixation_prob, mean_time = calculate_fixation_probability(
        args.population_size, args.initial_a_count, args.selection_coefficient, args.num_simulations
    )
    
    logging.info(f"Empirical fixation probability: {fixation_prob:.6f}")
    logging.info(f"Mean fixation time: {mean_time:.2f} steps")
    
    # Compare theoretical to empirical
    logging.info(f"Difference (theoretical - empirical): {theoretical_prob - fixation_prob:.6f}")
    
    # Generate trajectory plots
    if args.show_trajectories > 0:
        logging.info(f"Generating {args.show_trajectories} trajectory plots...")
        
        trajectories = []
        for i in range(args.show_trajectories):
            logging.info(f"Running trajectory simulation {i+1}/{args.show_trajectories}")
            _, _, trajectory = run_single_simulation(
                args.population_size, args.initial_a_count, args.selection_coefficient
            )
            trajectories.append(trajectory)
        
        # Plot trajectories
        plot_trajectories(trajectories, args.selection_coefficient, args.population_size, results_path)
    
    # Compare fixation probabilities across multiple selection coefficients
    logging.info("Comparing fixation probabilities across selection coefficients...")
    
    selection_values = np.linspace(-0.5, 0.5, 11)  # From -0.5 to 0.5
    theoretical_probs = []
    empirical_probs = []
    
    for s in selection_values:
        # Theoretical probability
        theoretical_prob = theoretical_fixation_probability(
            args.population_size, args.initial_a_count, s
        )
        theoretical_probs.append(theoretical_prob)
        
        # Empirical probability (with fewer simulations for speed)
        fixation_prob, _ = calculate_fixation_probability(
            args.population_size, args.initial_a_count, s, max(10, args.num_simulations // 10)
        )
        empirical_probs.append(fixation_prob)
        
        logging.info(f"s={s:.2f}: Theoretical={theoretical_prob:.4f}, Empirical={fixation_prob:.4f}")
    
    # Plot comparison
    plot_fixation_comparison(
        theoretical_probs, empirical_probs, selection_values, args.population_size, results_path
    )
    
    # Print summary to console
    print("\nMoran Process Simulation Summary:")
    print("="*50)
    print(f"Population size: {args.population_size}")
    print(f"Initial type A count: {args.initial_a_count}")
    print(f"Selection coefficient: {args.selection_coefficient}")
    print(f"Number of simulations: {args.num_simulations}")
    print("-"*50)
    print(f"Theoretical fixation probability: {theoretical_prob:.6f}")
    print(f"Empirical fixation probability: {fixation_prob:.6f}")
    print(f"Mean fixation time: {mean_time:.2f} steps")
    print("-"*50)
    print(f"Results saved to: {results_path}")
    print("="*50)

#######################################################################
# Visualization functions
#######################################################################
def plot_trajectories(trajectories, selection_coefficient, population_size, results_path):
    """
    Plot multiple simulation trajectories showing type A frequency over time
    
    Parameters:
    - trajectories: List of frequency trajectories from simulations
    - selection_coefficient: Selection coefficient used in simulations
    - population_size: Population size used in simulations
    - results_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj, alpha=0.7, linewidth=1, label=f"Simulation {i+1}" if i < 5 else None)
    
    plt.xlabel("Time Step")
    plt.ylabel("Frequency of Type A")
    plt.title(f"Moran Process Trajectories (N={population_size}, s={selection_coefficient})")
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    # Only show legend for first few trajectories
    if len(trajectories) > 5:
        plt.legend(loc='upper right')
    else:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(results_path, f"trajectories_N{population_size}_s{selection_coefficient}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logging.info(f"Trajectory plot saved to {plot_path}")
    return plot_path

def plot_fixation_comparison(theoretical, empirical, selection_values, population_size, results_path):
    """
    Plot comparison between theoretical and empirical fixation probabilities
    
    Parameters:
    - theoretical: List of theoretical fixation probabilities
    - empirical: List of empirical fixation probabilities
    - selection_values: List of selection coefficients used
    - population_size: Population size used in simulations
    - results_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(selection_values, theoretical, 'b-', linewidth=2, label='Theoretical')
    plt.plot(selection_values, empirical, 'ro', markersize=6, label='Empirical')
    
    plt.xlabel("Selection Coefficient (s)")
    plt.ylabel("Fixation Probability")
    plt.title(f"Fixation Probability Comparison (N={population_size})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(results_path, f"fixation_comparison_N{population_size}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logging.info(f"Fixation probability comparison plot saved to {plot_path}")
    return plot_path

#######################################################################
# Main function
#######################################################################
def main():
    parser = argparse.ArgumentParser(description="Moran Process Simulation")
    parser.add_argument("--population_size", type=int, default=100, help="Size of the population")
    parser.add_argument("--initial_a_count", type=int, default=1, help="Initial number of type A individuals")
    parser.add_argument("--selection_coefficient", type=float, default=0.1, help="Selection coefficient for type A")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of independent simulation runs")
    parser.add_argument("--show_trajectories", type=int, default=10, help="Number of trajectories to plot")
    parser.add_argument("--results_path", type=str, default="./results", help="Path to save results")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    parser.add_argument("--parameter_sweep", action="store_true", help="Run parameter sweep instead of single simulation")
    
    args = parser.parse_args()
    
    # Create timestamp for unique results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.parameter_sweep:
        results_dir = os.path.join(args.results_path, f"moran_process_parameter_sweep_{timestamp}")
    else:
        results_dir = os.path.join(args.results_path, f"moran_process_N{args.population_size}_s{args.selection_coefficient}_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logging
    logger, results_path = init_log(results_dir, args.log_level)
    
    if args.parameter_sweep:
        run_parameter_sweep(results_path, args)
    else:
        run_single_parameter_set(results_path, args)
    
    logging.info("Simulation complete!")

if __name__ == "__main__":
    main()

## example usage
# python src/main.py --population_size 100 --initial_a_count 1 --selection_coefficient 0.1 --num_simulations 100
# python src/main.py --parameter_sweep