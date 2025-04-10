"""
This script simulates the Moran process for a population of individuals
with two types (A and B) and calculates the fixation probability of type A.
It includes extensions for environmental fluctuations and mutation.

Author: BC (original) with extensions
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
from scipy import stats  

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

def _get_lsf_job_details() -> list[str]:
    """
    Retrieves environment variables for LSF job details, if available.
    """
    lsf_job_id = os.environ.get('LSB_JOBID')    # Job ID
    lsf_queue = os.environ.get('LSB_QUEUE')     # Queue name
    lsf_host = os.environ.get('LSB_HOSTS')      # Hosts allocated
    lsf_job_name = os.environ.get('LSB_JOBNAME')  # Job name
    lsf_command = os.environ.get('LSB_CMD')     # Command used to submit job

    details = [
        f"LSF Job ID: {lsf_job_id}",
        f"LSF Queue: {lsf_queue}",
        f"LSF Hosts: {lsf_host}",
        f"LSF Job Name: {lsf_job_name}",
        f"LSF Command: {lsf_command}"
    ]
    return details

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
    def __init__(self, size, initial_a_count, selection_coefficient, mutation_rate=0, env_fluctuation=None):
        """
        Initialize a population for Moran Process simulation
        
        Parameters:
        - size: Total population size
        - initial_a_count: Initial number of type A individuals
        - selection_coefficient: Selection advantage of type A over type B
        - mutation_rate: Probability of mutation during reproduction (default: 0)
        - env_fluctuation: Dictionary containing environmental fluctuation parameters (default: None)
          - 'amplitude': Amplitude of fluctuation
          - 'period': Period of fluctuation (in steps)
          - 'type': Type of fluctuation ('sine', 'random', etc.)
        """
        self.size = size
        self.base_selection_coefficient = selection_coefficient
        self.selection_coefficient = selection_coefficient
        self.mutation_rate = mutation_rate
        self.env_fluctuation = env_fluctuation
        self.step_count = 0
        
        # Initialize population state: 1 for type A, 0 for type B
        self.population = np.zeros(size, dtype=int)
        self.population[:initial_a_count] = 1
        np.random.shuffle(self.population)
        
        # Track current count of type A
        self.type_a_count = initial_a_count
        
        # Track selection coefficient over time (for environmental fluctuations)
        self.selection_history = [selection_coefficient]
        
    def get_a_frequency(self):
        """Return the current frequency of type A in the population"""
        return self.type_a_count / self.size
    
    def update_environment(self):
        """Update selection coefficient based on environmental fluctuations"""
        if self.env_fluctuation is None:
            return
        
        amplitude = self.env_fluctuation.get('amplitude', 0.1)
        period = self.env_fluctuation.get('period', 100)
        fluctuation_type = self.env_fluctuation.get('type', 'sine')
        
        if fluctuation_type == 'sine':
            # Sinusoidal fluctuation
            phase = 2 * np.pi * (self.step_count % period) / period
            fluctuation = amplitude * np.sin(phase)
            self.selection_coefficient = self.base_selection_coefficient + fluctuation
        
        elif fluctuation_type == 'random':
            # Random fluctuation
            fluctuation = amplitude * (2 * np.random.random() - 1)
            self.selection_coefficient = self.base_selection_coefficient + fluctuation
            
        elif fluctuation_type == 'step':
            # Step function (alternating environments)
            if (self.step_count % period) < (period / 2):
                self.selection_coefficient = self.base_selection_coefficient + amplitude
            else:
                self.selection_coefficient = self.base_selection_coefficient - amplitude
        
        # Record current selection coefficient
        self.selection_history.append(self.selection_coefficient)
    
    def apply_mutation(self, reproducing_type):
        """Apply mutation to the reproducing individual's type"""
        if np.random.random() < self.mutation_rate:
            # Mutation occurs - flip the type
            return 1 - reproducing_type
        else:
            # No mutation - return original type
            return reproducing_type
    
    def step(self):
        """
        Perform one step of the Moran process with mutation and environmental fluctuations:
        1. Update environment (if enabled)
        2. Select an individual for reproduction based on fitness
        3. Create offspring (with possible mutation)
        4. Select random individual for death
        5. Replace dead with offspring
        
        Returns:
        - True if population changed, False otherwise
        """
        # Update step counter
        self.step_count += 1
        
        # Update environment
        self.update_environment()
        
        # Calculate fitness-weighted selection probabilities
        fitness = np.ones(self.size)
        fitness[self.population == 1] = 1 + self.selection_coefficient
        
        # Ensure fitness is non-negative by adding a constant if needed
        if np.min(fitness) < 0:
            fitness += abs(np.min(fitness)) + 0.1
            
        fitness_sum = np.sum(fitness)
        selection_probs = fitness / fitness_sum
        
        # Select individual for reproduction
        reproducing_idx = np.random.choice(self.size, p=selection_probs)
        reproducing_type = self.population[reproducing_idx]
        
        # Apply mutation
        offspring_type = self.apply_mutation(reproducing_type)
        
        # Select random individual for death
        dying_idx = np.random.choice(self.size)
        dying_type = self.population[dying_idx]
        
        # Update population if there's a change
        if offspring_type != dying_type:
            self.population[dying_idx] = offspring_type
            self.type_a_count += (1 if offspring_type == 1 else -1) - (1 if dying_type == 1 else 0)
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
def run_single_simulation(population_size, initial_a_count, selection_coefficient, 
                         mutation_rate=0, env_fluctuation=None, max_steps=None):
    """
    Run a single simulation of the Moran process until fixation
    
    Parameters:
    - population_size: Size of the population
    - initial_a_count: Initial number of type A individuals
    - selection_coefficient: Selective advantage of type A
    - mutation_rate: Probability of mutation during reproduction
    - env_fluctuation: Dictionary with environmental fluctuation parameters
    - max_steps: Maximum number of steps (default: None for unlimited)
    
    Returns:
    - is_a_fixed: Whether type A fixed in the population
    - steps: Number of steps until fixation
    - trajectory: List of type A frequencies at each step
    - selection_history: List of selection coefficients at each step (for env_fluctuation)
    """
    # Set a reasonable default maximum steps to prevent infinite loops
    if max_steps is None:
        max_steps = max(10000, population_size * 100)
    
    pop = Population(population_size, initial_a_count, selection_coefficient, 
                    mutation_rate, env_fluctuation)
    
    trajectory = [pop.get_a_frequency()]
    steps = 0    
    while not pop.is_fixed():
        pop.step()
        steps += 1
        trajectory.append(pop.get_a_frequency())
        
        if steps >= max_steps:
            logging.debug(f"Simulation terminated after reaching maximum steps: {max_steps}")
            break
    
    return pop.is_type_a_fixed(), steps, trajectory, pop.selection_history

@timing_decorator
def calculate_fixation_probability(population_size, initial_a_count, selection_coefficient, 
                                  num_simulations=100, mutation_rate=0, env_fluctuation=None):
    """
    Calculate the empirical fixation probability of type A by running multiple simulations
    
    Parameters:
    - population_size: Size of the population
    - initial_a_count: Initial number of type A individuals
    - selection_coefficient: Selective advantage of type A
    - num_simulations: Number of independent simulation runs
    - mutation_rate: Probability of mutation during reproduction
    - env_fluctuation: Dictionary with environmental fluctuation parameters
    
    Returns:
    - fixation_probability: Empirical probability of type A fixation
    - mean_fixation_time: Mean time to fixation across simulations
    """
    fixed_count = 0
    fixation_times = []
    
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        is_a_fixed, steps, _, _ = run_single_simulation(
            population_size, initial_a_count, selection_coefficient, 
            mutation_rate, env_fluctuation
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
    
    Note: This function doesn't account for mutation or environmental fluctuations
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

#######################################################################
# Extension-specific functions
#######################################################################
def run_mutation_analysis(results_path, args):
    """
    Analyze the effect of mutation rate on fixation probability
    
    Parameters:
    - results_path: Path to save results
    - args: Command line arguments
    """
    logging.info("Starting mutation rate analysis...")
    
    # Define mutation rates to test
    mutation_rates = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    # Define selection coefficients to test
    selection_coefficients = [-0.5, -0.1, 0, 0.1, 0.5]
    
    # Track results
    results = []
    
    # Create results directory
    mutation_results_dir = os.path.join(results_path, "mutation_analysis")
    os.makedirs(mutation_results_dir, exist_ok=True)
    
    # Total experiments
    total_experiments = len(mutation_rates) * len(selection_coefficients)
    experiment_count = 0
    
    # Run simulations
    for mu in mutation_rates:
        for s in selection_coefficients:
            experiment_count += 1
            logging.info(f"Running experiment {experiment_count}/{total_experiments}: "
                       f"mutation_rate={mu}, selection_coefficient={s}")
            
            # Calculate theoretical fixation probability (without mutation)
            theo_prob = theoretical_fixation_probability(
                args.population_size, args.initial_a_count, s
            )
            
            # Run simulations
            fix_prob, mean_time = calculate_fixation_probability(
                args.population_size, args.initial_a_count, s, 
                args.num_simulations, mutation_rate=mu
            )
            
            # Record results
            results.append({
                'mutation_rate': mu,
                'selection_coefficient': s,
                'theoretical_fixation_prob': theo_prob,
                'empirical_fixation_prob': fix_prob,
                'mean_fixation_time': mean_time
            })
            
            logging.info(f"  Results: Fixation Probability = {fix_prob:.4f}, "
                       f"Mean Fixation Time = {mean_time:.1f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    csv_path = os.path.join(mutation_results_dir, "mutation_analysis_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Raw results saved to {csv_path}")
    
    # Plot results
    plot_mutation_effects(results_df, mutation_results_dir)
    
    # Create summary report
    create_mutation_analysis_report(results_df, mutation_results_dir)
    
    return results_df

def plot_mutation_effects(results_df, results_path):
    """
    Plot the effects of mutation rate on fixation probability and time
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - results_path: Path to save plots
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions
    
    # 1. Effect of mutation rate on fixation probability for different selection coefficients
    plt.figure(figsize=(10, 6))
    for s in results_df['selection_coefficient'].unique():
        subset = results_df[results_df['selection_coefficient'] == s]
        plt.plot(subset['mutation_rate'], subset['empirical_fixation_prob'], 
                 marker='o', label=f"s = {s}")
    
    plt.xlabel("Mutation Rate")
    plt.ylabel("Fixation Probability")
    plt.title("Effect of Mutation Rate on Fixation Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "mutation_fixation_probability.png"), dpi=300)
    plt.close()
    
    # 2. Effect of mutation rate on fixation time for different selection coefficients
    plt.figure(figsize=(10, 6))
    for s in results_df['selection_coefficient'].unique():
        subset = results_df[results_df['selection_coefficient'] == s]
        plt.plot(subset['mutation_rate'], subset['mean_fixation_time'], 
                 marker='o', label=f"s = {s}")
    
    plt.xlabel("Mutation Rate")
    plt.ylabel("Mean Fixation Time")
    plt.title("Effect of Mutation Rate on Fixation Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "mutation_fixation_time.png"), dpi=300)
    plt.close()
    
    # 3. Deviation from theoretical prediction
    plt.figure(figsize=(10, 6))
    for s in results_df['selection_coefficient'].unique():
        subset = results_df[results_df['selection_coefficient'] == s]
        deviation = subset['empirical_fixation_prob'] - subset['theoretical_fixation_prob']
        plt.plot(subset['mutation_rate'], deviation, 
                 marker='o', label=f"s = {s}")
    
    plt.xlabel("Mutation Rate")
    plt.ylabel("Deviation from Theoretical Prediction")
    plt.title("Effect of Mutation on Theoretical Predictions")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "mutation_theory_deviation.png"), dpi=300)
    plt.close()

def create_mutation_analysis_report(results_df, results_path):
    """
    Create a summary report for mutation analysis
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - results_path: Path to save the report
    """
    report_path = os.path.join(results_path, "mutation_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Moran Process: Mutation Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        # Write introduction
        f.write("This report summarizes how mutation affects the Moran Process dynamics.\n\n")
        
        # Summary statistics
        f.write("-"*80 + "\n")
        f.write("Summary Statistics:\n")
        f.write("-"*80 + "\n")
        
        f.write(f"Mutation rates tested: {sorted(results_df['mutation_rate'].unique())}\n")
        f.write(f"Selection coefficients tested: {sorted(results_df['selection_coefficient'].unique())}\n\n")
        
        # Effect of mutation rate for each selection coefficient
        f.write("-"*80 + "\n")
        f.write("Effect of Mutation Rate:\n")
        f.write("-"*80 + "\n")
        
        for s in sorted(results_df['selection_coefficient'].unique()):
            subset = results_df[results_df['selection_coefficient'] == s]
            
            f.write(f"For selection coefficient s = {s}:\n")
            
            # Calculate correlation
            corr = subset['mutation_rate'].corr(subset['empirical_fixation_prob'])
            f.write(f"  Correlation between mutation rate and fixation probability: {corr:.4f}\n")
            
            # Report trend
            if s > 0:
                if corr < 0:
                    f.write("  For advantageous mutations (s > 0), higher mutation rates decrease fixation probability.\n")
                    f.write("  This is because mutations can convert advantageous type A to less fit type B.\n")
                else:
                    f.write("  For advantageous mutations (s > 0), mutation rate has a positive effect on fixation probability.\n")
                    f.write("  This suggests back-mutation from type B to advantageous type A may be significant.\n")
            elif s < 0:
                if corr > 0:
                    f.write("  For deleterious mutations (s < 0), higher mutation rates increase fixation probability.\n")
                    f.write("  This suggests mutations help deleterious types escape elimination by creating advantageous types.\n")
                else:
                    f.write("  For deleterious mutations (s < 0), mutation rate has a negative effect on fixation probability.\n")
            else:  # s == 0
                f.write(f"  For neutral mutations (s = 0), mutation rate has {abs(corr):.4f} correlation with fixation probability.\n")
                f.write("  In neutral evolution, mutation should push the system toward a mutation-selection balance.\n")
            
            # Effect on fixation time
            time_corr = subset['mutation_rate'].corr(subset['mean_fixation_time'])
            f.write(f"  Correlation between mutation rate and fixation time: {time_corr:.4f}\n")
            
            if time_corr > 0:
                f.write("  Higher mutation rates increase the time to fixation.\n")
                f.write("  This is expected as mutations can reverse fixation trends, extending the process.\n")
            else:
                f.write("  Higher mutation rates decrease the time to fixation.\n")
                f.write("  This may be due to mutations accelerating the spread of more fit types.\n")
            
            f.write("\n")
        
        # Overall conclusions
        f.write("="*80 + "\n")
        f.write("Overall Conclusions:\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Mutation Effects on Fixation Probability:\n")
        f.write("   Mutations create a more complex dynamic than the basic Moran process.\n")
        f.write("   For advantageous types, mutations can reduce fixation probability by creating less fit types.\n")
        f.write("   For deleterious types, mutations can increase fixation probability by creating more fit types.\n\n")
        
        f.write("2. Mutation Effects on Fixation Time:\n")
        f.write("   Generally, mutations increase fixation time by creating a more dynamic equilibrium.\n")
        f.write("   At high mutation rates, true fixation may become impossible as the population\n")
        f.write("   reaches a mutation-selection balance rather than fixation of either type.\n\n")
        
        f.write("3. Deviation from Theory:\n")
        f.write("   The theoretical fixation probabilities derived for the basic Moran process\n")
        f.write("   become less accurate as mutation rates increase.\n")
        f.write("   This highlights the limitations of simple models when additional evolutionary forces are at play.\n\n")
    
    logging.info(f"Mutation analysis report created at {report_path}")
    return report_path

def run_environmental_fluctuation_analysis(results_path, args):
    """
    Analyze the effect of environmental fluctuations on fixation probability
    
    Parameters:
    - results_path: Path to save results
    - args: Command line arguments
    """
    logging.info("Starting environmental fluctuation analysis...")
    
    # Define environmental fluctuation parameters to test
    fluctuation_types = ['sine', 'random', 'step']
    amplitudes = [0.1, 0.3, 0.5]
    periods = [10, 50, 200]
    
    # Track results
    results = []
    
    # Create results directory
    env_results_dir = os.path.join(results_path, "environment_analysis")
    os.makedirs(env_results_dir, exist_ok=True)
    
    # Run experiment with no fluctuation for baseline
    theo_prob = theoretical_fixation_probability(
        args.population_size, args.initial_a_count, args.selection_coefficient
    )
    
    baseline_fix_prob, baseline_time = calculate_fixation_probability(
        args.population_size, args.initial_a_count, args.selection_coefficient, 
        args.num_simulations
    )
    
    results.append({
        'fluctuation_type': 'none',
        'amplitude': 0,
        'period': 0,
        'theoretical_fixation_prob': theo_prob,
        'empirical_fixation_prob': baseline_fix_prob,
        'mean_fixation_time': baseline_time
    })
    
    logging.info(f"Baseline (no fluctuation): Fixation Probability = {baseline_fix_prob:.4f}, "
               f"Mean Fixation Time = {baseline_time:.1f}")
    
    # Total experiments
    total_experiments = len(fluctuation_types) * len(amplitudes) * len(periods)
    experiment_count = 0
    
    # Run simulations with environmental fluctuations
    for fluct_type in fluctuation_types:
        for amplitude in amplitudes:
            for period in periods:
                experiment_count += 1
                logging.info(f"Running experiment {experiment_count}/{total_experiments}: "
                           f"type={fluct_type}, amplitude={amplitude}, period={period}")
                
                # Define environmental fluctuation parameters
                env_fluctuation = {
                    'type': fluct_type,
                    'amplitude': amplitude,
                    'period': period
                }
                
                # Run simulations
                fix_prob, mean_time = calculate_fixation_probability(
                    args.population_size, args.initial_a_count, args.selection_coefficient, 
                    args.num_simulations, env_fluctuation=env_fluctuation
                )
                
                # Record results
                results.append({
                    'fluctuation_type': fluct_type,
                    'amplitude': amplitude,
                    'period': period,
                    'theoretical_fixation_prob': theo_prob,
                    'empirical_fixation_prob': fix_prob,
                    'mean_fixation_time': mean_time
                })
                
                logging.info(f"  Results: Fixation Probability = {fix_prob:.4f}, "
                           f"Mean Fixation Time = {mean_time:.1f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    csv_path = os.path.join(env_results_dir, "environment_analysis_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Raw results saved to {csv_path}")
    
    # Plot results
    plot_environmental_effects(results_df, env_results_dir)
    
    # Generate sample trajectories
    for fluct_type in fluctuation_types:
        # Use mid-range amplitude and period for examples
        env_fluctuation = {
            'type': fluct_type,
            'amplitude': 0.3,
            'period': 50
        }
        
        # Run simulations and collect trajectories
        trajectories = []
        selection_histories = []
        
        for i in range(5):  # Generate 5 example trajectories
            _, _, trajectory, sel_history = run_single_simulation(
                args.population_size, args.initial_a_count, args.selection_coefficient,
                env_fluctuation=env_fluctuation
            )
            trajectories.append(trajectory)
            selection_histories.append(sel_history)
        
        # Plot trajectories with selection coefficient
        plot_env_trajectories(trajectories, selection_histories, fluct_type, env_results_dir)
    
    # Create summary report
    create_environmental_analysis_report(results_df, env_results_dir)
    
    return results_df

def plot_environmental_effects(results_df, results_path):
    """
    Plot the effects of environmental fluctuations on fixation probability and time
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - results_path: Path to save plots
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions
    
    # Extract baseline results (no fluctuation)
    baseline = results_df[results_df['fluctuation_type'] == 'none']
    baseline_prob = baseline['empirical_fixation_prob'].values[0]
    baseline_time = baseline['mean_fixation_time'].values[0]
    
    # Filter to only include results with fluctuations
    env_results = results_df[results_df['fluctuation_type'] != 'none']
    
    # 1. Effect of amplitude on fixation probability for different fluctuation types
    plt.figure(figsize=(10, 6))
    for fluct_type in env_results['fluctuation_type'].unique():
        # Average across periods for each amplitude
        subset = env_results[env_results['fluctuation_type'] == fluct_type]
        grouped = subset.groupby('amplitude')['empirical_fixation_prob'].mean().reset_index()
        
        plt.plot(grouped['amplitude'], grouped['empirical_fixation_prob'], 
                 marker='o', label=fluct_type)
    
    plt.axhline(y=baseline_prob, color='black', linestyle='--', 
                label=f"Baseline (no fluctuation): {baseline_prob:.3f}")
    
    plt.xlabel("Fluctuation Amplitude")
    plt.ylabel("Fixation Probability")
    plt.title("Effect of Environmental Fluctuation Amplitude on Fixation Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "env_amplitude_fixation_prob.png"), dpi=300)
    plt.close()
    
    # 2. Effect of period on fixation probability for different fluctuation types
    plt.figure(figsize=(10, 6))
    for fluct_type in env_results['fluctuation_type'].unique():
        # Average across amplitudes for each period
        subset = env_results[env_results['fluctuation_type'] == fluct_type]
        grouped = subset.groupby('period')['empirical_fixation_prob'].mean().reset_index()
        
        plt.plot(grouped['period'], grouped['empirical_fixation_prob'], 
                 marker='o', label=fluct_type)
    
    plt.axhline(y=baseline_prob, color='black', linestyle='--', 
                label=f"Baseline (no fluctuation): {baseline_prob:.3f}")
    
    plt.xlabel("Fluctuation Period")
    plt.ylabel("Fixation Probability")
    plt.title("Effect of Environmental Fluctuation Period on Fixation Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "env_period_fixation_prob.png"), dpi=300)
    plt.close()
    
    # 3. Effect on fixation time
    # For amplitude
    plt.figure(figsize=(10, 6))
    for fluct_type in env_results['fluctuation_type'].unique():
        subset = env_results[env_results['fluctuation_type'] == fluct_type]
        grouped = subset.groupby('amplitude')['mean_fixation_time'].mean().reset_index()
        
        plt.plot(grouped['amplitude'], grouped['mean_fixation_time'], 
                 marker='o', label=fluct_type)
    
    plt.axhline(y=baseline_time, color='black', linestyle='--', 
                label=f"Baseline (no fluctuation): {baseline_time:.0f}")
    
    plt.xlabel("Fluctuation Amplitude")
    plt.ylabel("Mean Fixation Time")
    plt.title("Effect of Environmental Fluctuation Amplitude on Fixation Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "env_amplitude_fixation_time.png"), dpi=300)
    plt.close()
    
    # For period
    plt.figure(figsize=(10, 6))
    for fluct_type in env_results['fluctuation_type'].unique():
        subset = env_results[env_results['fluctuation_type'] == fluct_type]
        grouped = subset.groupby('period')['mean_fixation_time'].mean().reset_index()
        
        plt.plot(grouped['period'], grouped['mean_fixation_time'], 
                 marker='o', label=fluct_type)
    
    plt.axhline(y=baseline_time, color='black', linestyle='--', 
                label=f"Baseline (no fluctuation): {baseline_time:.0f}")
    
    plt.xlabel("Fluctuation Period")
    plt.ylabel("Mean Fixation Time")
    plt.title("Effect of Environmental Fluctuation Period on Fixation Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, "env_period_fixation_time.png"), dpi=300)
    plt.close()
    
    # 4. Heatmap showing combined effects of amplitude and period for each fluctuation type
    for fluct_type in env_results['fluctuation_type'].unique():
        subset = env_results[env_results['fluctuation_type'] == fluct_type]
        
        # Create pivot table for heatmap
        heatmap_data = subset.pivot_table(
            values='empirical_fixation_prob',
            index='amplitude',
            columns='period'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                   cbar_kws={'label': 'Fixation Probability'})
        plt.title(f"Fixation Probability Heatmap for {fluct_type.capitalize()} Fluctuations")
        plt.xlabel("Period")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"env_{fluct_type}_heatmap.png"), dpi=300)
        plt.close()

def plot_env_trajectories(trajectories, selection_histories, fluctuation_type, results_path):
    """
    Plot population trajectories with selection coefficient fluctuations
    
    Parameters:
    - trajectories: List of frequency trajectories from simulations
    - selection_histories: List of selection coefficient histories
    - fluctuation_type: Type of environmental fluctuation
    - results_path: Path to save the plot
    """
    # Plot both trajectories and selection coefficient in a 2-row plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot population trajectories
    for i, traj in enumerate(trajectories):
        ax1.plot(range(len(traj)), traj, alpha=0.7, linewidth=1, 
                label=f"Simulation {i+1}" if i < 5 else None)
    
    ax1.set_ylabel("Frequency of Type A")
    ax1.set_title(f"Moran Process with {fluctuation_type.capitalize()} Environmental Fluctuations")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add reference lines
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    # Plot selection coefficient histories
    for i, hist in enumerate(selection_histories):
        time_steps = list(range(len(hist)))
        if i == 0:  # Only show one selection history to avoid clutter
            ax2.plot(time_steps, hist, alpha=0.7, color='red',
                    label="Selection Coefficient")
    
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Selection Coefficient")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_path, f"trajectories_{fluctuation_type}_env.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return plot_path

def create_environmental_analysis_report(results_df, results_path):
    """
    Create a summary report for environmental fluctuation analysis
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - results_path: Path to save the report
    """
    report_path = os.path.join(results_path, "environmental_analysis_report.txt")
    
    # Extract baseline results (no fluctuation)
    baseline = results_df[results_df['fluctuation_type'] == 'none']
    baseline_prob = baseline['empirical_fixation_prob'].values[0]
    baseline_time = baseline['mean_fixation_time'].values[0]
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Moran Process: Environmental Fluctuation Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        # Write introduction
        f.write("This report summarizes how environmental fluctuations affect the Moran Process dynamics.\n\n")
        
        # Summary statistics
        f.write("-"*80 + "\n")
        f.write("Summary Statistics:\n")
        f.write("-"*80 + "\n")
        
        f.write(f"Fluctuation types tested: {sorted(results_df['fluctuation_type'].unique())}\n")
        f.write(f"Amplitudes tested: {sorted(results_df['amplitude'].unique())}\n")
        f.write(f"Periods tested: {sorted(results_df['period'].unique())}\n\n")
        
        f.write(f"Baseline (no fluctuation) fixation probability: {baseline_prob:.4f}\n")
        f.write(f"Baseline (no fluctuation) mean fixation time: {baseline_time:.1f}\n\n")
        
        # Effect of fluctuation type
        f.write("-"*80 + "\n")
        f.write("Effect of Fluctuation Type:\n")
        f.write("-"*80 + "\n")
        
        # Filter to only include results with fluctuations
        env_results = results_df[results_df['fluctuation_type'] != 'none']
        
        for fluct_type in sorted(env_results['fluctuation_type'].unique()):
            subset = env_results[env_results['fluctuation_type'] == fluct_type]
            mean_prob = subset['empirical_fixation_prob'].mean()
            mean_time = subset['mean_fixation_time'].mean()
            
            prob_diff = mean_prob - baseline_prob
            time_diff = mean_time - baseline_time
            
            f.write(f"For {fluct_type} fluctuations:\n")
            f.write(f"  Mean fixation probability: {mean_prob:.4f} ")
            f.write(f"(Change: {prob_diff:+.4f} from baseline)\n")
            f.write(f"  Mean fixation time: {mean_time:.1f} ")
            f.write(f"(Change: {time_diff:+.1f} from baseline)\n\n")
            
            # Effect of amplitude for this fluctuation type
            f.write(f"  Effect of amplitude for {fluct_type} fluctuations:\n")
            for amp in sorted(subset['amplitude'].unique()):
                amp_subset = subset[subset['amplitude'] == amp]
                amp_prob = amp_subset['empirical_fixation_prob'].mean()
                amp_time = amp_subset['mean_fixation_time'].mean()
                
                f.write(f"    Amplitude {amp}: Fixation prob = {amp_prob:.4f}, Fixation time = {amp_time:.1f}\n")
            
            f.write("\n")
            
            # Effect of period for this fluctuation type
            f.write(f"  Effect of period for {fluct_type} fluctuations:\n")
            for period in sorted(subset['period'].unique()):
                period_subset = subset[subset['period'] == period]
                period_prob = period_subset['empirical_fixation_prob'].mean()
                period_time = period_subset['mean_fixation_time'].mean()
                
                f.write(f"    Period {period}: Fixation prob = {period_prob:.4f}, Fixation time = {period_time:.1f}\n")
            
            f.write("\n")
        
        # Overall conclusions
        f.write("="*80 + "\n")
        f.write("Overall Conclusions:\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Environmental Fluctuation Effects on Fixation Probability:\n")
        # Calculate overall effects
        overall_effect = env_results['empirical_fixation_prob'].mean() - baseline_prob
        
        if abs(overall_effect) < 0.05:
            f.write("   Environmental fluctuations have minimal overall effect on fixation probability.\n")
            f.write("   Positive and negative fluctuations may cancel each other out over time.\n\n")
        elif overall_effect > 0:
            f.write("   Environmental fluctuations generally increase fixation probability.\n")
            f.write("   This suggests fluctuations may create windows of opportunity for advantageous mutations.\n\n")
        else:
            f.write("   Environmental fluctuations generally decrease fixation probability.\n")
            f.write("   This suggests fluctuations create periods where the advantage is reduced or reversed.\n\n")
        
        f.write("2. Environmental Fluctuation Effects on Fixation Time:\n")
        # Calculate overall effects on time
        overall_time_effect = env_results['mean_fixation_time'].mean() - baseline_time
        
        if overall_time_effect > 0:
            f.write("   Environmental fluctuations increase fixation time.\n")
            f.write("   This suggests periods of reduced advantage slow down the fixation process.\n\n")
        else:
            f.write("   Environmental fluctuations decrease fixation time.\n")
            f.write("   This suggests periods of increased advantage can accelerate fixation.\n\n")
        
        f.write("3. Comparison of Fluctuation Types:\n")
        # Compare fluctuation types
        f.write("   Type effects (largest to smallest impact on fixation probability):\n")
        
        type_effects = []
        for fluct_type in env_results['fluctuation_type'].unique():
            subset = env_results[env_results['fluctuation_type'] == fluct_type]
            effect = abs(subset['empirical_fixation_prob'].mean() - baseline_prob)
            type_effects.append((fluct_type, effect))
        
        for fluct_type, effect in sorted(type_effects, key=lambda x: x[1], reverse=True):
            f.write(f"   - {fluct_type}: {effect:.4f} absolute change from baseline\n")
        
        f.write("\n")
        
        f.write("4. Practical Implications:\n")
        f.write("   Environmental fluctuations create more complex evolutionary dynamics than constant environments.\n")
        f.write("   The effect depends on the pattern, amplitude, and frequency of the fluctuations.\n")
        f.write("   Fluctuations may help maintain genetic diversity by preventing complete fixation.\n")
        f.write("   Evolutionary models assuming constant selection may be inaccurate in fluctuating environments.\n\n")
    
    logging.info(f"Environmental analysis report created at {report_path}")
    return report_path

#######################################################################
# Main function
#######################################################################
def main():
    parser = argparse.ArgumentParser(description="Moran Process Simulation with Extensions")
    parser.add_argument("--population_size", type=int, default=100, help="Size of the population")
    parser.add_argument("--initial_a_count", type=int, default=1, help="Initial number of type A individuals")
    parser.add_argument("--selection_coefficient", type=float, default=0.1, help="Selection coefficient for type A")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of independent simulation runs")
    parser.add_argument("--show_trajectories", type=int, default=10, help="Number of trajectories to plot")
    parser.add_argument("--results_path", type=str, default="./results", help="Path to save results")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    parser.add_argument("--parameter_sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--mutation_rate", type=float, default=0.0, help="Mutation rate (probability of type change during reproduction)")
    parser.add_argument("--env_fluctuation", action="store_true", help="Enable environmental fluctuations")
    parser.add_argument("--env_amplitude", type=float, default=0.1, help="Amplitude of environmental fluctuations")
    parser.add_argument("--env_period", type=int, default=100, help="Period of environmental fluctuations")
    parser.add_argument("--env_type", choices=["sine", "random", "step"], default="sine", help="Type of environmental fluctuation")
    parser.add_argument("--run_extension_analysis", action="store_true", help="Run detailed analysis of extensions")
    
    args = parser.parse_args()
    
    # Create timestamp for unique results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.parameter_sweep:
        results_dir = os.path.join(args.results_path, f"moran_process_parameter_sweep_{timestamp}")
    elif args.run_extension_analysis:
        results_dir = os.path.join(args.results_path, f"moran_process_extensions_analysis_{timestamp}")
    else:
        # Include mutation and environment info in directory name if enabled
        mutation_str = f"_mu{args.mutation_rate}" if args.mutation_rate > 0 else ""
        env_str = f"_env{args.env_amplitude}_{args.env_period}" if args.env_fluctuation else ""
        results_dir = os.path.join(args.results_path, f"moran_process_N{args.population_size}_s{args.selection_coefficient}{mutation_str}{env_str}_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logging
    logger, results_path = init_log(results_dir, args.log_level)

    # lsf deat
    logging.info("_get_lsf_job_details :")
    details = _get_lsf_job_details()
    logging.info(details)
    
    if args.run_extension_analysis:
        # Run extensive analysis of both extensions
        logging.info("Running detailed analysis of extensions (mutation and environmental fluctuation)...")
        
        # First analyze mutation effects
        mutation_results = run_mutation_analysis(results_path, args)
        
        # Then analyze environmental fluctuation effects
        env_results = run_environmental_fluctuation_analysis(results_path, args)
        
        logging.info("Extensions analysis complete!")
    elif args.parameter_sweep:
        run_parameter_sweep(results_path, args)
    else:
        # Setup environmental fluctuation if enabled
        env_fluctuation = None
        if args.env_fluctuation:
            env_fluctuation = {
                'type': args.env_type,
                'amplitude': args.env_amplitude,
                'period': args.env_period
            }
            logging.info(f"Environmental fluctuations enabled: {env_fluctuation}")
        
        # Run single parameter set with possible extensions
        run_single_parameter_set(results_path, args, args.mutation_rate, env_fluctuation)
    
    logging.info("Simulation complete!")

def run_single_parameter_set(results_path, args, mutation_rate=0, env_fluctuation=None):
    """Run simulation with a single set of parameters and possible extensions"""
    # Log simulation parameters
    logging.info(f"Starting Moran process simulation with parameters:")
    logging.info(f"Population size: {args.population_size}")
    logging.info(f"Initial type A count: {args.initial_a_count}")
    logging.info(f"Selection coefficient: {args.selection_coefficient}")
    logging.info(f"Number of simulations: {args.num_simulations}")
    
    if mutation_rate > 0:
        logging.info(f"Mutation rate: {mutation_rate}")
    
    if env_fluctuation:
        logging.info(f"Environmental fluctuation: {env_fluctuation}")
    
    # Calculate theoretical fixation probability (without extensions)
    theoretical_prob = theoretical_fixation_probability(
        args.population_size, args.initial_a_count, args.selection_coefficient
    )
    logging.info(f"Theoretical fixation probability (without extensions): {theoretical_prob:.6f}")
    
    # Run multiple simulations to calculate empirical fixation probability
    logging.info("Calculating empirical fixation probability...")
    
    fixation_prob, mean_time = calculate_fixation_probability(
        args.population_size, args.initial_a_count, args.selection_coefficient, 
        args.num_simulations, mutation_rate, env_fluctuation
    )
    
    logging.info(f"Empirical fixation probability: {fixation_prob:.6f}")
    logging.info(f"Mean fixation time: {mean_time:.2f} steps")
    
    # Compare theoretical to empirical
    logging.info(f"Difference (theoretical - empirical): {theoretical_prob - fixation_prob:.6f}")
    logging.info(f"Note: Difference expected with extensions enabled")
    
    # Generate trajectory plots
    if args.show_trajectories > 0:
        logging.info(f"Generating {args.show_trajectories} trajectory plots...")
        
        trajectories = []
        selection_histories = []
        
        for i in range(args.show_trajectories):
            logging.info(f"Running trajectory simulation {i+1}/{args.show_trajectories}")
            _, _, trajectory, sel_history = run_single_simulation(
                args.population_size, args.initial_a_count, args.selection_coefficient,
                mutation_rate, env_fluctuation
            )
            trajectories.append(trajectory)
            selection_histories.append(sel_history)
        
        # Plot trajectories
        if env_fluctuation:
            # For environmental fluctuations, plot with selection coefficient
            plot_env_trajectories(trajectories, selection_histories, 
                                 env_fluctuation['type'], results_path)
        else:
            # Standard trajectory plot
            plot_trajectories(trajectories, args.selection_coefficient, 
                             args.population_size, results_path, 
                             mutation_rate=mutation_rate)
    
    # Print summary to console
    print("\nMoran Process Simulation Summary:")
    print("="*50)
    print(f"Population size: {args.population_size}")
    print(f"Initial type A count: {args.initial_a_count}")
    print(f"Selection coefficient: {args.selection_coefficient}")
    
    if mutation_rate > 0:
        print(f"Mutation rate: {mutation_rate}")
    
    if env_fluctuation:
        print(f"Environmental fluctuation: {env_fluctuation['type']}, "
              f"Amplitude: {env_fluctuation['amplitude']}, "
              f"Period: {env_fluctuation['period']}")
    
    print(f"Number of simulations: {args.num_simulations}")
    print("-"*50)
    print(f"Theoretical fixation probability (without extensions): {theoretical_prob:.6f}")
    print(f"Empirical fixation probability: {fixation_prob:.6f}")
    print(f"Mean fixation time: {mean_time:.2f} steps")
    print("-"*50)
    print(f"Results saved to: {results_path}")
    print("="*50)

#######################################################################
# Modified visualization functions
#######################################################################
def plot_trajectories(trajectories, selection_coefficient, population_size, results_path, mutation_rate=0):
    """
    Plot multiple simulation trajectories showing type A frequency over time
    
    Parameters:
    - trajectories: List of frequency trajectories from simulations
    - selection_coefficient: Selection coefficient used in simulations
    - population_size: Population size used in simulations
    - results_path: Path to save the plot
    - mutation_rate: Mutation rate used in simulations (default: 0)
    """
    plt.figure(figsize=(10, 6))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj, alpha=0.7, linewidth=1, label=f"Simulation {i+1}" if i < 5 else None)
    
    plt.xlabel("Time Step")
    plt.ylabel("Frequency of Type A")
    
    # Include mutation rate in title if applicable
    if mutation_rate > 0:
        plt.title(f"Moran Process Trajectories (N={population_size}, s={selection_coefficient}, μ={mutation_rate})")
    else:
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
    
    # Include mutation rate in filename if applicable
    if mutation_rate > 0:
        plot_path = os.path.join(results_path, f"trajectories_N{population_size}_s{selection_coefficient}_mu{mutation_rate}.png")
    else:
        plot_path = os.path.join(results_path, f"trajectories_N{population_size}_s{selection_coefficient}.png")
    
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logging.info(f"Trajectory plot saved to {plot_path}")
    return plot_path

# Execute main function if script is run directly
if __name__ == "__main__":
    main()