## The Moran Process: Core Concept


The Moran process is a mathematical model in population genetics that simulates how genetic variants compete in a fixed-size population. This program simulates a population with two types of individuals (A and B) and tracks whether type A eventually takes over the entire population (fixation) or disappears.

### Key Features of the Model:

1. **Fixed Population Size**: The total number of individuals remains constant throughout the simulation.

2. **Selection Coefficient**: This is the key parameter that determines the fitness advantage (or disadvantage) of type A individuals over type B. 
   - If positive (e.g., 0.1), type A has an advantage
   - If negative (e.g., -0.1), type A has a disadvantage
   - If zero, both types have equal fitness (neutral evolution)

3. **Random Selection Process**: In each step:
   - One individual is selected to reproduce based on fitness
   - One random individual is selected to die (regardless of fitness)
   - The offspring replaces the dead individual

## How Selection Works

The selection coefficient (s) directly affects the reproduction probability:
- Type A fitness = 1 + s
- Type B fitness = 1

For example, with s = 0.1:
- Type A individuals are 10% more likely to be selected for reproduction
- With s = -0.2, Type A individuals are 20% less likely to be selected

The program calculates selection probabilities using:
```python
fitness = np.ones(self.size)
fitness[self.population == 1] = 1 + self.selection_coefficient
selection_probs = fitness / np.sum(fitness)
```

## Fixation Probability

A key outcome measured by the simulation is the fixation probability - the likelihood that type A eventually takes over the entire population.

The theoretical fixation probability is calculated using:
```python
if s != 0:  # Non-neutral selection
    r = 1 + s  # relative fitness
    numerator = 1 - (1/r)**i
    denominator = 1 - (1/r)**N
    return numerator / denominator
else:  # Neutral selection
    return i / N
```

Where:
- N = population size
- i = initial count of type A individuals
- s = selection coefficient

The program compares this theoretical probability with empirical results from multiple simulation runs.

## Parameter Effects

When explaining to your boss, emphasize these key relationships:

1. **Selection Coefficient Effect**:
   - Positive values increase type A's chance of fixation
   - Negative values decrease type A's chance of fixation
   - The stronger the selection (higher |s|), the stronger the effect

2. **Population Size Effect**:
   - For advantageous mutations (s > 0): Larger populations increase fixation probability
   - For deleterious mutations (s < 0): Larger populations decrease fixation probability
   - For neutral mutations (s = 0): Population size has little effect on fixation probability

3. **Initial Frequency Effect**:
   - Higher initial frequencies of type A lead to higher fixation probabilities

4. **Fixation Time**:
   - Larger populations take longer to reach fixation
   - Stronger selection leads to faster fixation

## What Makes This Implementation Robust

1. **Parameter Sweep**: The program can systematically test different combinations of population sizes, selection coefficients, and initial frequencies.

2. **Visualization**: It creates plots showing:
   - Multiple simulation trajectories
   - Comparison between theoretical and empirical fixation probabilities
   - Effects of different parameters

3. **Performance Tracking**: It measures computational performance, showing how simulation time scales with population size.

4. **Detailed Reporting**: The program generates comprehensive reports summarizing the parameter effects.

When discussing with your boss, you can highlight that this simulation demonstrates fundamental principles of evolution, including how natural selection and random drift interact to determine whether beneficial or harmful mutations spread through a population.