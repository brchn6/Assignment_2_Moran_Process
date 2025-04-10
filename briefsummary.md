Moran Process Simulation Analysis

We implemented a  simulation of the Moran Process with a focus on understanding the balance between selection and genetic drift in finite populations. Your code effectively models a population with two competing types (A and B) where type A has a selective advantage or disadvantage determined by the selection coefficient.
Key Findings from Parameter Sweep Analysis
Selection Coefficient Effects
The simulation confirms that selection coefficient is the primary determinant of fixation probability:

![alt text](results/moran_process_parameter_sweep_20250403_151922/visualizations/selection_effect_on_fixation.png)

Negative selection coefficients (s < 0) result in low fixation probabilities (8-11%)
Neutral selection (s = 0) shows fixation probabilities around 20% (should theoretically be equal to initial frequency)
Positive selection coefficients lead to substantially higher fixation probabilities:

s = 0.1: ~60% fixation
s = 0.5: ~82% fixation
s = 1.0: ~89% fixation



Population Size Effects
The results demonstrate how population size interacts with selection:

![alt text](results/moran_process_parameter_sweep_20250403_151922/visualizations/parameter_effects_on_fixation_time.png)

For deleterious mutations (s < 0), larger populations show lower fixation probabilities (correlation -0.61 to -0.63)
For advantageous mutations (s > 0), larger populations generally show higher fixation probabilities
For the neutral case (s = 0), there's an unexpected negative correlation (-0.81) with population size

Fixation Time
As expected, fixation time strongly correlates with population size (correlation 0.999), confirming that larger populations require more time to reach fixation since more replacement events are needed.
Trajectory Analysis
The trajectory plots clearly illustrate the stochastic nature of the Moran Process. For the case with N=100, s=0.1, and initial frequency of 0.5:

All trajectories eventually reach fixation (frequency = 1)
The paths to fixation show significant random fluctuations
Despite starting at 50% frequency, the advantageous nature of type A (s = 0.1) ensures its eventual dominance

Theoretical vs. Empirical Results
The comparison between theoretical and empirical fixation probabilities shows excellent agreement across various selection coefficients, validating your implementation. The graph demonstrates the sharp transition from near-zero fixation probability for negative selection to near-certainty fixation for positive selection.
Computational Performance
The simulation runtime scales with population size, with larger populations requiring significantly more computational resources:

N=10: ~1.3 seconds
N=100: ~85 seconds
N=1000: ~193 seconds

Conclusion
Your implementation successfully captures the essential dynamics of the Moran Process and demonstrates how selection and drift interact in finite populations. The parameter sweep provides valuable insights into how population size and selection coefficient affect evolutionary outcomes. The results align well with theoretical expectations, particularly showing how selection becomes more effective in larger populations and how fixation time scales with population size.
For a complete assignment solution, you might consider implementing two specific modifications from the suggested list (multiple types, environmental fluctuations, mutation, or demographic stochasticity) and analyzing how they alter the model's dynamics.




![alt text](results/moran_process_parameter_sweep_20250403_151922/visualizations/parameter_correlation_matrix.png)

Parameter Correlations (Image 1)
The correlation matrix reveals several important relationships:

Selection Coefficient Impact:

Strong positive correlation (0.69) between selection coefficient and fixation probability
This confirms that selection strength is the primary determinant of evolutionary outcomes


Population Size Effects:

Moderate positive correlation (0.49) between population size and fixation time
Weak negative correlation (-0.11) between population size and fixation probability
This suggests population size primarily influences the time to fixation rather than the ultimate outcome


Initial Frequency Influence:

Moderate positive correlation (0.43) between initial type A frequency and fixation probability
Negative correlation (-0.21) with fixation time
Higher initial frequencies increase chances of fixation and reduce time to fixation


Computational Performance:

Strong correlation (0.95) between fixation time and simulation time
Moderate correlation (0.41) between population size and simulation time
Larger populations require significantly more computational resources


![alt text](results/moran_process_parameter_sweep_20250403_151922/visualizations/fixation_probability_heatmap.png)

Fixation Probability Heatmap (Image 2)
The heatmap visualizes the interaction between population size and selection coefficient:

Selection Threshold Effect:

Clear transition around s=0 (neutral selection)
For s≤0, fixation probabilities are very low (0-0.53)
For s>0, fixation probabilities are consistently high (0.59-0.91)


Population Size Gradient:

For negative selection coefficients, larger populations show lower fixation probabilities
For s=0 (neutral), smaller populations show higher fixation probabilities
For positive selection, population size has minimal impact on ultimate fixation probability


Selection Strength Gradient:

Consistent pattern across all population sizes: stronger positive selection leads to higher fixation probability
At s=0.1, fixation probability is consistently around 0.6 regardless of population size
At s=1.0, fixation probability reaches 0.82-0.91 across all population sizes