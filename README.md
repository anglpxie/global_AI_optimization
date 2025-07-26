## Wine Classification with Genetic Algorithm with Elitism - Course Global Optimization in AI UNITS

### Executive summary
This project applies a Genetic Algorithm (GA) with elitism to optimize hyperparameters for a neural network (MLPClassifier) tasked with classifying wines based on chemical analysis. The combination of evolutionary computation and neural networks delivers high-accuracy models, demonstrating the power of global optimization strategies in AI applications.

## Project overview

### Objective

- **Classify wine samples into 3 cultivars** using 13 chemical features.
- **Optimize neural network architecture and learning parameters** using a GA with elitism to maximize classification accuracy.
- **Ensure reproducibility and rigorous evaluation** using cross-validation and thorough metric reporting.

### Dataset

- **Source:** [scikit-learn's wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- **Samples:** 178
- **Features:** 13 (alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline)
- **Classes:** 3 wine cultivars (class_0, class_1, class_2)
- **Class distribution:** [59, 71, 48] samples per class

## Methodology

### Data preparation

- Data split into **80% training** (142 samples) / **20% test** (36 samples) sets, stratified to balance classes.
- **StandardScaler** applied: features scaled for better neural network performance.

### Hyperparameter search space

| Hyperparameter         | Range / Options                | Notes                      |
|-----------------------|--------------------------------|----------------------------|
| Hidden layer 1        | 10–100 neurons                 | Integer                    |
| Hidden layer 2        | 10–100 neurons                 | Integer                    |
| Hidden layer 3        | 10–100 neurons                 | Integer                    |
| Alpha (L2 reg.)       | 0.0001–0.1                     | Float                      |
| Learning rate         | 0.001–0.1                      | Float                      |
| Activation function   | relu, tanh, logistic           | Encoded as integer index   |
| Solver                | adam, lbfgs, sgd               | Encoded as integer index   |

### Genetic algorithm design

#### Key concepts

- **Gene:** One hyperparameter (e.g., hidden layer 1 size).
- **Chromosome:** Full list of hyperparameters defining a candidate neural network.
- **Population:** Group of 40 candidate solutions each generation.

#### Steps

1. **Initialization:**  
   - 40 random chromosomes generated, each encoding a full set of hyperparameters.

2. **Fitness Evaluation:**  
   - Each chromosome is decoded into an MLPClassifier.
   - **5-fold cross-validation** on the training data calculates mean accuracy (fitness score).
   - Failed configurations return a low score (0.5) to penalize untrainable/invalid models.

3. **Selection:**  
   - **Elitism:** Top 15% (6 individuals) directly carried into next generation.
   - **Tournament selection:** Random tournaments (size=3) pick parents for reproduction.
   - **Parents can be chosen multiple times; not all individuals become parents.**

4. **Crossover:**  
   - **Blend Crossover (BLX-α)** for continuous parameters (alpha, learning_rate): Creates new values by interpolating between parents with 50% blending parameter.
   - **Uniform crossover** for discrete/categorical parameters: Traditional gene swapping for hidden layer sizes and activation/solver choices.
   - **Crossover rate:** 80%

5. **Mutation:**  
   - **Gaussian mutation** with parameter-specific handling.
   - Each gene in offspring has a **15% mutation rate**.
   - Respects parameter bounds and data types.

6. **New Population Formation:**  
   - New population = 6 elite individuals + 34 offspring (from crossover and mutation).

7. **Diversity & Convergence Tracking:**  
   - Diversity measured as mean standard deviation across all genes: ensures ongoing exploration.
   - Convergence tracked by how many individuals are within 1% of the generation's best score.

8. **Iteration:**  
   - Steps 2–7 are repeated for **50 generations** (fixed stop criterion).

9. **Result Extraction:**  
   - The overall best solution found across all generations is tested on the held-out test data.

## Results

### Optimization Process

- **Population size:** 40 individuals per generation
- **Total evaluations:** 2,000 fitness assessments (40 × 50 generations)
- **Search space:** 753,571 possible hidden layer combinations
- **Average diversity:** 5.302 (healthy exploration throughout evolution)
- **Final diversity:** 1.640 (balanced exploitation while maintaining variety)
- **Best validation accuracy (5-fold CV):** 1.0000 (100.00%)
- **Test set accuracy (unseen data):** 0.9722 (97.22%)
- **Balanced accuracy:** 0.9667 (96.67%)

### Best Neural Network Configuration

| Parameter                 | Value           |
|---------------------------|-----------------|
| Hidden Layers             | (71, 49, 94)    |
| Total Parameters          | ~9,290 weights  |
| Alpha (L2 reg.)           | 0.038929        |
| Learning Rate             | 0.027864        |
| Activation                | ReLU            |
| Solver                    | LBFGS           |

### Model Evaluation (Test Data)
- **Overall Accuracy:** 97.22%
- **Macro F1-Score:** 0.9710
- **Macro Precision:** 0.9778
- **Macro Recall:** 0.9667
- **Weighted F1-Score:** 0.9720
- **Weighted Precision:** 0.9741
- **Weighted Recall:** 0.9722

#### Detailed Classification Report

|           | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 1.0000    | 1.0000 | 1.0000   | 12      |
| Class 1   | 0.9333    | 1.0000 | 0.9655   | 14      |
| Class 2   | 1.0000    | 0.9000 | 0.9474   | 10      |
| **macro avg** | **0.9778** | **0.9667** | **0.9710** | **36** |
| **weighted avg** | **0.9741** | **0.9722** | **0.9720** | **36** |

## Population Dynamics Insights

- **Convergence:**  
  - Algorithm found optimal solution in **Generation 1** (initial best fitness: 1.0000).
  - Final convergence: **87.5%** of population within 1% of best solution.
  
- **Diversity:**  
  - Initial population covers wide solution space.
  - Gradual decrease from ~5.3 to ~1.6, avoiding premature convergence.
  - Maintained healthy exploration throughout all 50 generations.
  
- **Elitism Effectiveness:**  
  - Preserved top 15% (6 individuals) each generation for guaranteed quality.
  - Ensured best configurations weren't lost despite evolutionary randomness.
  
- **Exploration vs. Exploitation:**  
  - Mixed crossover strategy (BLX-α + uniform) optimally handled different parameter types.
  - Gaussian mutation with 15% rate continually introduced beneficial novelty.
  - Tournament selection allowed diverse solutions to contribute to next generations.

## Feature Importance Analysis

**Most Important Features (via permutation importance):**
1. Alcohol content
2. Alcalinity of ash  
3. Proline content

These chemical properties proved most discriminative for wine cultivar classification.

## Technical Achievements

 **Optimization excellence:**
- Achieved 97.22% test accuracy on challenging 3-class classification
- Successfully optimized 7 hyperparameters simultaneously
- Found robust neural network configuration resistant to overfitting

 **Algorithm design:**
- Implemented sophisticated mixed crossover strategy
- Maintained population diversity throughout evolution
- Demonstrated effective elitism without premature convergence

 **Validation Rigor:**
- Reproducible results with fixed random seeds (seed=42)
- 5-fold cross-validation for reliable fitness evaluation
- Real feature importance analysis via permutation testing

 **Performance monitoring:**
- Complete evolution tracking (fitness, diversity, convergence)
- Detailed population dynamics analysis
- Visualization of algorithm behavior and results

## Conclusion

This Genetic Algorithm with Elitism successfully demonstrates the effectiveness of evolutionary computation for automated hyperparameter optimization in machine learning. The algorithm achieved exceptional performance (97.22% test accuracy) while maintaining healthy population dynamics and avoiding common pitfalls like premature convergence.

**Key Success Factors:**
- **Parameter-specific strategies:** Different crossover/mutation approaches for continuous vs. discrete parameters
- **Balanced selection pressure:** Tournament selection with elitism preservation
- **Comprehensive evaluation:** 5-fold CV with multiple performance metrics
- **Diversity maintenance:** Ongoing monitoring and healthy exploration-exploitation balance

The project showcases how evolutionary algorithms can effectively navigate complex hyperparameter spaces to discover high-performing neural network configurations for real-world classification tasks.
=======
# anglpxie-global_AI_optimization
Exam project for Global Optmization in AI 
>>>>>>> 366b644abec149ac0ae62064f201698581e74994
