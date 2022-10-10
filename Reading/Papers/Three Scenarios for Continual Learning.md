# Three Scenarios for Continual Learning
## Skim Read
### Abstract
* Highlights the problem in evaluation of continual learning across papers 
* Sets out 3 scenarios based on test time task identity 
* Only uses MNIST still though
* Demonstrates substantial differences between scenarios in difficulty
* Finds rehersal-based methods are essentially needed compared to regularisation

### Methodology
* Includes "well-documented and easy-to-adapt code" which might be useful
* Focus on the problem where a single neural network model needs to sequentially learn a series of tasks 
* Found that the level of difficulty can change drastically based on if the task identity is available at inference time
* Easiest scenario is Task-IL where models are told which task they need to perform
* Domain-IL task identity is not available, only need to solve the task at hand
	* Tasks have the same task structure but different input distributions
* Class-IL is the hardest and most real-world where a model must solve each task seen so far and infer which task they are presented with
* Notes that this isn't the absolute hardest but some structure is needed or this will be too difficult
* Makes the case for MNIST permuted, might be worth while

### Results
* Quite an early paper so doesn't include the best rehersal methods
* Regularisation is pretty much useless on Class-IL and falls off in Domain-IL
* Uses DGR might be worth looking in to?
* Notes that MNIST is too easy and not that realistic but still proves the point of the paper

### Further Action
* Seems to provide a much solider backing to why Task-IL v Class-IL is a thing
* Useful paper for setting the groundwork
