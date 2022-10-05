# Learning without Forgetting
## Skim Read
### Abstract
- Problems arise when adding new capabilities to a CNN but training data for its existing capabilities are unavailable
- Proposes using only new task data to train the network while preserving original capabilities
- May be able to replace fine-tuning with similar old and new datasets for improved new task performance

### Methodology
- CNN with parameters \theta_s and task-specific parameters \theta_n
- Want to learn parameters \theta_n that work for the new and old task without access to previous data
- Nodes added to output layer for each new class - these are fully connected to the previous layer with randonly initialised weights
  - Also gives a suggestion for the number of new parameters to introduce (see paper)
- Train to minimise loss over all tasks and some regularisation using SGD
- Freeze \theta_s and \theta_o and train \theta_n to convergence (warm-up step)
- Then jointly train all \theta until convergence (joint-optimisation)
- Warm-up is needed to enhance fine-tuning of old task performance
- Instead of using the old training data we use the old output from the network to train on
  - Surely this becomes a signficant source of error?

### Conclusion
- LwF is a hybrid approach of knowledge distillation and fine-tuning
- Learns parameters that are discriminative for the new task while preserving outputs for the original tasks
- Has results on classification tasks
- Is feasible for expanding a network to the set of possible predictions on an nexisting network without the need to have access to the prior training data
- Looking solely at performance of the task this method outperforms fine-tuning when this was written
  - Fine-tuning approaches use a low learning rate in hope of finding parameters whch settle to "good" local minima
- Preserving outputs on previosu tasks is more direct and interpretable way to retain shared structures

### Further Action
- May be worth a deeper read, is a fairly old paper though so worth checking if there are works that have improved on this more recently
- Mentions another approach Less Forgetting Learning (Jung et al. 2016) but is also quite an old approach
