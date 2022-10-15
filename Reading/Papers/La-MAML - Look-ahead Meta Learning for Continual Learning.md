# La-MAML - Look-ahead Meta Learning for Continual Learning
## Skim Read
### Abstract
* Meta-learning show potential for reducing interference between tasks
* Optimisation-based
* Modulation of per-parameter learning rates (isn't this regularisation?)
* Claims to outperform even replay-based methods

### Methodology
* Has a good explanation on the basic concepts for meta-learning
* Proposes C-MAML which is the base algorithm followed by an extension known as La-MAML
* Follows on from OML paper ([[Meta-Learning Representations for Continual Learning]])
* C-MAML aims to optimise the OML objective online 
* Optimise model's parameters rather than a representation
* Has a replay buffer populated via reservoir sampling
* Sample from buffer and incoming data to form a meta-batch representing samples from current task and old tasks
* Inner-updates use current task's samples
* Outer-update (meta-update) uses the meta-batch
* La-MAML then extends this to further reduce interference between old and new task gradients
* Main point with C-MAML is issue when starting a new task - gradients aren't aligned
* Need meta-updates to be conservative w.r.t to the old tasks to avoid forgetting
* Leads to use of regularisation to limit magnitude and direction of gradient updates
* La-MAML uses a set of learnable per-parameter learning rates used for the inner-updates
* Is a hybrid regularisation, replay and meta-learning approach
* Alleviates the problem with regularisation were it becomes impossible to learn as LRs are reset each meta-update

### Results
* Uses sufficiently complex datasets
* Claim that it outperforms replay-based is wrong as it doesn't compare against the high quality SotA rather historic papers instead
* Well-written methodology though

### Further Action
