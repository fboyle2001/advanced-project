# Memory Aware Synapses: Learning what (not) to forget
## Skim Read
### Abstract
- Argues that due to limited model capacity and unlimited new knowledge, existing knowledge has to be preserved or erased selectively
- Computes importance of parameters of a neural network in an unsupervised and online manner
- Changes to important parameters can be penalised when learning a new task to preserve existing knowledge
- Shows SotA performance at time of release

### Methodology
- Estimate an importance weight for each parameter in the network
- The importance weights approximate the sensitivity of the learned function to a parameter change rather than a measure of the parameter uncertainty (or inverse of for importance)
- In addition to the task loss we have a regulariser that penalises changes to parameters deemed important using the approximated metric
- Regularisation weight controlled by a hyper-parameter
- Hebbian learning: Cells that fire together, wire together
- Resembles an implicit memory for each parameter of the network
- Proposes both a local and global version of the idea
- Uses pretrained AlexNet on ImageNet as backbone

### Results
- Outperformed LwF, EWC and SI
