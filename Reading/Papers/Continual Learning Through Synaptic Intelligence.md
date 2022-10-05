# Continual Learning Through Synaptic Intelligence
## Skim Read
### Abstract
- Deep learning struggles in domains where data distribution changes
- Contrasts to biological neural networks that continually adapt 
- Aims to bring in some of the biological complexity into the artifical neural networks
- Synapses accumulate task relevant information over time
- Exploit this information rapidly store new memories without forgetting old information
- Evaluated on classification tasks
- Shown to dramatically reduce forgetting while maintaining computational efficiency

### Methodology
- Simple structural regulariser
- Computed online and locally to each synapse
- Each synapse has a measure of its' importance associated with it
- Penalise changes to important parameters to avoid overwriting old memories
- Include this penalisation as a surrogate loss in the main loss function

### Conclusion
- Results apparently comparable to EWC
- Penalising changes to important weights seems to work well
- Regularisation penalty is similar to EWC however it is computed online rather than offline
- Approach requires individual synapses to act as higher dimensional dynamical systems rather than as single scalar synaptic weights
- Synapses accumulate task relevant information during training 
- Neuroscience research suggests that synapses are much more complex than they are modelled as in current NNs
- Montgomery and Madison 2002 suggests synaptic plasticity is dependent on individual histories

### Further Action
- Could this weighting provide some sort of priority to network pruning?
- Could be worth a deeper read into the methodology since the idea of intelligent synapses is interesting