# Learning to Learn with Feedback and Local Plasticity
## Skim Read
### Abstract
- Employs meta-learning to discover networks that learn using feedback connections and local, biologically inspired learning rules
- Feedback connections are not tied to the feedforward weights
- Avoids biologically implausible weight transport
- Matches or exceeds SotA gradient-based online meta-learning algorithms on classification problems
- Suggest the existence of a class of biologically plausible learning mechanisms which match the performance of, and overcome the limitations of, gradient descent-based learning

### Methodology
- Apply local plasticity rules in a neural network to update feedforward synaptic weights
- Endow the network with feedback connections that propagate information about target outputs to upstream neurons in order to guide this plasticity
- Employ meta-learning to optimise feedback weights, feedforward weight initialisations, and rates of synaptic plasticity.
- Uses meta-learning which is where the network's learning procedure is itself learned in an "outer loop" of optimisation
- Propagates the prediction error through a set of feedback weights
- Better performance when using raw targets over prediction errors for the classification problem
- Activiations of each layer are updated in response to the feedback
- Also has another update rule for specific layer??
- Network then undergoes synaptic plasticity using a local learning rule where the synaptic weight is updated based only on its existing value, the presynaptic activity and the postsynaptic activity resulting from feedback using Oja's learning rule
- Only allow plasticity in the final N network layers so initial layers are fixed feature extractors
- Can be shown that a sufficiently wide and deep neural network can approximately any learning algorithm using the described learning procedure
- The meta-learned parameters are the initialisation weights, feedback weights and plasticity coefficient and beta for each layer (?)

### Results
- Outperforms gradient-based methods
- Doesn't compare to other types such as rehearsal-based which are SotA
- Seems to set out a bit of a proposal that this is a space worth exploring
- Would have been good to see some more in-depth results
- Used Mini-ImageNet and Omniglot which are decent at least