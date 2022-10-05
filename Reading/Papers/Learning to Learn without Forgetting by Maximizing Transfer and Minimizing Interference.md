# Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference
### Abstract
- Proposes a new concept of temporally symmetric trade-off between transfer and interference that can be optimised by enforcing gradient alignment across examples
- Proposes a new algorithm using this called Meta-Experience Replay (MER) which combines experience replay with optimisation based meta-learning
- Learns parameters that reduce the chance of interference in future gradients while increasing transfer on future gradients
- Results show that MERs performance increases (while others decrease) in scenarios where the environment is more non-stationary and a smaller memory buffer is used

### Transfer-Interference Trade-Off
- Parameters T and loss L
- Transfer and Interference are opposites 
- Quantified by partial derivatives of L w.r.t T and dot producting between samples, J
- If J < 0 then we have interference and learning the new sample B will negatively affect the known sample A
- If J > 0 then we have transfer and learning the new sample B will positively affect the known sample A
- Goal is to minimise interference projecting backwards in time which has generally been achieved by reducing weight sharing
- Past solutions ignore temporal complexity and lump old memories together and new data together in two groups
- As we have an uncertain future of samples in CL it is important that while we limit the interference affecting the past samples we must not hinder learning of new samples in the future
- Weight sharing across examples arises both backwards and forwards in time
- Weight sharing across examples that enables transfer to improve future performance must not disrupt performance on what has already been seen
- Adopts a meta-learning perspective on CL, want to learn examples in a way that generalises to others in the overall distribution

### Methodology
- Ideally want to maximise dot product between gradients so that similar samples in the network share parameters where gradient aligns. Otherwise they keep separate
- Uses an experience replay
- Requires second order derivatives which are approximated using first-order Taylor expansion
- Use a modified version of the meta-learning algorithm called Reptile

### Results
- Solid results although doesn't compare to too many different algorithms in the area
- Is quite memory efficient but the results are a bit confusing 
- Would have been more useful if results on CIFAR-100 etc were included