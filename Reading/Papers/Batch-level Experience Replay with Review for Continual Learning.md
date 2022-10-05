# Batch-level Experience Replay with Review for Continual Learning
## Skim Read
### Abstract
- Challenge at CVPR2020, achieved 1st place 
- CORe50 dataset
- This paper introduces the review trick seen in the empirical survey

### Methodology
- Based on Experience Replay
- Adds a review step before the final testing to remind the model of knowledge learned during training
- Review step uses a lower learning rate to prevent overfitting
- Like most other CL algorithm uses a pre-trained model on ImageNet
- Treats the CL problem as solving the Transfer-Interference Trade-Off
- Reduces number of memory retrieval and update steps by retrieving samples when a new batch of data arrives and updating the memory after training the current batch

### Results
- Won the CVPR2020 challenge
- Attains high final val accuracy but struggles on average val accuracy like many other CL algorithms

