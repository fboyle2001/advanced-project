# End of Week 8
- Lots of techniques implemented

## Literature Review
- Feedback

## Implementations
### Learning to Prompt
- Pretrained Vision Transformer backbone (checked, this is pretrained on ImageNet)
- Define a set of prompts (used 10 prompts)
- Each prompt is of the dimension L_p x Feature Dimension (5 x 768 used)
- Define a set of key-value pairs, each key links to a prompt
- Each key is 1 x Feature Dimension (1 x 768 used)
- Each key and prompt are optimisable
- For each batch of samples:
    - Compute the feature embedding of the sample (1+196 x 768)
    - Select the N nearest keys (N = 5) by distance between feature embedding of sample and keys
    - Select the corresponding prompts
    - Prepend the prompts to the sample feature embedding (1+221 x 768)
    - Feed the prepended embedding into the transformer's encoder (essentially everything up to the classification head)
    - Take the first 1+NL_p tokens
    - Pass these to a single MLP classification layer (consisting of AvgPool -> Linear)
    - Calculate CE loss and Key Loss (i.e. distance between key and the image features, goal is to bring them closer)
    - Optimise keys, prompts, and classification MLP
- Results:
    - Very promising technique, achieved accuracy of 67.7%
    - Quite imbalanced though
    - Translated well to CIFAR-100 (runtime was ~1h15m) average accuracy 61.58%
    - Accuracies: 18.01%, 31.87%, 43.40%, 53.09%, 61.58%
    
### Supervised Contrastive Replay
- Instead of CE loss, use Supervised Contrastive Loss
- Goal is to bring similar label samples together in the feature space and put others far away
- For each batch:
    - Draw 100 samples from memory
    - Stack the samples and the batch
    - Duplicate and augment the stacked (so have 200 + 2B samples)
    - Compute the feature embeddings
    - Compute the SCLoss
- To classify:
    - Compute the mean feature embedding vector for each class using memory samples
    - Compute the feature embedding of the sample
    - Classify as the closest mean feature embedding vector
- Results: 
    - Average accuracy on CIFAR-10 was 61.8%
    - Didn't test on CIFAR-100 yet

### Dark Experience Replay (DER / DER++)
- 3 loss terms
- Designed as a simple baseline (similar to GDumb's purpose)
- Instead of storing just (x, y) we store (x, y, logits using model at time of storage)
- For each sample:
    - Draw 2 separate (potentially overlapping) batches from memory (call them A and B)
    - Augment the batch and both memory batches
    - Compute the logits on all of them (i.e. feed into the model)
    - Compute the CE loss on the actual batch
    - Compute the MSE Loss between the stored logits of A and the current model logits of A
    - Compute the CE loss of the labels of B and the current predictions of B (DER excludes this loss)
    - Sum these up and backprop through
- Results:
    - DER++: 46.8%
    - DER: 34.5%
    - DER++ outperforms before the CE loss acts as a regularising term on the output

## CIFAR-100 Experiments
- GDumb: 35.58% (~21 minutes) (5000 memory samples)
- DER: 3.38% (~7 minutes) (5000 memory samples)
- DER++: (5000 memory samples)
- L2P: 61.58% (runtime was ~1h15m) (10 prompts, 5 selected per sample)
    - Quite a bit lower than the actual implementations, think it is likely down to hyperparameters
- Rainbow: Took too long to run last night, probably an overnight job I expect

## Novel Experimentation
- Next week I want to bring concrete plan for experimentation to do over the break
- Realistically, the future of CL has to be using pre-trained models as the backbone
- They're readily available
- The idea of selecting uncertain samples is seen in SCR and Rainbow which are good techniques
- Can extract features from the pretrained backbone
- Need some way to use these features to determine how uncertain the model is about a sample
- Will research this

## Plans
- Don't think I need to implement any more new literature techniques
- Change EWC to EWC++ which is a better online version
- Maybe implement generic episodic replay which is very basic baseline
    - Instead of storing the samples and training at the end like GDumb, train as we save and mix with the memory samples 
- Metrics:
    - Average forgetting: decrease in accuracy of task i evaluated at task i compared with accuracy evaluated at task j
    - Need to clean up the graphs, especially for CIFAR-100
    - For individual graphs, group classes by task instead
    - Can look at top-1 accuracy, not sure it's a very good metric though (look at the L2P graph!)
    - Better visualisation will be like in the literature where they are all on a line graph together of course
- More CIFAR-100 evaluation
- Setup a better evaluation framework now I'm out of initial implementation phase