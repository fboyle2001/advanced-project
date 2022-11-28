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

### Dark Experience Replay (DER++)
- 3 loss terms
- Designed as a simple baseline (similar to GDumb's purpose)
- Instead of storing just (x, y) we store (x, y, logits using model at time of storage)
- For each sample:
    - Draw 2 separate (potentially overlapping) batches from memory (call them A and B)
    - Augment the batch and both memory batches
    - Compute the logits on all of them (i.e. feed into the model)
    - Compute the CE loss on the actual batch
    - Compute the MSE Loss between the stored logits of A and the current model logits of A
    - Compute the CE loss of the labels of B and the current predictions of B
    - Sum these up and backprop through
- Results:
    - Not great but too be expected

## CIFAR-100 Experiments

## Plans