# computer_vision-Task5

## ROC Curve

The *Receiving operating characteristic (ROC)* graph attempts to interpret how good (or bad) a binary classifier is doing. There are several reasons why a simple confusion matrix isn’t enough to test your models. Still, the ROC representation solves incredibly well the following: *the possibility to set more than one threshold in one visualization*.

### Steps:

#### 	1- Choosing a threshold: 

   * The ROC curve’s whole idea is to check out different thresholds.

   * There are different ways to do it, but we will take the simplest. Just by setting the thresholds into equally distant partitions.

     Note: The thresholds that we need to look at are equal to the number of partitions we set, plus one. We will iterate over every threshold defined in this step.

     #### 2- Threshold Comparison:

     * In every iteration, we must compare the predicted probability against the current threshold. 
     * If the threshold is higher than the predicted probability, we label the sample as a 0, and with 1 on the contrary.

     #### 3- Calculating TPR and FPR:

     * calculate the TPR and FPR at every iteration; It’s precisely the same we saw in the last step.
     * The only difference is that we need to save the TPR and FPR in a list before going into the next iteration. The list of TPRs and FPRs pairs is the line in the ROC curve. 

The core of the algorithm is to iterate over the thresholds defined in step 1. We go through steps 2 & 3 to add the TPR and FPR pair to the list at every iteration.