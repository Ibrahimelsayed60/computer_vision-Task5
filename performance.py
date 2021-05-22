import typing_extensions
import pandas as pd
import numpy as np


def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


def roc_from_scratch(probabilities, y_test, partitions=100):
    roc = np.array([])
    for i in range(partitions + 1):
        
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)




######## Test ###########################


#### 1. Performance test

# partitions = 100
# ROC = roc_from_scratch(prob_vector, y_test, partitions=partitions)
# fpr, tpr = ROC[:, 0], ROC[:, 1]
# rectangle_roc = 0
# for k in range(partitions):
#         rectangle_roc = rectangle_roc + (fpr[k]- fpr[k + 1]) * tpr[k]
# rectangle_roc


#### 2. ROC curve Test
# import seaborn as sns
# sns.set()
# plt.figure(figsize=(15,7))

# ROC = roc_from_scratch(prob_vector,y_test,partitions=10)
# plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=100)
# plt.title('ROC Curve',fontsize=20)
# plt.xlabel('False Positive Rate',fontsize=16)
# plt.ylabel('True Positive Rate',fontsize=16)


