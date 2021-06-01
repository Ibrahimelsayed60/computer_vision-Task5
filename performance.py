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


def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix


def mae_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
	return sum_error / float(len(actual))

def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)


# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


######## Test ###########################
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# #import pandas as pd
# #import numpy as np
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt

# X, y = make_classification(n_samples=1000, n_informative=10, n_features=20, flip_y=0.2)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# prob_vector = model.predict_proba(X_test)[:, 1]


# ROC = roc_from_scratch(prob_vector,y_test,partitions=10)
# plt.plot(ROC[:,0],ROC[:,1],color='#0F9D58')
# plt.title('ROC Curve',fontsize=20)
# plt.xlabel('False Positive Rate',fontsize=16)
# plt.ylabel('True Positive Rate',fontsize=16)
# plt.show()



# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(y_test, prob_vector)

# plt.figure(figsize=(15, 7))
# plt.scatter(fpr, tpr, s=100, alpha=0.5, color="blue", label="Scikit-learn")
# plt.scatter(
#     ROC[:, 0], ROC[:, 1], color="red", s=100, alpha=0.3, label="Our implementation"
# )
# plt.title("ROC Curve", fontsize=20)
# plt.xlabel("False Positive Rate", fontsize=16)
# plt.ylabel("True Positive Rate", fontsize=16)
# plt.legend()
# plt.show()
# #### 1. Performance test

# # partitions = 100
# # ROC = roc_from_scratch(prob_vector, y_test, partitions=partitions)
# # fpr, tpr = ROC[:, 0], ROC[:, 1]
# # rectangle_roc = 0
# # for k in range(partitions):
# #         rectangle_roc = rectangle_roc + (fpr[k]- fpr[k + 1]) * tpr[k]
# # rectangle_roc


# #### 2. ROC curve Test
# # import seaborn as sns
# # sns.set()
# # plt.figure(figsize=(15,7))

# ROC = roc_from_scratch(prob_vector,y_test,partitions=10)
# plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=100)
# plt.title('ROC Curve',fontsize=20)
# plt.xlabel('False Positive Rate',fontsize=16)
# plt.ylabel('True Positive Rate',fontsize=16)


