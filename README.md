# Concept-Drift-Detection-in-Data-Stream
Initially start by building the classification model in the training phase.  Before making a  prediction  for the unlabeled data, it is passed through concept drift  detection procedure and check how it relates to the current behavior of the model.

Procedure:
At first, find the centroids of all the available classes in the classification model. These centroids, training data with which the model was developed, Test data whose class needs to be predicted and Threshold act as input to the proposed concept drift detection technique. The parameter ‘decision_value’ measures the closeness of the test data with the classes present in the model. It can be calculated as: first  compute the distance of test object from the centroid of the class and then make count of objects which are in vicinity of test object i.e. all those data points (by which the model was trained) whose distance from the centroid is less than the distance between the test object and centroid. Then the ratio of this count and total data points of the class gives the value of ‘decision_value’.
After the ‘decision_value’ values are computed for test object with reference to each class, compare it with the threshold whose value defines the sensitivity of the detection. Smaller the threshold value the more it is sensitive to detect the concept drift. If all the values of ‘decision_value’ corresponding to test object with respect to all the classes are greater than the threshold then it can be concluded that it would lead to concept drift in the model.

Run:
Just have the 'sea.data1.txt' file in the current working directory and run the code.

Results:
Tested on two artificial data sets called SEA and HYPERPLAN and the accuracy of the classification model was compared between the classification accuracy of the model with CDD and other without CDD. Results show that with the help of CDD procedure we can significantly improve the prediction quality of the classification model. 

SEA dataset:
Average accuracy with CD detection was found to be incerased to 98.13% from 82.28%.*

HYPERPLANE dataset:
Average accuracy with CD detection was found to be incerased to 100% from 90.55%.*

*This is the case when the CDD procedure was highly sensitive to detect the slight deviation from the the target features.

# Citation
```
@inproceedings{punn2018testing,
  title={Testing Concept Drift Detection Technique on Data Stream},
  author={Punn, Narinder Singh and Agarwal, Sonali},
  booktitle={International Conference on Big Data Analytics},
  pages={89--99},
  year={2018},
  organization={Springer}
}

```
