# linear_classifier
CMPSC165B Homework 2 Programming Assignment

This 3-class basic linear classifier was implemented using a one-versus-rest approach. It takes 3 dimensional data, and it uses the data to calculate the decision boundaries against each class. The decision boundary is found by calculating the centroid of each class, and the finding the orthogonal bisector of the two centroids.

This program outputs some accuracy metrics about the model when testing against the provided testing data.

To run the program, do
```python hw2.py <training data> <testing data>```