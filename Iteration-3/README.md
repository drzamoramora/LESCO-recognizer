# LESCO-Iteration-3
This iteration, takes the code from Iteration 1 and replaces PCA with TruncateSVD and t-SNE to check if there is an improvement in accuracy in the test set.

## Results

Iteraiton Results:
- Iteration 1 Manhattan test set Accuracy: 77%
- Iteration 3 Manhattan test set Accuracy (t-SNE): 5%
- Iteration 3 Manhattan test set Accuracy (TruncateSVD)): **91%**

# Summary

Classification accuracy was greatly improved with the replacement of PCA with TruncateSVD. 

Other big improvements using TruncateSVD:
- Cosine: 79%
- Euclidean: 82%
- Minkowski: 82%
