# CS 373 Project

## Report Link to Google Docs

https://docs.google.com/document/d/1rUu1db1OwYbn5zadc1HH7aAy-Js6BcN-ZGa27DQFMzk/edit?usp=sharing

## Project Setup
One can run any of the algorithms individually by running the following format in the terminal:

`python main.py *algorithm*`

Where `*algorithm*` is replaced by the following short-hand expressions for each algorithm.

| Short Hand | Algorithm |
| ---------- | --------- |
| bsknn      | Bootstrapping - K Nearest Neighbors |
| bssvm      | Bootstrapping - Support Vector Machine |
| tfknn      | Ten-Fold Cross Validation - K Nearest Neighbors |
| tfsvm      | Ten-Fold Cross Validation - Support Vector Machine |
| bs         | Both Bootstrapping Algorithms |
| tf         | Both Ten-Fold Cross Validation Algorithms |
| knn        | Both K Nearest Neighbors Algorithms |
| svm        | Both Support Vector Machine Algorithms |
| all        | All 4 algorithms |
| datainfo   | Prints dataset |

So if I wanted to run all the SVM algorithms, I'd type in the terminal:

`python main.py svm`

And all the SVM algorithms will run. Each short-hand will give a time estimate for how long it will take to run all the algorithms requested. They can also be chained as such: 

`python main.py bsknn bssvm tfsvm datainfo`

## Project TODOS

- [X] Initial Project Setup
- [X] Data Extraction
- [X] SVM Bootstrapping
- [X] KNN Bootstrapping
- [X] SVM Ten-Fold
- [X] KNN Ten-Fold
- [X] ROC/Accuracy for SVM Bootstrapping
- [X] ROC/Accuracy for SVM Ten-Fold
- [X] ROC/Accuracy for KNN Bootstrapping
- [X] ROC/Accuracy for KNN Ten-Fold
- [ ] Conclusions

## Project Info

1.	Team number 
      
    4
2.	Students’ name and Purdue e-mail

    Cade Henschen (chensche@purdue.edu)
    Claudia Duncan (dunca106@purdue.edu)
    Denton Paul (paul96@purdue.edu)
3.	Definition of the problem

    Based on a set of component principles (acidity, density, sugar, etc.) is the wine able to be classified as quality (that being, rated above >= 7 on a 1 to 10 scale)?
4.  Which dataset will be used?
    

    We are using a dataset which has a series of quantitative attributes of each type of wine (such as the pH, density, etc.) and its associated quality rating. The data we're using has about 1600 samples and 11 features per sample. We plan to transform the dataset in only one way - that being to transform the quality rating to a binary state, which means to change any quality rating greater or equal to 7 to a 1 (indicating to be labeled quality) or a 0 if not (indicating to be labeled not quality). Overall, all features seem important, and doesn't create too much complexity in its current state.
5.  URL where the above dataset is available.


    https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv
6.  Which TWO machine learning algorithms are going to be used?


    The two algorithms we choose to use were SVM and K nearest neighbors. The reason we decided to use SVM was to see if there is any degree of linear (or semi linear) separation of certain traits that cause wine to lean more quality over not - SVM allows us to use a more linear separation of that algorithm while also using concepts like margin and weighted miscoring to generalize the division from overfitting. We chose KNN to see if wines of similar characteristics and features were similarly classified as quality or not.
7.  Which cross-validation technique(s) is(are) going to be used?


    For our cross-validation, we choose to use 10-fold and bootstrapping as our methods of choice. 10-fold sees how well the algorithm generalizes against itself and to see if the associations with quality and not have a strong enough separation (through both our KNN and SVM algorithms) to allow accurate individual testing of data that was once a part of it. We chose 10-fold so k = 10; largely because this dataset is massive for LOOCV. We also choose bootstrapping to help create random datasets to test accuracy at a larger scale than with k-fold. The number of bootstraps we chose were B = 30
8.  Which hyper-parameter(s) is(are) going to be tuned.


    For SVM, we are going to concentrate on C as our hyperparameter - which influences the scoring of misclassified points when creating the margin. While trivially it seems like a large value of C (heavily siding with no or few misclassifications) would be the best approach, we do not want to veer into overfitting and causing an incredibly thin margin because of that. Testing a range of C allows us to classify that happy medium between an appropriate weight/margin and an appropriate generalization. For KNN we are choosing K, or the number of nearest neighbors, as our parameter. K allows us to generalize the number of similar wines to look into when concluding our current test point, which can allow for more generalization.
9.  Which TWO experimental results will you show?


    Using ROC curves and plotting the accuracy of subset data, we'll be able to analysis the experimental result of our algorithms in a more robust and clear way. ROC curves allow us to see how far beyond a random classifier the one we landed on is. While we will likely never create a perfect classifier, having a high true positive rate and minimal false positive rate would greatly aid in reinforcing the predictions of the model. ROC best helps us understand if the model even works at all from how it compares to random assignment. Calculating the accuracy in subsets of data allows us to see if there is a consistent generalization to our model, or if random sampling and k-fold simply got lucky at a smaller scale of testing. Accuracy will also help us understand if generalizing occurs or if the rigidity of the algorithm largely comes from overfitting.
10. Which programming language are you going to use?


    Python
