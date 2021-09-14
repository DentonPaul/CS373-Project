import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
import matplotlib.patches as patches


def run(X, y, print_info):
    print("=" * 25)
    print("Ten-fold validating KNN with K hyper-parameter:")

    # K Nearest Neighbors K hyper-parameter list

    K_list = [3, 5, 7, 9, 11]

    # Number of k-fold iterations

    K = 10

    # Best K hyper-parameter and error associated

    best_K = 0.0
    best_err = 1.1
    best_acc = 0.0

    if print_info:
        print("=================================")

    # Loop through all K values, try each with k-fold

    for K_neighbor in K_list:
        err, acc = k_fold(K, X, y, K_neighbor)
        if print_info:
            print("K =", K_neighbor, " err =", err, " acc =", acc)
        if err < best_err:
            best_err = err
            best_K = K_neighbor
            best_acc = acc

    if print_info:
        print("=================================")

    print(f"""
        Best K = {best_K}
        Error = {best_err}
        Accuracy = {best_acc}
        """)
    print("ROC images located at: images/tenfoldKNN_ROC/")
    print("=" * 25)


def k_fold(K, X, y, K_neighbor):

    # initialize error and accuracy

    n, d = X.shape
    bs_err = np.zeros(K)
    accuracy = np.zeros(K)

    # Define ROC figure and arrows

    fig1 = plt.figure(figsize=[12, 12])
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(
        patches.Arrow(0.45, 0.5, -0.25, 0.25, width=0.3, color='green', alpha=0.5)
    )
    ax1.add_patch(
        patches.Arrow(0.5, 0.45, 0.25, -0.25, width=0.3, color='red', alpha=0.5)
    )

    # initialize tpr and fpr
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(K):

        # k-fold dataset separation

        lower_range = math.floor((n * i) / K)
        upper_range = math.floor(((n * (i + 1)) / K) - 1)

        test_samples = np.arange(lower_range, upper_range + 1)
        train_samples = np.setdiff1d(np.arange(0, n), test_samples)

        X_train, X_test, y_train, y_test = X[train_samples], X[test_samples], y[train_samples], y[test_samples]

        # Using scikit-learn's KNN algorithm

        alg = KNeighborsClassifier(n_neighbors=K_neighbor)
        alg.fit(X_train, y_train)
        y_predict = alg.predict(X_test)

        # Assess the mistakes made with test samples

        bs_err[i] = np.mean(y_test != y_predict)
        accuracy[i] = metrics.accuracy_score(y_test, y_predict)

        # Compute ROC statistics

        probs = alg.predict_proba(X_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, probs[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)

        # create line for a single fold in ROC chart

        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    save_roc(plt, tprs, mean_fpr, K_neighbor)

    return np.mean(bs_err), np.mean(accuracy)


def save_roc(plt_obj, tprs, mean_fpr, K_neighbor):
    plt_obj.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt_obj.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.2f )' % mean_auc, lw=2, alpha=1)

    plt_obj.xlabel('False Positive Rate')
    plt_obj.ylabel('True Positive Rate')
    plt_obj.title(f'KNN ROC (K={K_neighbor})')
    plt_obj.legend(loc="lower right")
    plt_obj.text(0.32, 0.7, 'More accurate area', fontsize=12)
    plt_obj.text(0.63, 0.4, 'Less accurate area', fontsize=12)
    plt_obj.savefig(f'images/tenfoldKNN_ROC/ROC_K{K_neighbor}')
