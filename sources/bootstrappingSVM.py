import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
import matplotlib.patches as patches


def run(X, y, print_info):
    print("=" * 25)
    print("Bootstrapping SVM with C hyper-parameter:")

    # Support Vector Machine C hyper-parameter list

    C_list = [0.01, 0.1, 1.0, 5.0, 10.0]

    # Number of Bootstrapping iterations

    B = 30

    # Best C hyper-parameter and error/accuracy associated

    best_C = 0.0
    best_err = 1.1
    best_acc = 0.0

    if print_info:
        print("=================================")

    # Loop through all C values, try each with bootstrapping

    for C in C_list:
        err, acc = bootstrapping(B, X, y, C)
        if print_info:
            print("C=", C, ", err=", err)
        if err < best_err:
            best_err = err
            best_C = C
            best_acc = acc

    if print_info:
        print("=================================")

    print(f"""
        Best C = {best_C}
        Error = {best_err}
        Accuracy = {best_acc}
        """)
    print("ROC images located at: images/bootstrappingSVM_ROC/")
    print("=" * 25)


def bootstrapping(B, X, y, C):
    n = len(X)
    bs_err = np.zeros(B)
    accuracy = np.zeros(B)

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

    for b in range(B):
        # Bootstrap dataset separation

        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))

        X_train, X_test, y_train, y_test = X[train_samples], X[test_samples], y[train_samples], y[test_samples]

        # Using scikit-learn's SVM algorithm

        alg = SVC(C=C, probability=True)
        alg.fit(X[train_samples], y[train_samples])
        y_predict = alg.predict(X_test)

        # Assess the mistakes made with test samples

        bs_err[b] = np.mean(y[test_samples] != alg.predict(X[test_samples]))
        accuracy[b] = metrics.accuracy_score(y_test, y_predict)

        # Compute ROC statistics

        probs = alg.predict_proba(X_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, probs[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)

        # create line for a single fold in ROC chart

        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC bootstrap %d (AUC = %0.2f)' % (b, roc_auc))

    save_roc(plt, tprs, mean_fpr, C)

    return np.mean(bs_err), np.mean(accuracy)


def save_roc(plt_obj, tprs, mean_fpr, C):
    plt_obj.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt_obj.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.2f )' % mean_auc, lw=2, alpha=1)

    plt_obj.xlabel('False Positive Rate')
    plt_obj.ylabel('True Positive Rate')
    plt_obj.title(f'SVM + Bootstrapping ROC (C={C})')
    plt_obj.legend(loc="lower right")
    plt_obj.text(0.32, 0.7, 'More accurate area', fontsize=12)
    plt_obj.text(0.63, 0.4, 'Less accurate area', fontsize=12)
    plt_obj.savefig(f'images/bootstrappingSVM_ROC/ROC_C{str(C).replace(".", "")}')
