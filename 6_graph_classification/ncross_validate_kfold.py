from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from grakel import KMTransformer
import numpy as np
from collections.abc import Iterable

from sklearn import metrics
from grakel.graph import is_adjacency as valid_matrix


def check_nfolds(nfolds, y):
    for i, kfolds in enumerate(nfolds):
        for train, test in kfolds:
            if all(y[test] == 1) or all(y[test] == 0):
                return False
    return True


def validate_nfolds(y, kfolder, n_iter):
    while True:
        nfolds = tuple(tuple(kfolder.split(y)) for _ in range(n_iter))
        if(check_nfolds(nfolds, y)):
            print("nfolds validated!")
            return nfolds
        print("kfold with only 1 class")


def Ncross_validate_Kfold_SVM(K, y,
                              n_iter=10, n_splits=10, C_grid=None,
                              random_state=None, scoring="accuracy", fold_reduce=None, arefit='accuracy'):
    # Initialise C_grid
    if C_grid is None:
        C_grid = ((10. ** np.arange(-7, 7, 2)) / len(y)).tolist()
    elif type(C_grid) is np.array:
        C_grid = np.squeeze(C_grid)
        if len(C_grid.shape) != 1:
            raise ValueError(
                'C_grid should either be None or a squeezable to 1 dimension np.array')
        else:
            C_grid = list(C_grid)

    # Initialise fold_reduce:
    if fold_reduce is None:
        fold_reduce = np.mean
    elif not isinstance(callable, fold_reduce):
        raise ValueError('fold_reduce should be a callable')

    # Initialise and check random state
    random_state = check_random_state(random_state)

    # Initialise sklearn pipeline objects
    kfolder = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    estimator = make_pipeline(KMTransformer(), SVC(kernel='precomputed'))

    # Make all the requested folds
    nfolds = validate_nfolds(y, kfolder, n_iter)

    out_acc = list()
    out_f1 = list()
    out_auc = list()
    out_prec = list()
    out_sens = list()
    out_c_sens = list()
    out_c_spec = list()

    for ks in K:
        k_acc = list()
        k_f1 = list()
        k_auc = list()
        k_prec = list()
        k_sens = list()
        k_c_sens = list()
        k_c_sens = list()

        if valid_matrix(ks):
            pg = {"svc__C": C_grid, "kmtransformer__K": [Bunch(mat=ks)]}
        elif isinstance(ks, Iterable) and all(valid_matrix(k) for k in ks):
            pg = [{"svc__C": C_grid, "kmtransformer__K": [
                Bunch(mat=k)]} for k in ks]
        else:
            raise ValueError('Not a valid object for kernel matrix/ces')

        for kfolds in nfolds:
            fold_acc = list()
            fold_f1 = list()
            fold_prec = list()
            fold_sens = list()
            fold_auc = list()
            fold_c_sens = list()
            fold_c_spec = list()

            for train, test in kfolds:
                gs = GridSearchCV(estimator, param_grid=pg, scoring=scoring, refit=arefit,
                                  cv=ShuffleSplit(n_splits=1,
                                                  test_size=0.1,
                                                  random_state=random_state)).fit(train, y[train])
                fold_acc.append(gs.score(test, y[test]))

                predicted = gs.predict(test)

                f1 = metrics.f1_score(y[test], predicted)
                fold_f1.append(f1)

                prec = metrics.precision_score(y[test], predicted)
                fold_prec.append(prec)

                sens = metrics.recall_score(y[test], predicted)
                fold_sens.append(sens)

                tn, fp, fn, tp = metrics.confusion_matrix(
                    y[test], predicted).ravel()
                c_sens = tp / (tp+fn)
                c_spec = tn / (tn+fp)
                fold_c_sens.append(c_sens)
                fold_c_spec.append(c_spec)

                auc = metrics.roc_auc_score(
                    y[test], gs.decision_function(test))
                fold_auc.append(auc)

            k_acc.append(fold_reduce(fold_acc))
            k_f1.append(fold_reduce(fold_f1))
            k_auc.append(fold_reduce(fold_auc))
            k_prec.append(fold_reduce(fold_prec))
            k_sens.append(fold_reduce(fold_sens))
            k_c_sens.append(fold_reduce(fold_c_sens))
            k_c_sens.append(fold_reduce(fold_c_spec))

        out_acc.append(k_acc)
        out_f1.append(k_f1)
        out_auc.append(k_auc)
        out_prec.append(k_prec)
        out_sens.append(k_sens)
        out_c_sens.append(k_c_sens)
        out_c_spec.append(k_c_sens)
    return out_acc, out_f1, out_auc, out_prec, out_sens, out_c_sens, out_c_spec
