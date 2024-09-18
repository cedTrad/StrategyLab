import numpy as np
from sklearn.model_selection import KFold

class PurgedKFold(KFold):
    def __init__(self, n_splits=5, embargo=0):
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        for start, end in test_starts:
            t_ind = np.arange(start, end)
            train_indices = np.concatenate([np.arange(0, start), np.arange(end, len(X))])
            # Embargo applied here
            if self.embargo > 0:
                train_indices = np.setdiff1d(train_indices, np.arange(end, end + self.embargo))
            yield train_indices, t_ind



