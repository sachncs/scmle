"""A stand-in for the `tabpfn` package, for testing the adapter plumbing.

It mimics the parts of the real interface the framework uses: classes with
`fit`, `predict`, `predict_proba` and `classes_`, built from simple models.
It is not TabPFN and proves nothing about its accuracy.
"""

from sklearn import linear_model


class TabPFNClassifier(linear_model.LogisticRegression):
    """Logistic regression behind TabPFN's constructor."""

    def __init__(self, device="cpu", random_state=0, n_estimators=8):
        super().__init__(max_iter=500, random_state=random_state)
        self.device = device
        self.n_estimators = n_estimators


class TabPFNRegressor(linear_model.Ridge):
    """Ridge regression behind TabPFN's constructor."""

    def __init__(self, device="cpu", random_state=0, n_estimators=8):
        super().__init__(random_state=random_state)
        self.device = device
        self.n_estimators = n_estimators
