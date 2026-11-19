"""Module-level custom components that a fresh process can import."""

import numpy
from sklearn import base


class Centroid(base.ClassifierMixin, base.BaseEstimator):
    """Nearest class mean, with probabilities from a softmax of distances."""

    def fit(self, features, target):
        features = numpy.asarray(features, dtype=float)
        self.classes_ = numpy.unique(target)
        self.means_ = numpy.stack(
            [
                numpy.nanmean(features[numpy.asarray(target) == label], axis=0)
                for label in self.classes_
            ]
        )
        return self

    def predict_proba(self, features):
        features = numpy.nan_to_num(numpy.asarray(features, dtype=float))
        distance = ((features[:, None, :] - self.means_[None, :, :]) ** 2).sum(
            axis=2
        )
        weight = numpy.exp(-(distance - distance.min(axis=1, keepdims=True)))
        return weight / weight.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]
