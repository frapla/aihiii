# if not other indicated, the code is from scikit-learn 1.5.1 .venv/lib/python3.11/site-packages/sklearn/multioutput.py

from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.utils import Bunch
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, _fit_context, clone
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping, _routing_enabled, process_routing
from sklearn.utils._response import _get_response_values
from sklearn.utils.validation import _check_response_method, check_is_fitted

LOG: logging.Logger = logging.getLogger(__name__)


class _BaseChainOwn(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "base_estimator": [HasMethods(["fit", "predict"])],
        "order": ["array-like", StrOptions({"random"}), None],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
    }

    def __init__(self, base_estimator, *, n_time_steps: int = 1, order=None, cv=None, random_state=None, verbose=False):
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.n_time_steps = n_time_steps

    def _log_message(self, *, estimator_idx, n_estimators, processing_msg):
        if not self.verbose:
            return None
        return f"({estimator_idx} of {n_estimators}) {processing_msg}"

    def _get_predictions(self, X, *, output_method):
        """Get predictions for each model in the chain."""
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_output_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_feature_chain = np.zeros((X.shape[0], len(self.estimators_)))

        # `RegressorChain` does not have a `chain_method_` parameter so we
        # default to "predict"
        chain_method = getattr(self, "chain_method_", "predict")
        hstack = sp.hstack if sp.issparse(X) else np.hstack
        for chain_idx, estimator in enumerate(self.estimators_):
            # change: add range
            chain_range = range(self.n_time_steps * (chain_idx // self.n_time_steps), chain_idx)
            LOG.debug("Chain range for prediction position %s: %s", chain_idx, list(chain_range))
            previous_predictions = Y_feature_chain[:, chain_range]
            # if `X` is a scipy sparse dok_array, we convert it to a sparse
            # coo_array format before hstacking, it's faster; see
            # https://github.com/scipy/scipy/issues/20060#issuecomment-1937007039:
            if sp.issparse(X) and not sp.isspmatrix(X) and X.format == "dok":
                X = sp.coo_array(X)
            X_aug = hstack((X, previous_predictions))

            feature_predictions, _ = _get_response_values(
                estimator,
                X_aug,
                response_method=chain_method,
            )
            Y_feature_chain[:, chain_idx] = feature_predictions

            output_predictions, _ = _get_response_values(
                estimator,
                X_aug,
                response_method=output_method,
            )
            Y_output_chain[:, chain_idx] = output_predictions

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_output = Y_output_chain[:, inv_order]

        return Y_output

    @abstractmethod
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        # changed: fix order type
        self.order_ = np.array(range(Y.shape[1]))

        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        LOG.debug("Fitting RegressorChain with %s esimators", len(self.estimators_))

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch(estimator=Bunch(fit=fit_params))

        if hasattr(self, "chain_method"):
            chain_method = _check_response_method(
                self.base_estimator,
                self.chain_method,
            ).__name__
            self.chain_method_ = chain_method
        else:
            # `RegressorChain` does not have a `chain_method` parameter
            chain_method = "predict"

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            # changed: multichannel time series
            chain_range = range(self.n_time_steps * (chain_idx // self.n_time_steps), chain_idx)
            x_aug = np.hstack((X, Y[:, chain_range]))
            LOG.debug("Chain range for prediction position %s: %s-%s", chain_idx, x_aug.shape, y.shape)
            estimator.fit(x_aug, y, **routed_params.estimator.fit)

        return self

    def predict(self, X):
        """Predict on the data matrix X using the ClassifierChain model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        return self._get_predictions(X, output_method="predict")


class MultichannelRegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChainOwn):
    """A multi-label model that arranges regressions into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    Read more in the :ref:`User Guide <regressorchain>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the regressor chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is 'random' a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        If True, chain progress is output as each model is completed.

        .. versionadded:: 1.2

    Attributes
    ----------
    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : Equivalent for classification.
    MultiOutputRegressor : Learns each output independently rather than
        chaining.

    Examples
    --------
    >>> from sklearn.multioutput import RegressorChain
    >>> from sklearn.linear_model import LogisticRegression
    >>> logreg = LogisticRegression(solver='lbfgs')
    >>> X, Y = [[1, 0], [0, 1], [1, 1]], [[0, 2], [1, 1], [2, 0]]
    >>> chain = RegressorChain(base_estimator=logreg, order=[0, 1]).fit(X, Y)
    >>> chain.predict(X)
    array([[0., 2.],
           [1., 1.],
           [2., 0.]])
    """

    @_fit_context(
        # RegressorChain.base_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method at each step
            of the regressor chain.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().fit(X, Y, **fit_params)
        return self

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.base_estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def _more_tags(self):
        return {"multioutput_only": True}
