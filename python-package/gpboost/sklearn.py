# coding: utf-8
"""
Scikit-learn wrapper interface for GPBoost.

Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
"""
import copy

from inspect import signature

import numpy as np

from .basic import Dataset, GPBoostError, _ConfigAliases, _choose_param_value, _log_warning
from .compat import (SKLEARN_INSTALLED, _GPBoostClassifierBase,
                     GPBoostNotFittedError, _GPBoostLabelEncoder, _GPBoostModelBase,
                     _GPBoostRegressorBase, _GPBoostCheckXY, _GPBoostCheckArray, _GPBoostCheckSampleWeight,
                     _GPBoostAssertAllFinite, _GPBoostCheckClassificationTargets, _GPBoostComputeSampleWeight,
                     pd_DataFrame, dt_DataTable)
from .engine import train


class _ObjectiveFunctionWrapper:
    """Proxy class for objective function."""

    def __init__(self, func):
        """Construct a proxy class.

        This class transforms objective function to match objective function with signature ``new_func(preds, dataset)``
        as expected by ``gpboost.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with signature ``func(y_true, y_pred)`` or ``func(y_true, y_pred, group)
            and returns (grad, hess):

                y_true : array-like of shape = [n_samples]
                    The target values.
                y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The predicted values.
                group : array-like
                    Group/query data.
                    Only used in the learning-to-rank task.
                    sum(group) = n_samples.
                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
                grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The value of the first order derivative (gradient) for each sample point.
                hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The value of the second order derivative (Hessian) for each sample point.

        .. note::

            For binary task, the y_pred is margin.
            For multi-class task, the y_pred is group by class_id first, then group by row_id.
            If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
            and you should group grad and hess in this way as well.
        """
        self.func = func

    def __call__(self, preds, dataset):
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The value of the first order derivative (gradient) for each sample point.
        hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The value of the second order derivative (Hessian) for each sample point.
        """
        labels = dataset.get_label()
        argc = len(signature(self.func).parameters)
        if argc == 2:
            grad, hess = self.func(labels, preds)
        elif argc == 3:
            grad, hess = self.func(labels, preds, dataset.get_group())
        else:
            raise TypeError("Self-defined objective function should have 2 or 3 arguments, got %d" % argc)
        """weighted for objective"""
        weight = dataset.get_weight()
        if weight is not None:
            """only one class"""
            if len(weight) == len(grad):
                grad = np.multiply(grad, weight)
                hess = np.multiply(hess, weight)
            else:
                num_data = len(weight)
                num_class = len(grad) // num_data
                if num_class * num_data != len(grad):
                    raise ValueError("Length of grad and hess should equal to num_class * num_data")
                for k in range(num_class):
                    for i in range(num_data):
                        idx = k * num_data + i
                        grad[idx] *= weight[i]
                        hess[idx] *= weight[i]
        return grad, hess


class _EvalFunctionWrapper:
    """Proxy class for evaluation function."""

    def __init__(self, func):
        """Construct a proxy class.

        This class transforms evaluation function to match evaluation function with signature ``new_func(preds, dataset)``
        as expected by ``gpboost.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
            and returns (eval_name, eval_result, is_higher_better) or
            list of (eval_name, eval_result, is_higher_better):

                y_true : array-like of shape = [n_samples]
                    The target values.
                y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The predicted values.
                weight : array-like of shape = [n_samples]
                    The weight of samples.
                group : array-like
                    Group/query data.
                    Only used in the learning-to-rank task.
                    sum(group) = n_samples.
                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
                eval_name : string
                    The name of evaluation function (without whitespaces).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

        .. note::

            For binary task, the y_pred is probability of positive class (or margin in case of custom ``objective``).
            For multi-class task, the y_pred is group by class_id first, then group by row_id.
            If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
        """
        self.func = func

    def __call__(self, preds, dataset):
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        eval_name : string
            The name of evaluation function (without whitespaces).
        eval_result : float
            The eval result.
        is_higher_better : bool
            Is eval result higher better, e.g. AUC is ``is_higher_better``.
        """
        labels = dataset.get_label()
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)
        elif argc == 3:
            return self.func(labels, preds, dataset.get_weight())
        elif argc == 4:
            return self.func(labels, preds, dataset.get_weight(), dataset.get_group())
        else:
            raise TypeError("Self-defined eval function should have 2, 3 or 4 arguments, got %d" % argc)


class GPBoostModel(_GPBoostModelBase):
    """Implementation of the scikit-learn API for GPBoost."""

    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=0, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True, importance_type='split', **kwargs):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : string, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for GPBoostRegressor, 'binary' or 'multiclass' for GPBoostClassifier, 'lambdarank' for GPBoostRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequence of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.
        importance_type : string, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess`` or
        ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true : array-like of shape = [n_samples]
                The target values.
            y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            group : array-like
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the first order derivative (gradient) for each sample point.
            hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) for each sample point.

        For binary task, the y_pred is margin.
        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
        and you should group grad and hess in this way as well.
        """
        if not SKLEARN_INSTALLED:
            raise GPBoostError('scikit-learn is required for gpboost.sklearn')

        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        self._Booster = None
        self._evals_result = None
        self._best_score = None
        self._best_iteration = None
        self._other_params = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight = None
        self._class_map = None
        self._n_features = None
        self._n_features_in = None
        self._classes = None
        self._n_classes = None
        self.set_params(**kwargs)

    def _more_tags(self):
        return {
            'allow_nan': True,
            'X_types': ['2darray', 'sparse', '1dlabels'],
            '_xfail_checks': {
                'check_no_attributes_set_in_init':
                'scikit-learn incorrectly asserts that private attributes '
                'cannot be set in __init__: '
                '(see https://github.com/microsoft/LightGBM/issues/2628)'
            }
        }

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        params.update(self._other_params)
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, '_' + key):
                setattr(self, '_' + key, value)
            self._other_params[key] = value
        return self

    def fit(self, X, y,
            gp_model=None, use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_group=None,
            eval_metric=None, early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto',
            callbacks=None, init_model=None):
        """Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).
        gp_model : GPModel or None, optional (default=None)
            GPModel object for the GPBoost algorithm
        use_gp_model_for_validation : bool, optional (default=True)
            If True, the gp_model (Gaussian process and/or random effects) is also used (in addition to the tree model)
            for calculating predictions on the validation data
        train_gp_model_cov_pars : bool, optional (default=True)
            If True, the covariance parameters of the gp_model (Gaussian process and/or random effects) are estimated
            in every boosting iterations, otherwise the gp_model parameters are not estimated. In the latter case, you
            need to either estimate them beforehand or provide the values via the 'init_cov_pars' parameter when creating
            the gp_model
        sample_weight : array-like of shape = [n_samples] or None, optional (default=None)
            Weights of training data.
        init_score : array-like of shape = [n_samples] or None, optional (default=None)
            Init score of training data.
        group : array-like or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        eval_set : list or None, optional (default=None)
            A list of (X, y) tuple pairs to use as validation sets.
        eval_names : list of strings or None, optional (default=None)
            Names of eval_set.
        eval_sample_weight : list of arrays or None, optional (default=None)
            Weights of eval data.
        eval_class_weight : list or None, optional (default=None)
            Class weights of eval data.
        eval_init_score : list of arrays or None, optional (default=None)
            Init score of eval data.
        eval_group : list of arrays or None, optional (default=None)
            Group data of eval data.
        eval_metric : string, callable, list or None, optional (default=None)
            If string, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for GPBoostRegressor, 'logloss' for GPBoostClassifier, 'ndcg' for GPBoostRanker.
        early_stopping_rounds : int or None, optional (default=None)
            Activates early stopping. The model will train until the validation score stops improving.
            Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
            to continue training.
            Requires at least one validation data and one metric.
            If there's more than one, will check all of them. But the training data is ignored anyway.
            To check only the first metric, set the ``first_metric_only`` parameter to ``True``
            in additional parameters ``**kwargs`` of the model constructor.
        verbose : bool or int, optional (default=True)
            Requires at least one evaluation data.
            If True, the eval metric on the eval set is printed at each boosting stage.
            If int, the eval metric on the eval set is printed at every ``verbose`` boosting stage.
            The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed.

            .. rubric:: Example

            With ``verbose`` = 4 and at least one item in ``eval_set``,
            an evaluation metric is printed every 4 (instead of 1) boosting stages.

        feature_name : list of strings or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
        callbacks : list of callback functions or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        init_model : string, Booster, GPBoostModel or None, optional (default=None)
            Filename of GPBoost model, Booster instance or GPBoostModel instance used for continue training.

        Returns
        -------
        self : object
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (eval_name, eval_result, is_higher_better) or
        list of (eval_name, eval_result, is_higher_better):

            y_true : array-like of shape = [n_samples]
                The target values.
            y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            weight : array-like of shape = [n_samples]
                The weight of samples.
            group : array-like
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For binary task, the y_pred is probability of positive class (or margin in case of custom ``objective``).
        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
        """
        if self._objective is None:
            if isinstance(self, GPBoostRegressor):
                self._objective = "regression"
            elif isinstance(self, GPBoostClassifier):
                self._objective = "binary"
            elif isinstance(self, GPBoostRanker):
                self._objective = "lambdarank"
            else:
                raise ValueError("Unknown GPBoostModel type.")
        if callable(self._objective):
            self._fobj = _ObjectiveFunctionWrapper(self._objective)
        else:
            self._fobj = None
        evals_result = {}
        params = self.get_params()
        # user can set verbose with kwargs, it has higher priority
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and self.silent:
            params['verbose'] = -1
        params.pop('silent', None)
        params.pop('importance_type', None)
        params.pop('n_estimators', None)
        params.pop('class_weight', None)
        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(np.iinfo(np.int32).max)
        for alias in _ConfigAliases.get('objective'):
            params.pop(alias, None)
        if self._n_classes is not None and self._n_classes > 2:
            for alias in _ConfigAliases.get('num_class'):
                params.pop(alias, None)
            params['num_class'] = self._n_classes
        if hasattr(self, '_eval_at'):
            for alias in _ConfigAliases.get('eval_at'):
                params.pop(alias, None)
            params['eval_at'] = self._eval_at
        params['objective'] = self._objective
        if self._fobj:
            params['objective'] = 'None'  # objective = nullptr for unknown objective

        # Do not modify original args in fit function
        # Refer to https://github.com/microsoft/LightGBM/pull/2619
        eval_metric_list = copy.deepcopy(eval_metric)
        if not isinstance(eval_metric_list, list):
            eval_metric_list = [eval_metric_list]

        # Separate built-in from callable evaluation metrics
        eval_metrics_callable = [_EvalFunctionWrapper(f) for f in eval_metric_list if callable(f)]
        eval_metrics_builtin = [m for m in eval_metric_list if isinstance(m, str)]

        # register default metric for consistency with callable eval_metric case
        original_metric = self._objective if isinstance(self._objective, str) else None
        if original_metric is None:
            # try to deduce from class instance
            if isinstance(self, GPBoostRegressor):
                original_metric = "l2"
            elif isinstance(self, GPBoostClassifier):
                original_metric = "multi_logloss" if self._n_classes > 2 else "binary_logloss"
            elif isinstance(self, GPBoostRanker):
                original_metric = "ndcg"

        # overwrite default metric by explicitly set metric
        params = _choose_param_value("metric", params, original_metric)

        # concatenate metric from params (or default if not provided in params) and eval_metric
        params['metric'] = [params['metric']] if isinstance(params['metric'], (str, type(None))) else params['metric']
        params['metric'] = [e for e in eval_metrics_builtin if e not in params['metric']] + params['metric']
        params['metric'] = [metric for metric in params['metric'] if metric is not None]

        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            _X, _y = _GPBoostCheckXY(X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2)
            if sample_weight is not None:
                sample_weight = _GPBoostCheckSampleWeight(sample_weight, _X)
        else:
            _X, _y = X, y

        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _GPBoostComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)

        self._n_features = _X.shape[1]
        # copy for consistency
        self._n_features_in = self._n_features

        def _construct_dataset(X, y, sample_weight, init_score, group, params,
                               categorical_feature='auto'):
            return Dataset(X, label=y, weight=sample_weight, group=group,
                           init_score=init_score, params=params,
                           categorical_feature=categorical_feature)

        train_set = _construct_dataset(_X, _y, sample_weight, init_score, group, params,
                                       categorical_feature=categorical_feature)

        valid_sets = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError('{} should be dict or list'.format(name))

            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(eval_sample_weight, 'eval_sample_weight', i)
                    valid_class_weight = _get_meta_data(eval_class_weight, 'eval_class_weight', i)
                    if valid_class_weight is not None:
                        if isinstance(valid_class_weight, dict) and self._class_map is not None:
                            valid_class_weight = {self._class_map[k]: v for k, v in valid_class_weight.items()}
                        valid_class_sample_weight = _GPBoostComputeSampleWeight(valid_class_weight, valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = _get_meta_data(eval_init_score, 'eval_init_score', i)
                    valid_group = _get_meta_data(eval_group, 'eval_group', i)
                    valid_set = _construct_dataset(valid_data[0], valid_data[1],
                                                   valid_weight, valid_init_score, valid_group, params)
                valid_sets.append(valid_set)

        if isinstance(init_model, GPBoostModel):
            init_model = init_model.booster_

        self._Booster = train(params=params, train_set=train_set, num_boost_round=self.n_estimators,
                              gp_model=gp_model, use_gp_model_for_validation=use_gp_model_for_validation,
                              train_gp_model_cov_pars=train_gp_model_cov_pars,
                              valid_sets=valid_sets, valid_names=eval_names,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, fobj=self._fobj, feval=eval_metrics_callable,
                              verbose_eval=verbose, feature_name=feature_name,
                              callbacks=callbacks, init_model=init_model)

        if evals_result:
            self._evals_result = evals_result

        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            self._best_iteration = self._Booster.best_iteration

        self._best_score = self._Booster.best_score

        self.fitted_ = True

        # free dataset
        if gp_model is None:
            self._Booster.free_dataset()
        del train_set, valid_sets
        return self

    def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False,
                group_data_pred=None, group_rand_coef_data_pred=None,
                gp_coords_pred=None, gp_rand_coef_data_pred=None,
                cluster_ids_pred=None, vecchia_pred_type=None,
                num_neighbors_pred=None, predict_cov_mat=False, predict_var=False, **kwargs):
        """Return the predicted value for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        group_data_pred : numpy array with numeric or string data or None, optional (default=None)
            Labels of group levels for grouped random effects. Used only if the Booster has a GPModel
        group_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
            Covariate data for grouped random coefficients. Used only if the Booster has a GPModel
        gp_coords_pred : numpy array with numeric data or None, optional (default=None)
            Coordinates (features) for Gaussian process. Used only if the Booster has a GPModel
        gp_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
            Covariate data for Gaussian process random coefficients. Used only if the Booster has a GPModel
        cluster_ids_pred : one dimensional numpy array (vector) with integer data or None, optional (default=None)
            IDs / labels indicating independent realizations of random effects / Gaussian processes
            (same values = same process realization). Used only if the Booster has a GPModel
        predict_cov_mat : bool, optional (default=False)
            If True, the (posterior / conditional) predictive covariance is calculated in addition to the
            (posterior / conditional) predictive mean. Used only if the Booster has a GPModel
        predict_var : bool, optional (default=False)
            If True, (posterior / conditional) predictive variances are calculated in addition to the
            (posterior / conditional) predictive mean. Used only if the Booster has a GPModel
        vecchia_pred_type : string, optional (default=None)
            The type of Vecchia approximation used for making predictions.
            This is discontinued here. Use the function 'set_prediction_data' 
            of the 'gp_model' to specify this
        num_neighbors_pred : integer or None, optional (default=None)
            The number of neighbors for making predictions.
            This is discontinued here. Use the function 'set_prediction_data' 
            of the 'gp_model' to specify this
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        if self._n_features is None:
            raise GPBoostNotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            X = _GPBoostCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features_ is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        return self._Booster.predict(X, pred_latent=raw_score, start_iteration=start_iteration, num_iteration=num_iteration,
                                     pred_leaf=pred_leaf, pred_contrib=pred_contrib,
                                     group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                                     gp_coords_pred=gp_coords_pred, gp_rand_coef_data_pred=gp_rand_coef_data_pred,
                                     cluster_ids_pred=cluster_ids_pred, vecchia_pred_type=vecchia_pred_type,
                                     num_neighbors_pred=num_neighbors_pred, predict_cov_mat=predict_cov_mat,
                                     predict_var=predict_var, **kwargs)

    @property
    def n_features_(self):
        """:obj:`int`: The number of features of fitted model."""
        if self._n_features is None:
            raise GPBoostNotFittedError('No n_features found. Need to call fit beforehand.')
        return self._n_features

    @property
    def n_features_in_(self):
        """:obj:`int`: The number of features of fitted model."""
        if self._n_features_in is None:
            raise GPBoostNotFittedError('No n_features_in found. Need to call fit beforehand.')
        return self._n_features_in

    @property
    def best_score_(self):
        """:obj:`dict` or :obj:`None`: The best score of fitted model."""
        if self._n_features is None:
            raise GPBoostNotFittedError('No best_score found. Need to call fit beforehand.')
        return self._best_score

    @property
    def best_iteration_(self):
        """:obj:`int` or :obj:`None`: The best iteration of fitted model if ``early_stopping_rounds`` has been specified."""
        if self._n_features is None:
            raise GPBoostNotFittedError('No best_iteration found. Need to call fit with early_stopping_rounds beforehand.')
        return self._best_iteration

    @property
    def objective_(self):
        """:obj:`string` or :obj:`callable`: The concrete objective used while fitting this model."""
        if self._n_features is None:
            raise GPBoostNotFittedError('No objective found. Need to call fit beforehand.')
        return self._objective

    @property
    def booster_(self):
        """Booster: The underlying Booster of this model."""
        if self._Booster is None:
            raise GPBoostNotFittedError('No booster found. Need to call fit beforehand.')
        return self._Booster

    @property
    def evals_result_(self):
        """:obj:`dict` or :obj:`None`: The evaluation results if ``early_stopping_rounds`` has been specified."""
        if self._n_features is None:
            raise GPBoostNotFittedError('No results found. Need to call fit with eval_set beforehand.')
        return self._evals_result

    @property
    def feature_importances_(self):
        """:obj:`array` of shape = [n_features]: The feature importances (the higher, the more important).

        .. note::

            ``importance_type`` attribute is passed to the function
            to configure the type of importance values to be extracted.
        """
        if self._n_features is None:
            raise GPBoostNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self._Booster.feature_importance(importance_type=self.importance_type)

    @property
    def feature_name_(self):
        """:obj:`array` of shape = [n_features]: The names of features."""
        if self._n_features is None:
            raise GPBoostNotFittedError('No feature_name found. Need to call fit beforehand.')
        return self._Booster.feature_name()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags


class GPBoostRegressor(GPBoostModel, _GPBoostRegressorBase):
    """GPBoost regressor."""

    def fit(self, X, y,
            gp_model = None, use_gp_model_for_validation = True, train_gp_model_cov_pars = True,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_metric=None, early_stopping_rounds=None,
            verbose=True, feature_name='auto', categorical_feature='auto',
            callbacks=None, init_model=None):
        """Docstring is inherited from the GPBoostModel."""
        super().fit(X, y,
                    gp_model=gp_model, use_gp_model_for_validation=use_gp_model_for_validation,
                    train_gp_model_cov_pars=train_gp_model_cov_pars,
                    sample_weight=sample_weight, init_score=init_score,
                    eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight,
                    eval_init_score=eval_init_score, eval_metric=eval_metric,
                    early_stopping_rounds=early_stopping_rounds, verbose=verbose, feature_name=feature_name,
                    categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self

    _base_doc = GPBoostModel.fit.__doc__
    _base_doc = (_base_doc[:_base_doc.find('group :')]
                 + _base_doc[_base_doc.find('eval_set :'):])
    _base_doc = (_base_doc[:_base_doc.find('eval_class_weight :')]
                 + _base_doc[_base_doc.find('eval_init_score :'):])
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_group :')]
                   + _base_doc[_base_doc.find('eval_metric :'):])


class GPBoostClassifier(GPBoostModel, _GPBoostClassifierBase):
    """GPBoost classifier."""

    def fit(self, X, y,
            gp_model=None, use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto',
            callbacks=None, init_model=None):
        """Docstring is inherited from the GPBoostModel."""
        _GPBoostAssertAllFinite(y)
        _GPBoostCheckClassificationTargets(y)
        self._le = _GPBoostLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._class_map = dict(zip(self._le.classes_, self._le.transform(self._le.classes_)))
        if isinstance(self.class_weight, dict):
            self._class_weight = {self._class_map[k]: v for k, v in self.class_weight.items()}

        self._classes = self._le.classes_
        self._n_classes = len(self._classes)

        if self._n_classes > 2:
            # Switch to using a multiclass objective in the underlying GPBoost instance
            ova_aliases = {"multiclassova", "multiclass_ova", "ova", "ovr"}
            if self._objective not in ova_aliases and not callable(self._objective):
                self._objective = "multiclass"

        if not callable(eval_metric):
            if isinstance(eval_metric, (str, type(None))):
                eval_metric = [eval_metric]
            if self._n_classes > 2:
                for index, metric in enumerate(eval_metric):
                    if metric in {'logloss', 'binary_logloss'}:
                        eval_metric[index] = "multi_logloss"
                    elif metric in {'error', 'binary_error'}:
                        eval_metric[index] = "multi_error"
            else:
                for index, metric in enumerate(eval_metric):
                    if metric in {'logloss', 'multi_logloss'}:
                        eval_metric[index] = 'binary_logloss'
                    elif metric in {'error', 'multi_error'}:
                        eval_metric[index] = 'binary_error'

        # do not modify args, as it causes errors in model selection tools
        valid_sets = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = [None] * len(eval_set)
            for i, (valid_x, valid_y) in enumerate(eval_set):
                if valid_x is X and valid_y is y:
                    valid_sets[i] = (valid_x, _y)
                else:
                    valid_sets[i] = (valid_x, self._le.transform(valid_y))

        super().fit(X, _y,
                    gp_model=gp_model, use_gp_model_for_validation=use_gp_model_for_validation,
                    train_gp_model_cov_pars=train_gp_model_cov_pars,
                    sample_weight=sample_weight, init_score=init_score, eval_set=valid_sets,
                    eval_names=eval_names, eval_sample_weight=eval_sample_weight,
                    eval_class_weight=eval_class_weight, eval_init_score=eval_init_score,
                    eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose, feature_name=feature_name, categorical_feature=categorical_feature,
                    callbacks=callbacks, init_model=init_model)
        return self

    _base_doc = GPBoostModel.fit.__doc__
    _base_doc = (_base_doc[:_base_doc.find('group :')]
                 + _base_doc[_base_doc.find('eval_set :'):])
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_group :')]
                   + _base_doc[_base_doc.find('eval_metric :'):])

    def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False,
                group_data_pred=None, group_rand_coef_data_pred=None,
                gp_coords_pred=None, gp_rand_coef_data_pred=None,
                cluster_ids_pred=None, vecchia_pred_type=None,
                num_neighbors_pred=None, predict_cov_mat=False, predict_var=False, **kwargs):
        """Docstring is inherited from the GPBoostModel."""
        result = self.predict_proba(X=X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration,
                                    pred_leaf=pred_leaf, pred_contrib=pred_contrib,
                                    group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                                    gp_coords_pred=gp_coords_pred, gp_rand_coef_data_pred=gp_rand_coef_data_pred,
                                    cluster_ids_pred=cluster_ids_pred, vecchia_pred_type=vecchia_pred_type,
                                    num_neighbors_pred=num_neighbors_pred, predict_cov_mat=predict_cov_mat, predict_var=predict_var,
                                    **kwargs)
        if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            if self._Booster.has_gp_model:
                if self._n_classes > 2:
                    result = result['response_mean']
                else:
                    result = np.vstack((1. - result['response_mean'], result['response_mean'])).transpose()
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    predict.__doc__ = GPBoostModel.predict.__doc__

    def predict_proba(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                      pred_leaf=False, pred_contrib=False,
                      group_data_pred=None, group_rand_coef_data_pred=None,
                      gp_coords_pred=None, gp_rand_coef_data_pred=None,
                      cluster_ids_pred=None, vecchia_pred_type=None,
                      num_neighbors_pred=None, predict_cov_mat=False, predict_var=False, **kwargs):
        """Return the predicted probability for each class for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        group_data_pred : numpy array with numeric or string data or None, optional (default=None)
            Labels of group levels for grouped random effects. Used only if the Booster has a GPModel
        group_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
            Covariate data for grouped random coefficients. Used only if the Booster has a GPModel
        gp_coords_pred : numpy array with numeric data or None, optional (default=None)
            Coordinates (features) for Gaussian process. Used only if the Booster has a GPModel
        gp_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
            Covariate data for Gaussian process random coefficients. Used only if the Booster has a GPModel
        cluster_ids_pred : one dimensional numpy array (vector) with integer data or None, optional (default=None)
            IDs / labels indicating independent realizations of random effects / Gaussian processes
            (same values = same process realization). Used only if the Booster has a GPModel
        predict_cov_mat : bool, optional (default=False)
            If True, the (posterior / conditional) predictive covariance is calculated in addition to the
            (posterior / conditional) predictive mean. Used only if the Booster has a GPModel
        predict_var : bool, optional (default=False)
            If True, (posterior / conditional) predictive variances are calculated in addition to the
            (posterior / conditional) predictive mean. Used only if the Booster has a GPModel
        vecchia_pred_type : string, optional (default=None)
            The type of Vecchia approximation used for making predictions.
            This is discontinued here. Use the function 'set_prediction_data' 
            of the 'gp_model' to specify this
        num_neighbors_pred : integer or None, optional (default=None)
            The number of neighbors for making predictions.
            This is discontinued here. Use the function 'set_prediction_data' 
            of the 'gp_model' to specify this
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_probability : array-like of shape = [n_samples, n_classes]
            The predicted probability for each class for each sample.
        X_leaves : array-like of shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        result = super().predict(X=X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration,
                                 pred_leaf=pred_leaf, pred_contrib=pred_contrib,
                                 group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                                 gp_coords_pred=gp_coords_pred, gp_rand_coef_data_pred=gp_rand_coef_data_pred,
                                 cluster_ids_pred=cluster_ids_pred, vecchia_pred_type=vecchia_pred_type,
                                 num_neighbors_pred=num_neighbors_pred, predict_cov_mat=predict_cov_mat,
                                 predict_var=predict_var, **kwargs)
        if callable(self._objective) and not (raw_score or pred_leaf or pred_contrib):
            _log_warning("Cannot compute class probabilities or labels "
                         "due to the usage of customized objective function.\n"
                         "Returning raw scores instead.")
            return result
        elif self._n_classes > 2 or raw_score or pred_leaf or pred_contrib or self._Booster.has_gp_model:
            return result
        else:
            return np.vstack((1. - result, result)).transpose()

    @property
    def classes_(self):
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if self._classes is None:
            raise GPBoostNotFittedError('No classes found. Need to call fit beforehand.')
        return self._classes

    @property
    def n_classes_(self):
        """:obj:`int`: The number of classes."""
        if self._n_classes is None:
            raise GPBoostNotFittedError('No classes found. Need to call fit beforehand.')
        return self._n_classes


class GPBoostRanker(GPBoostModel):
    """GPBoost ranker."""

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None, eval_metric=None,
            eval_at=(1, 2, 3, 4, 5), early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto',
            callbacks=None, init_model=None):
        """Docstring is inherited from the GPBoostModel."""
        # check group data
        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            elif len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should be equal to eval_set")
            elif (isinstance(eval_group, dict)
                  and any(i not in eval_group or eval_group[i] is None for i in range(len(eval_group)))
                  or isinstance(eval_group, list)
                  and any(group is None for group in eval_group)):
                raise ValueError("Should set group for all eval datasets for ranking task; "
                                 "if you use dict, the index should start from 0")

        self._eval_at = eval_at
        super().fit(X, y, sample_weight=sample_weight, init_score=init_score, group=group,
                    eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight,
                    eval_init_score=eval_init_score, eval_group=eval_group, eval_metric=eval_metric,
                    early_stopping_rounds=early_stopping_rounds, verbose=verbose, feature_name=feature_name,
                    categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self

    _base_doc = GPBoostModel.fit.__doc__
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_class_weight :')]
                   + _base_doc[_base_doc.find('eval_init_score :'):])
    _base_doc = fit.__doc__
    _before_early_stop, _early_stop, _after_early_stop = _base_doc.partition('early_stopping_rounds :')
    fit.__doc__ = (_before_early_stop
                   + 'eval_at : iterable of int, optional (default=(1, 2, 3, 4, 5))\n'
                   + ' ' * 12 + 'The evaluation positions of the specified metric.\n'
                   + ' ' * 8 + _early_stop + _after_early_stop)
