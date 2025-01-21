from statistics import LinearRegression

from lazypredict import LazyClassifier, LazyRegressor
from lightgbm import LGBMRegressor
from matplotlib.widgets import Lasso
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.ensemble._weight_boosting import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._bagging import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, GammaRegressor, HuberRegressor, Lars, LarsCV, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, PoissonRegressor, QuantileRegressor, RANSACRegressor, Ridge, RidgeCV, SGDClassifier, SGDRegressor, TweedieRegressor
from sklearn.linear_model._bayes import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree._classes import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble._forest import ExtraTreesClassifier, ExtraTreesRegressor
from xgboost import XGBRegressor

regressors_whitelist = [
    AdaBoostRegressor, BaggingRegressor, BayesianRidge, DecisionTreeRegressor, DummyRegressor, ElasticNet, ElasticNetCV,
    ExtraTreeRegressor, ExtraTreesRegressor, GammaRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
    HuberRegressor, KNeighborsRegressor, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression,
    LinearSVR, MLPRegressor, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, PoissonRegressor,
    QuantileRegressor, RANSACRegressor, RandomForestClassifier, Ridge, RidgeCV, SGDRegressor, SVR, TransformedTargetRegressor,
    TweedieRegressor, XGBRegressor, LGBMRegressor
]


def get_lazy_regressor(verbose=0, ignore_warnings=True, custom_metric=None, exclude: list[str] = None, predictions=False) -> LazyRegressor:
    whitelist = regressors_whitelist
    if exclude:
        whitelist = [regressor for regressor in regressors_whitelist if regressor.__name__ not in exclude]

    return LazyRegressor(verbose=verbose, ignore_warnings=ignore_warnings, custom_metric=custom_metric, regressors=whitelist, predictions=predictions)
