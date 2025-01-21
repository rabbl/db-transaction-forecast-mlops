from ..LazyPredict import get_lazy_regressor, regressors_whitelist
from lazypredict import LazyRegressor


def test_regressors_whitelist():
    assert isinstance(regressors_whitelist, list)


def test_get_lazy_regressor():
    assert isinstance(get_lazy_regressor(), LazyRegressor)
    assert isinstance(get_lazy_regressor(verbose=1), LazyRegressor)
    assert isinstance(get_lazy_regressor(ignore_warnings=False), LazyRegressor)
    assert isinstance(get_lazy_regressor(custom_metric='r2'), LazyRegressor)
    assert isinstance(get_lazy_regressor(exclude=['LinearRegression']), LazyRegressor)
    assert isinstance(get_lazy_regressor(predictions=True), LazyRegressor)
    assert isinstance(get_lazy_regressor(
        verbose=1, ignore_warnings=False,
        custom_metric='r2', exclude=['LinearRegression'],
        predictions=True), LazyRegressor
    )


def test_can_import_from_module():
    from ..LazyPredict import get_lazy_regressor, regressors_whitelist
    assert callable(get_lazy_regressor)
    assert isinstance(regressors_whitelist, list)
