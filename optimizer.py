import numpy as np
import scipy.optimize

__all__ = ['create_optimizer']


def to_array(item):
    if isinstance(item, (tuple, list)):
        return np.array(item)
    elif isinstance(item, np.ndarray):
        return item


_big_number = 9e99
_small_number = 1e-6


def none_or_call(indicator, _callable, *args, **kwargs):
    if indicator is not None:
        return _callable(*args, **kwargs)


class ParametersSet:
    def __init__(self, data):
        self._data = to_array(data)

    @property
    def values(self):
        return self._data

    def __eq__(self, other):
        if isinstance(other, ParametersSet):
            return np.all(self.values == other.values)
        else:
            raise NotImplementedError

    def change(self, values, max_correction=None):
        return ParametersSet(
            self._reduce_changes(to_array(values), max_correction)
        )

    def _reduce_changes(self, values, max_correction):
        return self._clip(
            values,
            none_or_call(max_correction, np.subtract, self._data, max_correction),
            none_or_call(max_correction, np.add, self._data, max_correction)
        )

    def clip(self, lower=None, upper=None):
        return ParametersSet(
            self._clip(self._data, lower, upper)
        )

    def clip_softly(self, lower, upper, softening_ratio=1e-6):
        clipped = self._clip(self._data, lower, upper)

        negative_exceeding = np.multiply(np.subtract(self._data, lower).clip(-_big_number, 0.), softening_ratio)
        positive_exceeding = np.multiply(np.subtract(self._data, upper).clip(0., _big_number), softening_ratio)

        clipped = np.add(clipped, negative_exceeding)
        clipped = np.add(clipped, positive_exceeding)

        return ParametersSet(clipped)

    def _clip(self, data, lower=None, upper=None):
        return np.maximum(
            self._to_limit_array(lower, -_big_number),
            np.minimum(
                self._to_limit_array(upper, _big_number),
                data.copy()
            )
        )

    def _to_limit_array(self, item, extreme):
        if isinstance(item, (float, int, type(None))):
            array = self._data.copy()
            array.fill(extreme if item is None else item)
            return array
        else:
            return to_array(item)

    def __str__(self):
        return "{name}: {data}".format(name=self.__class__.__name__, data=self._data)


class FullyStressDesignUpdater:
    _bounds_softening = 1e-6
    _default_bounds = (_small_number, 1 - _small_number)

    def __init__(self, calculate_exceeded_volume, max_correction=None, bounds=None):
        self._calculate_exceeded_volume = calculate_exceeded_volume
        self._max_correction = max_correction
        self._bounds = bounds or self._default_bounds

    def update_parameters(self, initial, field):
        updated = initial.change(
            np.multiply(initial.values, np.abs(field)),
            max_correction=self._max_correction
        )
        _lambda = self._estimate_lambda(updated)
        return (
            initial
                .change(np.divide(updated.values, _lambda))
                .clip(*self._bounds)
        )

    def _estimate_lambda(self, parameters):
        def calc_exceeded(_lambda):
            return self._calculate_exceeded_volume(
                (
                    parameters
                        .change(np.divide(parameters.values, _lambda))
                        .clip_softly(self._bounds[0], self._bounds[1], self._bounds_softening)
                        .values

                )
            )
        return scipy.optimize.fsolve(calc_exceeded, 1.)[0]


_engines = {
    'fully_stress_design': FullyStressDesignUpdater
}


def create_engine(_type, *args, **kwargs):
    return _engines[_type](*args, **kwargs)


class Optimizer:
    def __init__(self, parameters, engine, calculate_strain_energy_density, accuracy=1e-5):

        self._parameters_set = parameters
        self._engine = engine
        self._calculate_strain_energy_density = calculate_strain_energy_density
        self._accuracy = accuracy

    def solve(self):

        while 1:
            field = self._calculate_strain_energy_density(self._parameters_set.values)
            new_parameters = self._engine.update_parameters(self._parameters_set, field)
            if self._check_convergence(self._parameters_set.values, new_parameters.values):
                break
            else:
                self._parameters_set = new_parameters

        return new_parameters

    def _check_convergence(self, old_parameters, new_parameters):
        return np.linalg.norm(np.subtract(new_parameters, old_parameters)) < self._accuracy

    @property
    def parameters(self):
        return self._parameters_set.values


def create_optimizer(engine, initial, field_calculator, exceeded_volume_calculator,
                     max_correction=None, bounds=None):
    return Optimizer(
        ParametersSet(initial),
        create_engine(engine, exceeded_volume_calculator, max_correction=max_correction, bounds=bounds),
        field_calculator
    )
