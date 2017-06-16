import unittest

import numpy as np
from toptim.optimizer import ParametersSet, create_engine


class ParametersSetTest(unittest.TestCase):
    def test_Create_DataProvided_StoreValuesAsArray(self):

        data = [1., 2., 3.]

        _set = self._create_set(data)

        self.assertTrue(isinstance(_set.values, np.ndarray))
        np.testing.assert_array_equal(np.array(data), _set.values)

    def test_Change_Always_ReturnCloneWithChangedData(self):

        _set = self._create_set([1.0])

        result = _set.change([2.2])

        self._check_if_clone(_set, result)
        self._check_result([2.2], result)

    def test_Change_MaxCorrection_ReturnCloneClippedData(self):
        max_correction = 0.1
        old_values = [1., 2., 0., 0.1, 0.2]
        new_values = [-1, 0.5, 2., 0.1, 0.25]

        _set = self._create_set(old_values)

        result = _set.change(new_values, max_correction)

        self._check_if_clone(_set, result)
        self._check_result([1. - max_correction, 2. - max_correction, 0. + max_correction, 0.1, 0.25], result)

    def test_Clip_ScalarLimit_ReturnCloneDataWithinGivenLimits(self):
        _set = self._create_set([0., -1., 2., 3.])

        result = _set.clip(lower=0.5, upper=2.5)

        self._check_if_clone(_set, result)
        self._check_result([0.5, 0.5, 2., 2.5], result)

    def test_Clip_ArrayLimit_ReturnCloneDataWithinIndividualLimits(self):
        _set = self._create_set([0., -1., 2., 3.])

        lower = np.full((4, ), 0.5)
        upper = np.full((4, ), 2.5)

        result = _set.clip(lower, upper)

        self._check_if_clone(_set, result)
        self._check_result([0.5, 0.5, 2., 2.5], result)

    def test_Clip_Softly_ReturnCloneDataSlightlyExceedingLimits(self):
        softening = 1e-6
        _set = self._create_set([0., -1., 2., 3.])

        result = _set.clip_softly(0.5, 2.5, softening)

        self._check_if_clone(_set, result)
        self._check_result([0.5 - softening*0.5, 0.5 - softening*1.5, 2., 2.5 + softening*0.5], result)

    def test_Clip_SoftlyAndTwoEqualExceedLimit_ReturnCloneDataSlightlyExceedingLimits(self):
        softening_ratio = 1e-2
        _set = self._create_set([-1., -1.])

        result = _set.clip_softly(0.0, 2.0, softening_ratio)

        self._check_if_clone(_set, result)
        self._check_result([0.0 - 1.0*softening_ratio, 0.0 - 1.0*softening_ratio], result)

    def test_Clip_SoftlyAndParametersWithinLimits_ReturnCloneData(self):
        tolerance = 0.1
        _set = self._create_set([1., 2.])

        result = _set.clip_softly(0.5, 2.5, tolerance)

        self._check_if_clone(_set, result)
        self._check_result([1., 2.], result)

    def _check_result(self, expected, result):
        np.testing.assert_allclose(
            np.array(expected),
            result.values
        )

    def test_Equal_Always_CompareData(self):

        set_1 = self._create_set([1., 2.])
        set_2 = self._create_set([3., 4.])
        set_3 = self._create_set([1., 2.])

        self.assertTrue(set_1 == set_3)
        self.assertFalse(set_1 == set_2)

    def _check_if_clone(self, original, output):
        self.assertNotEqual(id(original), id(output))

    def _create_set(self, data, **kwargs):
        return ParametersSet(data, **kwargs)


class FullyStressDesignEngineTest(unittest.TestCase):
    def test_Update_ParametersWithinLimits_ReturnValuesMultipliedByAbsStrains(self):

        parameters = ParametersSet([0.1, 0.2])
        field = np.array([0.3, 0.4])

        def constraint_calculator(*args):
            return 0.

        engine = self._create_engine(constraint_calculator)

        updated = engine.update_parameters(parameters, field)

        np.testing.assert_allclose(
            np.multiply(parameters.values, field),
            updated.values)

    def test_Update_ParametersViolateGlobalLimits_ReturnUpdatedParametersWithinGlobalBoundsWithTolerance(self):

        _min, _max = 1e-6, 1. - 1e-6
        parameters = ParametersSet([0.1, 0.2, 0.2])
        field = np.array([30., 0.2, 0.0])

        def constraint_calculator(*args):
            return 0.

        engine = self._create_engine(constraint_calculator, bounds=(_min, _max))

        updated = engine.update_parameters(parameters, field)

        _tol = engine._bounds_softening

        self.assertTrue(updated.values[0] >= _max)
        self.assertTrue(updated.values[0] <= _max + _tol)
        self.assertAlmostEqual(0.04, updated.values[1])
        self.assertTrue(updated.values[-1] <= _min)
        self.assertTrue(updated.values[-1] >= _min - _tol)

    def test_Update_ParametersViolateCorrectionLimit_ReturnUpdatedParametersWithinCorrectionLimit(self):
        max_correction = 0.1
        parameters = ParametersSet([0.1, 0.1, 0.2])
        field = np.array([30., 0.5, 0.0])

        def constraint_calculator(*args):
            return 0.

        engine = self._create_engine(constraint_calculator, max_correction=max_correction)

        updated = engine.update_parameters(parameters, field)

        np.testing.assert_allclose(
            np.array([0.1 + max_correction, 0.05, 0.1]),
            updated.values)

    def test_Update_NoMaxCorrection_ReturnParametersSatisfyingConstraintAndGlobalBounds(self):

        parameters = ParametersSet([0.1, 0.1, 0.2])
        field = np.array([30., 0.5, 0.0])
        min_bound = 0.1
        max_bound = 0.9

        constraint_calculator = self.create_constraint_calculator(1.)

        engine = self._create_engine(calculate_exceeded_volume=constraint_calculator, bounds=(min_bound, max_bound))

        updated = engine.update_parameters(parameters, field)

        self.assertAlmostEqual(constraint_calculator(updated.values), 0., places=6)
        self.assertTrue(np.all(updated.values >= min_bound))
        self.assertTrue(np.all(updated.values <= max_bound))

    def _create_engine(self, calculate_exceeded_volume, **kwargs):
        return create_engine('fully_stress_design', calculate_exceeded_volume, **kwargs)

    def create_constraint_calculator(self, expected_parameters_sum):
        def calc(parameters):
            return np.sum(parameters) - expected_parameters_sum

        return calc


if __name__ == '__main__':
    unittest.main()
