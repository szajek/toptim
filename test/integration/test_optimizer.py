import unittest
import numpy as np
from toptim.optimizer import create_optimizer


"""
glossary:
 - Sed/SED - StrainsEnergyDensity
"""


def create_exceeded_volume_calculator(expected_volume):
    def calc(parameters):
        return np.sum(parameters) - expected_volume
    return calc


class OptimizerTest(unittest.TestCase):
    def test_Solve_SedIsFixed_ReturnParameterForTheBiggestSedMovedToUpperBound(self):

        def calculate_strain_energy_density(parameters):
            return np.array([3., 1.])

        exceeded_volume_calculator = create_exceeded_volume_calculator(1.)

        result = self._create_and_solve(np.full((2,), 0.5), calculate_strain_energy_density, exceeded_volume_calculator)

        self._check_result_and_constraint(np.array([1., 0.]), result)
        self._check_volume_constraint(exceeded_volume_calculator, result)

    def test_Solve_SedEqualsParameter_ReturnParametersMovedToExtremes(self):

        def calculate_strain_energy_density(parameters):
            return np.array(parameters)

        exceeded_volume_calculator = create_exceeded_volume_calculator(1.)

        result = self._create_and_solve([0.35, 0.45], calculate_strain_energy_density, exceeded_volume_calculator)

        self._check_result_and_constraint(np.array([0., 1.]), result)
        self._check_volume_constraint(exceeded_volume_calculator, result)

    def test_Solve_SedIsInversionOfParameterValue_ReturnParametersMovedToCenter(self):

        def calculate_strain_energy_density(parameters):
            return np.power(parameters, -1.)

        exceeded_volume_calculator = create_exceeded_volume_calculator(1.)

        result = self._create_and_solve([0.1, 0.9], calculate_strain_energy_density, exceeded_volume_calculator)

        self._check_result_and_constraint(np.array([0.5, 0.5]), result)
        self._check_volume_constraint(exceeded_volume_calculator, result)

    def _check_result_and_constraint(self, expected_result, output):
        np.testing.assert_allclose(
            expected_result,
            output.values,
            atol=1e-2,
        )

    def _check_volume_constraint(self, exceeded_volume_calculator, output):
        self.assertAlmostEqual(exceeded_volume_calculator(output.values), 0., places=4)

    def _create_and_solve(self, initial_value, calculate_strain_energy_density, exceeded_volume_calculator):
        o = self._create_optimizer(initial_value, calculate_strain_energy_density, exceeded_volume_calculator)
        return o.solve()

    def _create_optimizer(self, *args):
        return create_optimizer('fully_stress_design', *args)
