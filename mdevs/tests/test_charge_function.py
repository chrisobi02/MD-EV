import pytest
from mdevs.formulations.non_linear_charging import NonLinearFragmentGenerator
from mdevs.formulations.base import *

INPUT_DIR = 'data/instances_regular/I-1-1-50-01.json'

@pytest.fixture
def generator():
    return NonLinearFragmentGenerator(file=INPUT_DIR)

def test_charge_function_inverse(generator: NonLinearFragmentGenerator):
    assert generator.get_charge_at_time(0) == 0
    for charge in range(generator.config.MAX_CHARGE + 1):
        time = generator.charge_inverse(charge)
        assert abs(charge - generator.get_charge_at_time(time)) <= 1

def test_get_charge(generator: NonLinearFragmentGenerator):
    """Ensure get_charge_at_time correctly calculates charge"""
    import pdb; pdb.set_trace()
    for charge in range(0, generator.config.MAX_CHARGE + 1, 2):
        for alternate in range(0, generator.config.MAX_CHARGE - charge + 1, 2):
            recharge_time = generator.charge_inverse(charge + alternate)
            start_time = generator.charge_inverse(charge)
            assert abs(generator.get_charge(charge, recharge_time - start_time)  - charge - alternate) <= 1

def test_specific_calculation(generator: NonLinearFragmentGenerator):
    """Ensure get_charge_at_time correctly calculates charge"""
    import pdb; pdb.set_trace()
    
    charge = generator.get_charge(150, 1)
    assert charge == 152