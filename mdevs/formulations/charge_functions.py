from abc import ABC, abstractmethod
from mdevs.formulations.models import CalculationConfig
import math


class ChargeCalculator(ABC):
    """
    Abstract class which defines the methods required to calculate the charge and time cost for a given MDEVS instance.
    dilation_factor is based off the assumption of 100% charge being the maximum charge,
    and PERCENTAGE_CHARGE_PER_UNIT being half a charge unit.
    This allows one to experiment with the same overall charge function while dilating it to have different absolute capacities
    Similary, the discretise parameter allows for using the same charge function in rounded or exact charge situations.
    """
    CHARGE_TYPE = ""
    def __init__(self, config: CalculationConfig, discretise=True, name=None) -> None:
        self.config = config
        self.discretise = discretise
        # % difference from an undiscretised maximum charge of 100%
        self.dilation_factor = 2 * self.config.UNDISCRETISED_MAX_CHARGE / 100 * self.config.PERCENTAGE_CHARGE_PER_UNIT 

    def get_charge(self, charge: int, recharge_time: int):
        """
        Determines the state of charge given a charge level and the amount of time it has to charge.
        """
        if recharge_time == 0:
            return charge
        
        if recharge_time < 0:
            return -1
        
        final_charge = self.get_charge_at_time(self.charge_inverse(charge) + recharge_time)
        return final_charge

    @abstractmethod
    def get_charge_at_time(self, t: int) -> int:
        """Returns the charge level from 0% when charging for t units of time."""
        

    @abstractmethod
    def charge_inverse(self, charge: int):
        """
        Returns the time to charge to a certain level. This should work as an inverse to get_charge_at_time
        """
        

class PaperChargeFunction(ChargeCalculator):
    """
    Implements the charging function as described in the paper
    """
    CHARGE_TYPE= 'non-linear'
    def get_charge_at_time(self, t: int) -> int:
        """returns the charge level from 0% when charging for t units of time as outlined in the paper."""
        if t <= 80:
            charge = 2 * t
        elif t < 160:
            charge = 640/3 - ((12800 / 9) / (t - 160 / 3))
        else:
            charge = 200
        charge = math.floor(charge * self.dilation_factor)
        if charge > self.config.MAX_CHARGE:
            raise ValueError(f"Charge level {charge} exceeds the maximum charge level")
        if self.discretise:
            charge = math.floor(charge)
        return charge
    
    def charge_inverse(self, charge: int):
        """Returns the time to charge to a certain level"""
        if charge <= 160 * self.dilation_factor:
            t = charge / (2 * self.dilation_factor)
        elif charge < int(200 * self.dilation_factor):
            t = 160 * (880 * self.dilation_factor - 3 * charge) / ( 1920 * self.dilation_factor - 9 * charge)
            # t = 160 * (charge * self.dilation_factor - 240) / (3 * charge - 640* self.dilation_factor)
        else:
            t = 160
        if self.discretise:
            t = math.ceil(t)
        return t

class LinearChargeFunction(ChargeCalculator):
    """
    Implements a linear charging function with the specified charge speed.
    """
    CHARGE_TYPE= 'linear'
    def __init__(self, percentage_per_minute: int=2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.charge_unit_per_time_unit = percentage_per_minute / self.config.PERCENTAGE_CHARGE_PER_UNIT * self.config.TIME_UNIT_IN_MINUTES

    def get_charge_at_time(self, t: int) -> int:
        """returns the charge level from 0% when charging for t units of time as a linear function."""
        charge = self.charge_unit_per_time_unit * t
        val = charge * self.dilation_factor
        if self.discretise:
            val = math.floor(val)
        return min(val, self.config.MAX_CHARGE)
    
    def charge_inverse(self, charge: int):
        """Returns the time to charge to a certain level"""
        if charge > self.config.MAX_CHARGE * self.dilation_factor:
            charge = self.config.MAX_CHARGE
        t = charge / (self.charge_unit_per_time_unit * self.dilation_factor)
        if self.discretise:
            t = math.ceil(t)
        return t
    

class ConstantChargeFunction(ChargeCalculator):
    """
    Implements a constant-time charging function.
    NOTE: This implementation assumes the recharge delay is the full time required to swap a battery
    In other words, this will always return config.MAX_CHARGE for any non-negative time.
    This is for consistency with the constant-time charging formulation.
    """
    CHARGE_TYPE= 'constant'
    def get_charge(self, charge: int, recharge_time: int):
        if recharge_time < 0:
            return -1
        return self.config.MAX_CHARGE
    
    def get_charge_at_time(self, t: int) -> int:
        """returns the charge level from 0% when charging for t units of time."""
        return self.config.MAX_CHARGE
    
    def charge_inverse(self, charge: int):
        """Returns the time to charge to a certain level"""
        return 0

class NoChargeFunction(ChargeCalculator):
    """
    Implements a no-charge function. Effectively discounts charge as a constraint.
    This is used to solve the model without considering charge constraints.
    """
    CHARGE_TYPE= 'none'
    def get_charge(self, charge: int, recharge_time: int):
        return self.config.MAX_CHARGE
    
    def get_charge_at_time(self, t: int) -> int:
        """returns the charge level from 0% when charging for t units of time."""
        return self.config.MAX_CHARGE
    
    def charge_inverse(self, charge: int):
        """Returns the time to charge to a certain level"""
        return 0