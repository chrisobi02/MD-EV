from dataclasses import dataclass
from typing import NamedTuple
from enum import Enum

# speed in metres per hour
VEHICLE_MOVE_SPEED_PER_HOUR = 10000
# Length of 1 time unit in minutes
TIME_UNIT_IN_MINUTES = 0.5
VEHICLE_MOVE_SPEED_PER_UNIT = VEHICLE_MOVE_SPEED_PER_HOUR / 60 * TIME_UNIT_IN_MINUTES
# percentage of maximum charge per unit of charge
CHARGE_PER_UNIT = 0.5
UNDISCRETISED_MAX_CHARGE = 100
CHARGE_MAX = int(UNDISCRETISED_MAX_CHARGE / CHARGE_PER_UNIT)
# percentage of max charge used per metre - 5% of max charge per km
CHARGE_PER_METRE = (0.05 * UNDISCRETISED_MAX_CHARGE) / (1000 * CHARGE_PER_UNIT)
# Time cost of swapping batteries over in time units
RECHARGE_TIME = 6


class Flow(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


@dataclass(frozen=True, order=True)
class TimedDepot:
    time: int
    id: int
    
    @property
    def route_str(self):
        return f"D{self.id}"

class ChargeDepot(TimedDepot):
    charge: int
@dataclass(frozen=True)
class Job:
    id: int
    start_time: int
    end_time: int
    charge: int
    building_start_id: int
    building_end_id: int
    start_location: tuple
    end_location: tuple
    id_offset: int = None #Offset to convert to IP / paper indices
    @property
    def route_str(self):
        return f"{self.id}"
    
    @property
    def offset_id(self):
        """Returns the IP / paper index of the job."""
        return self.id + self.id_offset


@dataclass(frozen=True)
class Building:
    id: int
    entrance: tuple
    type: str
    capacity: int = None
    location: tuple = None  # Only applies to depots

    @property
    def route_str(self):
        return f"D{self.id}"
     
    @property 
    def offset_id(self):
        return self.id


@dataclass(frozen=True)
class Fragment:
    id: int
    jobs: tuple[Job]  # An ordered tuple of jobs
    start_time: int
    end_time: int
    start_depot_id: int
    end_depot_id: int
    charge: int

    @property
    def route_str(self):
        return " -> ".join([str(j.id) for j in self.jobs])

    @property
    def verbose_str(self):
        return f"{self.id}:\n   {self.start_time} -> {self.end_time}, {self.start_depot_id} -> {self.end_depot_id}\n    {[j.id for j in self.jobs]}"



@dataclass(frozen=True, order=True)
class TimedFragment:
    time: int
    id: int
    direction: Flow


@dataclass(frozen=True)
class ContractedFragment:
    jobs: tuple[Job]
    start_depot_ids: list[int]
    end_depot_ids: list[int]

@dataclass(frozen=True)
class Route:
    jobs: tuple[Job]
    start_depot_id: int
    end_depot_id: int
    start_time: int
    end_time: int
@dataclass()
class TimedDepotStore:
    start: TimedDepot=None
    end: TimedDepot=None