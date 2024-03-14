from dataclasses import dataclass
from typing import NamedTuple
from enum import Enum

# speed in metres per hour
VEHICLE_MOVE_SPEED_PER_HOUR = 10000
# Length of 1 time unit in minutes
TIME_UNIT_IN_MINUTES = 0.5
VEHICLE_MOVE_SPEED_PER_UNIT = VEHICLE_MOVE_SPEED_PER_HOUR / 60 * TIME_UNIT_IN_MINUTES
# percentage of max charge used per metre
CHARGE_PER_METRE = 0.05 * 100 / 1000
# percentage of maximum charge per unit of charge
CHARGE_PER_UNIT = 0.5
CHARGE_MAX = 100 / CHARGE_PER_UNIT
# Time cost of swapping batteries over in time units
RECHARGE_TIME = 6


class Flow(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


@dataclass(frozen=True, order=True)
class TimedDepot:
    time: int
    id: int


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


@dataclass(frozen=True)
class Building:
    id: int
    entrance: tuple
    type: str
    capacity: int = None
    location: tuple = None  # Only applies to depots


@dataclass(frozen=True)
class Fragment:
    id: int
    jobs: tuple[Job]  # An ordered tuple of jobs
    start_time: int
    end_time: int
    start_depot_id: int
    end_depot_id: int

    def is_departure(self, timed_depot: TimedDepot) -> bool:
        """
        Determines if a given fragment is an arrival or a departure from the timed node
        Conditions:
        - start_depot_id must match
        - The end time must be less than the timed_depot's time (if it is larger, then it could be an end time)
        """
        # if fragment.start_time > timed_depot.time:
        #     print(f"fragment is ")

        return (
            self.start_depot_id == timed_depot.id and self.end_time < timed_depot.time
        )

    @property
    def verbose_str(self):
        return f"{self.id}:\n   {self.start_time} -> {self.end_time}, {self.start_depot_id} -> {self.end_depot_id}\n    {[j.id for j in self.jobs]}"


# used in the time/space compresion of the network  ("time", int), ("id", int), ("direction", Flow)


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