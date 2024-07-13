from dataclasses import dataclass
from enum import Enum

class Flow(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"

@dataclass
class CalculationConfig:
    """
    Parameters which define the calculations for a given MDEVS instance.
    All distance measures are in metres. 
    The default values are those used in the paper
    """
    UNDISCRETISED_MAX_CHARGE: int = 100
    CHARGE_PER_UNIT: float = 0.5
    PERCENTAGE_CHARGE_PER_METRE: float = 5e-5 # 5% per km
    TIME_UNIT_IN_MINUTES: float = 0.5
    RECHARGE_TIME: int = 6
    VEHICLE_MOVE_SPEED_PER_MINUTE: float = 1000 / 6 

    @property
    def MAX_CHARGE(self) -> int:
        return int(self.UNDISCRETISED_MAX_CHARGE / self.CHARGE_PER_UNIT)

    @property
    def CHARGE_PER_METRE(self) -> float:
        return self.PERCENTAGE_CHARGE_PER_METRE * self.UNDISCRETISED_MAX_CHARGE / self.CHARGE_PER_UNIT

    @property
    def VEHICLE_MOVE_SPEED_PER_UNIT(self) -> float:
        return self.VEHICLE_MOVE_SPEED_PER_MINUTE * self.TIME_UNIT_IN_MINUTES

@dataclass(frozen=True, order=True)
class TimedDepot:
    """A Depot which tracks time state."""
    time: int
    id: int
    
    @property
    def route_str(self):
        return f"D{self.id}"

class ChargeDepot(TimedDepot):
    """A Depot which contains both charge and time state."""
    charge: int
@dataclass(frozen=True)
class Job:
    """Encodes the timing, charge and location information for a job in the MDEVS instance."""
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
    """Encodes a sequence of Jobs which are executed in sequence without visiting a charge station."""
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

@dataclass()
class TimedDepotStore:
    start: TimedDepot=None
    end: TimedDepot=None

@dataclass(order=True)
class Label:
    """A dataclass which tracks a given node's"""
    uncompressed_end_depot: TimedDepot | ChargeDepot
    end_depot: TimedDepot | ChargeDepot
    flow: float | int
    prev_label: 'Label'
    f_id: int | None # Fragment id

@dataclass()
class Arc:
    end_depot: TimedDepot | ChargeDepot # Target depot
    start_depot: TimedDepot | ChargeDepot 
    flow: float # Number of vehicles
    f_id: int | None # Fragment id

@dataclass()
class Route:
    """Dataclass which encompasses a route and the many forms it can be expressed"""
    route_list: list[TimedDepot | ChargeDepot | Fragment]
    
    @property
    def jobs(self) -> set[Job]:
        return set(j for f in self.route_list if isinstance(f, Fragment) for j in f.jobs)
    
    @classmethod
    def from_timed_fragments(cls, route_list: list[TimedDepot | ChargeDepot | TimedFragment]):
        """Creates a Route from a list of timed fragments/depots."""
        raise NotImplementedError()
