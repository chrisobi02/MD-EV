from enum import Enum
from typing import TypeVar, TypedDict
from dataclasses import dataclass


T = TypeVar("T")

class Flow(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"
    UNDIRECTED = "UNDIRECTED"

UNREACHABLE = -1
VAR_EPS = 1e-3
FORWARD_LABEL_EPS = 1e-5

class Statistics(TypedDict):
    objective: int = None
    runtime: float = None
    label: str = None
    gap: int = None

class FragmentStatistics(Statistics):
    type: str = None
    num_fragments: int = None
    fragment_generation_time: float = None
    initial_timed_network_generation: float = None
    initial_timed_network_nodes: int = None
    initial_timed_network_arcs: int = None

class NonLinearStatistics(FragmentStatistics):
    inspection_time: float = None
    num_lp_iters: int = None
    num_mip_iters: int = None
    mip_runtime: float = None
    lp_runtime: float = None
    infeasible_route_segments: int = 0
@dataclass()
class CalculationConfig:
    """
    Parameters which define the calculations for a given MDEVS instance.
    All distance measures are in metres. 
    The default values are those used in the paper
    """
    UNDISCRETISED_MAX_CHARGE: int = 100
    PERCENTAGE_CHARGE_PER_UNIT: float = 0.5
    PERCENTAGE_CHARGE_PER_METRE: float = 5e-5 # 5% per km
    TIME_UNIT_IN_MINUTES: float = 0.5
    RECHARGE_DELAY_IN_MINUTES: int = 3
    VEHICLE_MOVE_SPEED_PER_MINUTE: float = 1000 / 6 

    @property
    def MAX_CHARGE(self) -> int:
        return int(self.UNDISCRETISED_MAX_CHARGE / self.PERCENTAGE_CHARGE_PER_UNIT)

    @property
    def CHARGE_PER_METRE(self) -> float:
        return self.PERCENTAGE_CHARGE_PER_METRE * self.UNDISCRETISED_MAX_CHARGE / self.PERCENTAGE_CHARGE_PER_UNIT

    @property
    def VEHICLE_MOVE_SPEED_PER_UNIT(self) -> float:
        return self.VEHICLE_MOVE_SPEED_PER_MINUTE * self.TIME_UNIT_IN_MINUTES
    
    @property
    def CHARGE_BUFFER(self) -> float:
        """Delay between arrival and charging"""
        return self.RECHARGE_DELAY_IN_MINUTES / self.TIME_UNIT_IN_MINUTES


@dataclass(frozen=True, order=True)
class ChargeDepot:
    """A Depot which contains both charge and time state."""
    id: int
    time: int
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
class ChargeFragment(Fragment):
    """A copy of a fragment which denotes its starting charge level."""
    start_charge: int

    @classmethod
    def from_fragment(cls, start_charge: int, fragment: 'Fragment | ChargeFragment') -> 'ChargeFragment':
        """Utilty method to create a new ChargeFragment with a given initial charge and fragment information"""
        return cls(
            id=fragment.id,
            jobs=fragment.jobs,
            start_time=fragment.start_time,
            end_time=fragment.end_time,
            start_depot_id=fragment.start_depot_id,
            end_depot_id=fragment.end_depot_id,
            charge=fragment.charge,
            start_charge=start_charge,
        )
    
    @property
    def end_charge(self) -> int:
        """Utility method to return the end state charge."""
        return self.start_charge - self.charge if self.charge <= self.start_charge else ValueError("Negative charge")

    @property
    def start_charge_depot(self) -> ChargeDepot:
        """Utility method to return the start depot."""
        return ChargeDepot(id=self.start_depot_id, time=self.start_time, charge=self.start_charge)
    
    @property
    def end_charge_depot(self) -> ChargeDepot:
        """Utility method to return the end depot."""
        return ChargeDepot(id=self.end_depot_id, time=self.end_time, charge=self.end_charge)


@dataclass(frozen=True, order=True)
class TimedFragment:
    time: int
    id: int
    direction: Flow

@dataclass()
class ChargeDepotStore:
    start: ChargeDepot=None
    end: ChargeDepot=None

@dataclass(frozen=True)
class FrozenChargeDepotStore:
    """For hashing purposes."""
    start: ChargeDepot=None
    end: ChargeDepot=None

@dataclass()
class Label:
    """A dataclass which tracks a given node's"""
    uncompressed_end_depot: ChargeDepot | ChargeDepot
    end_depot: ChargeDepot | ChargeDepot
    flow: float | int
    f_id: int | None  = None# Fragment id
    f_charge: int | None = None
    prev_label: 'Label | None' = None

    def __lt__(self, other: 'Label') -> bool:
        if self.uncompressed_end_depot != other.uncompressed_end_depot:
            return self.uncompressed_end_depot < other.uncompressed_end_depot
        if self.end_depot != other.end_depot:
            return self.end_depot < other.end_depot
        if self.flow != other.flow:
            return self.flow < other.flow
        return False

    def __le__(self, other: 'Label') -> bool:
        if self.uncompressed_end_depot != other.uncompressed_end_depot:
            return self.uncompressed_end_depot <= other.uncompressed_end_depot
        if self.end_depot != other.end_depot:
            return self.end_depot <= other.end_depot
        if self.flow != other.flow:
            return self.flow <= other.flow
        return True

    def __gt__(self, other: 'Label') -> bool:
        if self.uncompressed_end_depot != other.uncompressed_end_depot:
            return self.uncompressed_end_depot > other.uncompressed_end_depot
        if self.end_depot != other.end_depot:
            return self.end_depot > other.end_depot
        if self.flow != other.flow:
            return self.flow > other.flow
        return False

    def __ge__(self, other: 'Label') -> bool:
        if self.uncompressed_end_depot != other.uncompressed_end_depot:
            return self.uncompressed_end_depot >= other.uncompressed_end_depot
        if self.end_depot != other.end_depot:
            return self.end_depot >= other.end_depot
        if self.flow != other.flow:
            return self.flow >= other.flow
        return True


@dataclass()
class Arc:
    end_depot: ChargeDepot | ChargeDepot # Target depot
    start_depot: ChargeDepot | ChargeDepot 
    flow: float # Number of vehicles
    f_id: int | None = None # Fragment id
    f_charge: int | None = None


@dataclass()
class Route:
    """Dataclass which encompasses a route and the many forms it can be expressed"""
    route_list: list[ChargeDepot | Fragment]
    
    @property
    def jobs(self) -> set[Job]:
        return set(j for f in self.route_list if isinstance(f, Fragment) for j in f.jobs)
    
    @classmethod
    def from_timed_fragments(cls, route_list: list[ChargeDepot | TimedFragment]):
        """Creates a Route from a list of timed fragments/depots."""
        raise NotImplementedError()

Location = TypeVar("Location", Building, Job)