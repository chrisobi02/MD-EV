import json
import os
import time
from collections import defaultdict
from gurobipy import Model, GRB
import pandas as pd
from typing import TypeVar
from abc import ABC, abstractmethod
from itertools import product
from gurobipy import Model, GRB, quicksum
from dataclasses import dataclass
from enum import Enum
import heapq
from typing import TypedDict

from mdevs.formulations.models import *
from mdevs.formulations.charge_functions import ChargeCalculator


class BaseMDEVCalculator(ABC):
    """
    Base class used to centralise data calculations across all model implementations
    """
    
    def __init__(
            self,
            file: str, 
            charge_calculator_class: ChargeCalculator,
            config=CalculationConfig,
            charge_calculator_kwargs: dict={},
        ) -> None:
        """
        params: dictionary which specify overrides to the default CalculationParameter.
        """
        self.base_dir = os.path.dirname(file)
        self.data = json.load(open(file))
        self.buildings_by_id: dict[int, Building] = self.to_dataclass_by_id(self.data["buildings"], Building)
        self.buildings: list[Building] = list(self.buildings_by_id.values())
        self.depots_by_id: dict[int, Building] = {
            building.id: building
            for building in self.buildings
            if building.type == "depot"
        }
        self.depots: list[Building] = list(self.depots_by_id.values())
        self.jobs_by_id: dict[int, Job] = self.to_dataclass_by_id(self.data["jobs"], Job, id_offset=len(self.depots_by_id))
        self.jobs: list[Job] = list(self.jobs_by_id.values())
        self.config = config
        self.charge_calculator: ChargeCalculator = charge_calculator_class(config=config, **charge_calculator_kwargs)
        self.statistics: Statistics = Statistics(label=self.data['label'])

    def to_dataclass_by_id(
        self, json_data: list[dict], dataclass: T, id_field: str = "id", **kwargs
    ) -> dict[int, T]:
        return {
            data[id_field] if id_field else i: dataclass(
                **{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()} | kwargs
            )
            for i, data in enumerate(json_data)
        }
    
    def distance(self, start: list, end: list) -> int:
        """Manhattan norm."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def get_internal_distance(self, location: tuple) -> int:
        """
        Returns the internal distance from a location to the entrance of a building.
        Internal distances are stored in relation to the entrance.
        """
        return sum(location)

    def get_internal_job_distance(self, job: Job, start=True):
        """
        Returns the distance of the job to the entrance of its building. 
        start keyword dictates whether it's the end of the job or the start.
        """
        return self.get_internal_distance(job.start_location if start else job.end_location)
  
    def distance_to_time(self, distance: int) -> int:
        return round(distance / self.config.VEHICLE_MOVE_SPEED_PER_UNIT)

    def distance_to_charge(self, distance: int) -> int:
        return round(distance * self.config.CHARGE_PER_METRE)
        
    def generate_building_distance_matrix(self) -> list[list[int]]:
        """Generates the distance matrix between building entrances. [building_id][building_id]"""
        self.building_distance = [
            [0 for _ in range(len(self.buildings))] for _ in range(len(self.buildings))
        ]

        for i, building in enumerate(self.buildings):
            for j, other_building in enumerate(self.buildings):
                if building.id == other_building.id:
                    continue
                distance = self.distance(building.entrance, other_building.entrance)
                self.building_distance[i][j] = distance
        return self.building_distance
      
    def get_job_to_depot_distance(self, job: Job, depot: Building, is_job_origin: bool) -> int:
        """Returns the distance between a job and a building entrance."""
        job_building_id = job.building_end_id if is_job_origin else job.building_start_id
        if job_building_id == depot.id:
            location = job.end_location if is_job_origin else job.start_location 
            # Same building
            distance = self.distance(location, depot.location)
        else:
            distance = (
                self.building_distance[job_building_id][depot.id]
                + self.get_internal_job_distance(job, start=not is_job_origin)
                + self.get_internal_distance(depot.location)
            )
        return distance
    
    def get_job_distance(self, job_1: Job, job_2: Job) -> int:
        """Returns the distance between two jobs."""
        if job_1.building_end_id == job_2.building_start_id:
            distance = self.distance(job_1.end_location, job_2.start_location)
        else:
            distance = (
                self.building_distance[job_1.building_end_id][job_2.building_start_id]
                + self.get_internal_job_distance(job_1, start=False)
                + self.get_internal_job_distance(job_2, start=True)
            )
        return distance

    def get_charge(self, charge: int, t: int) -> int:
        return self.charge_calculator.get_charge(charge, t)

    def get_charge_at_time(self, t: int) -> int:
        return self.charge_calculator.get_charge_at_time(t)

    def charge_inverse(self, charge: int):
        return self.charge_calculator.charge_inverse(charge)

    def read_solution(self, instance_type:str=None, sheet_name=None, charging_style='constant-time') -> tuple[list[list[int]], list[str]]:
        """Reads the solution given by the paper into a list of their routes which has waiting arcs and fragments."""
        try:
            data = pd.read_excel(r"mdevs/data/mdevs_solutions.xlsx", sheet_name=sheet_name)
        except:
            data = pd.read_excel(r"data/mdevs_solutions.xlsx", sheet_name=sheet_name)
        if not sheet_name:
            for sheet in data:
                if instance_type in sheet:
                    data = data[sheet]
                    break

        curr_sol_str = data.query(
            f"ID_instance == {self.data['ID']} and battery == '{charging_style}'"
        )["solution"].values[0]
        solution_routes = []
        string_solution_routes=[]
        # Convert the matches to integers and find the maximum since sometimes there is a weird offset
        num_depots = len(self.depots_by_id)
        num_jobs = len(self.jobs_by_id)
        for route in curr_sol_str.split(","):
            current_route = []
            print(route[1:-1].split(">"))
            for location in route[1:-1].split(">"):
                depot_id = None
                if "D" in location:
                    depot_id = int(location[1:]) - 1
                elif "S" in location:
                    depot_id = int(location[1:]) - num_jobs - num_depots - 1
                else:
                    job_id = int(location) - num_depots - 1
                    current_route.append(self.jobs_by_id[job_id])

                if depot_id is not None:
                    if len(current_route) == 0 or current_route[-1] != depot_id:
                        current_route.append(self.depots_by_id[depot_id])
            # Convert route into its fragments
            # Remove double depots
            if all(isinstance(d, Building) for d in current_route[:2]):
                current_route = current_route[1:]
            if all(isinstance(d, Building) for d in current_route[-2:]):
                current_route = current_route[:-1]
            solution_routes.append(current_route)
            string_solution_routes.append(self.stringify_route(current_route))
        return solution_routes, string_solution_routes

    def stringify_route(self, route: list) -> list[str]:
        """Converts a route into a string"""
        return " -> ".join([loc.route_str for loc in route])
    
class BaseFragmentGenerator(BaseMDEVCalculator):
    TYPE = None
    def __init__(
            self,
            file: str, 
            charge_calculator_class: ChargeCalculator,
            config=CalculationConfig(),
            charge_calculator_kwargs: dict={},
        ) -> None:
        super().__init__(file, charge_calculator_class, config=config, charge_calculator_kwargs=charge_calculator_kwargs)
        self.generate_cost_matrices()
        self.fragment_set: set[Fragment] = set()
        self.fragments_by_id: dict[int, Fragment] = {}
        self.fragment_vars_by_charge_fragment: dict[ChargeFragment, GRB.Var] = {}
        self.charge_depots_by_depot: dict[int, list[ChargeDepot]] = defaultdict(list)
        self.timed_fragments_by_charge_depot: dict[ChargeDepot, set[TimedFragment]] = defaultdict(set[TimedFragment])
        self.charge_depots_by_charge_fragment: dict[ChargeFragment, ChargeDepotStore] = defaultdict(ChargeDepotStore)
        self.charge_fragments: set[ChargeFragment] = set()
        self.recharge_arcs: set[FrozenChargeDepotStore] = set()
        self.recharge_arcs_by_charge_depot: dict[ChargeDepot, set[FrozenChargeDepotStore]] = defaultdict(
            set[FrozenChargeDepotStore]
        )
        self.all_arcs_by_charge_depot = defaultdict(set[FrozenChargeDepotStore | ChargeFragment])
        self.model = Model("fragment_network")
        self.minimum_charge_by_time_by_depot: dict[tuple[int, int], int] = {}
        self.statistics: FragmentStatistics = FragmentStatistics(type=self.TYPE, **self.statistics)

    def generate_cost_matrices(self):
        self.generate_building_distance_matrix()
        self.generate_job_cost_matrix()
        self.generate_job_to_depot_matrices()
        self.generate_depot_to_job_matrices()

    @property
    def charge_depots(self) -> set[ChargeDepot]:
        return set(
            charge_depot for charge_depots in self.charge_depots_by_depot.values() for charge_depot in charge_depots
        )

    def generate_job_to_depot_matrices(self) -> None:
        """Generates the charge and time cost for going from a job to a depot. [job_id][depot_id]."""
        self.job_to_depot_distance_matrix = [
            [0 for _ in range(len(self.depots_by_id))] for _ in range(len(self.jobs))
        ]
        self.job_to_depot_charge_matrix = [
            [0 for _ in range(len(self.depots_by_id))] for _ in range(len(self.jobs))
        ]
        self.job_to_depot_time_matrix = [
            [0 for _ in range(len(self.depots_by_id))] for _ in range(len(self.jobs))
        ]
        for i, job in enumerate(self.jobs):
            for j, depot in self.depots_by_id.items():
                distance = self.get_job_to_depot_distance(job, depot, is_job_origin=True)
                self.job_to_depot_distance_matrix[i][j] = distance
                self.job_to_depot_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.job_to_depot_time_matrix[i][j] = self.distance_to_time(distance)

    def generate_depot_to_job_matrices(self) -> None:
        """Generates the charge and time cost for going from a depot to a job"""
        self.depot_to_job_distance_matrix = [
            [0 for _ in range(len(self.jobs))] for _ in range(len(self.depots_by_id))
        ]
        self.depot_to_job_charge_matrix = [
            [0 for _ in range(len(self.jobs))] for _ in range(len(self.depots_by_id))
        ]
        self.depot_to_job_time_matrix = [
            [0 for _ in range(len(self.jobs))] for _ in range(len(self.depots_by_id))
        ]

        for i, depot in self.depots_by_id.items():
            for j, job in enumerate(self.jobs):
                distance = self.get_job_to_depot_distance(job, depot, is_job_origin=False)
                self.depot_to_job_distance_matrix[i][j] = distance
                self.depot_to_job_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.depot_to_job_time_matrix[i][j] = self.distance_to_time(distance)

    def generate_job_cost_matrix(self):
        """Generates the charge and time cost"""
        self.job_distance_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))]
        self.job_charge_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))]
        self.job_time_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))]
        for i, job in enumerate(self.jobs):
            for j, other_job in enumerate(self.jobs):
                distance = self.get_job_distance(job, other_job)
                self.job_distance_matrix[i][j] = distance
                self.job_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.job_time_matrix[i][j] = self.distance_to_time(distance)

    def _get_jobs_reachable_from(
            self,
            charge: int,
            job: Job,
        ) -> set[Fragment]:
        """
        Generates a fragment starting at a given job and depot with a given charge at a given time.
        The three conditions that must be satisfied are:
        1. The candidate job must be reachable before its start time.
        2. The candidate job must be able to reach a depot from the new job.
        3. A detour to a depot will not result in the vehicle reaching the job with a higher charge than it would have directly.
        """
        reachable_jobs = []
        job_end_time = job.end_time
        for next_job in self.jobs:
            # 1.
            arrival_time = job_end_time + self.job_time_matrix[job.id][next_job.id]
            if next_job.start_time < arrival_time:
                continue
            # 2.
            charge_cost = next_job.charge + self.job_charge_matrix[job.id][next_job.id] + min(
                self.job_to_depot_charge_matrix[next_job.id][depot.id]
                for depot in self.depots_by_id.values()
            )
            if charge < charge_cost:
                continue
            
            # Find the depot with the minimum time to traverse from job, then back to next job
            closest_depot = min(
                (
                    depot for depot in self.depots
                    if self.job_to_depot_time_matrix[job.id][depot.id] <= charge
                ), 
                key=(
                    lambda depot: self.job_to_depot_time_matrix[job.id][depot.id]
                    + self.depot_to_job_time_matrix[depot.id][next_job.id]
                )
            )
            recharge_time = (
                next_job.start_time 
                - job_end_time
                - self.job_to_depot_time_matrix[job.id][closest_depot.id] 
                - self.depot_to_job_time_matrix[closest_depot.id][next_job.id]
                - self.config.CHARGE_BUFFER
            )
            
            # 3.
            detour_charge = (
                self.get_charge(charge - self.job_to_depot_charge_matrix[job.id][closest_depot.id], recharge_time)
                - self.depot_to_job_charge_matrix[closest_depot.id][next_job.id]
            )
            direct_charge = charge - self.job_charge_matrix[job.id][next_job.id]
            if detour_charge >= direct_charge:
                continue
            reachable_jobs.append(next_job)

        return reachable_jobs
    
    def generate_fragments(self, file: str=None) -> set[tuple[Job]]:
        """
        Enumerates all possible fragments which satisfy the following requirements:
        1. The fragment leaves from a depot and returns to a depot with non-zero charge
        2. All jobs included within the fragment can be executed in sequence
        3. Any job which takes longer to reach than the time to return to a depot, 
           recharge to the same or higher level and reach the job before its start is ignored
        """
        # Only starting points for any fragment: any building in depots
        # only generate starting at the latest possible leaving time from depot (deadhanging is allowed)
        if file is not None:
            try:
                self.fragment_set = self.read_fragments(file)
                self.fragments_by_id = {fragment.id: fragment for fragment in self.fragment_set}
                return self.fragment_set
            except FileNotFoundError:
                print(f"File {file} not found. Generating fragments...")

        job_set: set[tuple[Job]] = set()
        time0 = time.time()
        self.feasible_job_sequences = defaultdict(set)
        for job in self.jobs:
            # get the minimum depot to job time and charge (the same since charge is prop to time)
            closest_depot = min(
                self.depots_by_id.values(),
                key=lambda depot: self.depot_to_job_time_matrix[depot.id][job.id],
            )
            charge = self.config.MAX_CHARGE - self.depot_to_job_charge_matrix[closest_depot.id][job.id] - job.charge
            start_time = job.start_time - self.depot_to_job_time_matrix[closest_depot.id][job.id]
            if start_time < 0:
                continue
            self._generate_job_sequence_starting_at(job_set, [job], job, charge)
        # Now create all combinations of start/ends which can be executed
        fragment_counter = 0
        for job_sequence in job_set:
            sequence_charge = sum(
                self.job_charge_matrix[job.id][next_job.id] for job, next_job in zip(job_sequence, job_sequence[1:])
            ) + sum(job.charge for job in job_sequence)

            start_job = job_sequence[0]
            end_job = job_sequence[-1]
            for start_id, end_id in product(self.depots_by_id, repeat=2):
                # check charge is acceptable
                total_charge = (
                    self.depot_to_job_charge_matrix[start_id][start_job.id] 
                    + sequence_charge 
                    + self.job_to_depot_charge_matrix[end_job.id][end_id]
                )
                start_time = start_job.start_time - self.depot_to_job_time_matrix[start_id][start_job.id]
                if total_charge <= self.config.MAX_CHARGE and start_time >= 0:
                    self.fragment_set.add(
                        Fragment(
                            id=fragment_counter,
                            jobs=job_sequence,
                            start_time=start_time,
                            end_time=(
                                end_job.end_time 
                                + self.job_to_depot_time_matrix[end_job.id][end_id] 
                                + self.config.CHARGE_BUFFER
                            ),
                            start_depot_id=start_id,
                            end_depot_id=end_id,
                            charge=total_charge,
                        )
                    )
                    fragment_counter += 1
                
        self.fragments_by_id = {fragment.id: fragment for fragment in self.fragment_set}
        self.statistics["num_fragments"] = len(self.fragment_set)
        self.statistics["fragment_generation_time"] = time.time() - time0
        self.job_sequences = job_set
        return job_set

    def _generate_job_sequence_starting_at(
        self,
        job_set: set[Fragment],
        current_jobs: list[Job],
        job: Job,
        charge: int,
    ) -> set[Fragment]:
        """
        Generates a fragment starting at a given job and depot with a given charge at a given time.
        """
        # Get all jobs which can be reached from the current job
        # Add this partial part of the journey as a fragment.
        for id, depot in self.depots_by_id.items():
            if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge:
                job_set.add(tuple(current_jobs))
                break
        reachable_jobs = self._get_jobs_reachable_from(charge, job)
        for next_job in reachable_jobs:
            # manage memory
            # Otherwise, generate a fragment starting at the next job
            self._generate_job_sequence_starting_at(
                job_set,
                current_jobs + [next_job],
                next_job,
                charge - next_job.charge - self.job_charge_matrix[job.id][next_job.id],
            )

        return job_set

    def get_fragments_from(
            self, 
            start: ChargeDepot,
            current_route: list, 
            fragments_by_charge_depot: dict[ChargeDepot, set[tuple[ChargeDepot, int]]], 
            flow_by_waiting_arc_start: dict[ChargeDepot, dict[ChargeDepot, int]]
        ) -> list[Fragment | ChargeDepot]:
        """Traverse the waiting arcs and fragments to get the route from a given start depot."""

        if len(fragments_by_charge_depot[start]) != 0:
            next_depot, f_id = fragments_by_charge_depot[start].pop()
            current_route.extend([self.fragments_by_id[f_id], next_depot])

        elif len(flow_by_waiting_arc_start[start]) != 0:
            next_depot = min(flow_by_waiting_arc_start[start])
            flow_by_waiting_arc_start[start][next_depot] -= 1
            if flow_by_waiting_arc_start[start][next_depot] == 0:
                flow_by_waiting_arc_start[start].pop(next_depot)
        else:
            return current_route
        self.get_fragments_from(next_depot, current_route, fragments_by_charge_depot, flow_by_waiting_arc_start)
        return current_route

    def build_model(self):
        """Solves the network flow mip"""
        self.model = Model("fragment_network")
        self.fragment_vars_by_charge_fragment = {
            cf: self.model.addVar(vtype=GRB.BINARY, name=f"f_{cf.id}_c_{cf.start_charge}")
            for cf in self.charge_fragments
        }

        self.recharge_arc_var_by_depot_store = {
            recharge_arc: self.model.addVar(vtype=GRB.INTEGER, name=f"w_{recharge_arc.start}_{recharge_arc.end}")
            for recharge_arc in self.recharge_arcs
        }
        self.starting_counts = {
            depot: self.model.addVar(vtype=GRB.INTEGER, name=f"sc_{depot}")
            for depot in self.charge_depots_by_depot
        }
        self.finishing_counts = {
            depot: self.model.addVar(vtype=GRB.INTEGER, name=f"fc_{depot}")
            for depot in self.charge_depots_by_depot
        }

        self.add_flow_balance_constraints()

        # Coverage
        self.charge_fragments_by_job: defaultdict[Job, set[ChargeFragment]] = defaultdict(set[ChargeFragment])
        for cf in self.charge_fragments:
            for job in cf.jobs:
                self.charge_fragments_by_job[job].add(cf)
        
        self.coverage = {}
        for job in self.jobs:
            self._add_coverage_constraint(job)

        self.start_depot_limits = {
            depot: self.model.addConstr(
                self.starting_counts[depot.id]
                <= depot.capacity,
                name=f"depot_limit_{depot.id}",
            )
            for depot in self.depots
            if depot.capacity != -1
        }
        self.end_depot_limits = {
            depot: self.model.addConstr(
                self.finishing_counts[depot.id]
                <= depot.capacity,
                name=f"depot_limit_{depot.id}",
            )
            for depot in self.depots
            if depot.capacity != -1
        }

        # end same as start
        self.vehicle_conservation = self.model.addConstr(
            quicksum(self.starting_counts.values())
            == quicksum(self.finishing_counts.values())
        )
        self.model.setObjective(quicksum(self.starting_counts.values()))

    def _add_coverage_constraint(self, job: Job) -> None:
        self.coverage[job] = self.model.addConstr(
            quicksum(
                self.fragment_vars_by_charge_fragment[cf]
                for cf in self.charge_fragments_by_job[job]
            )
            == 1,
            name=f"job_{job.id}",
        )

    def set_solution(self, solution: list[set[ChargeFragment]], n_vehicles=None) -> None:
        all_solution_fragments = set(f for s in solution for f in s)
        if n_vehicles is not None:
            self.model.addConstr(
                quicksum(self.starting_counts.values()) == n_vehicles,
                name="vehicle_count",
            )

        for charge_fragment in self.charge_fragments:
            if charge_fragment not in all_solution_fragments:
                self.model.remove(self.fragment_vars_by_charge_fragment[charge_fragment])
                del self.fragment_vars_by_charge_fragment[charge_fragment]
        
        charge_routes = self.get_validated_timed_solution(solution, n_vehicles)
        waiting_arc_flows: dict[tuple[ChargeDepot, ChargeDepot], int] = defaultdict(int)
        # Find the waiting arcs for each route and add those in too
        for route in charge_routes:
            route = route.copy()
            # Iterate until we hit a TimedFragment, then we reset
            prev_node = route.pop(0)
            while len(route) != 0:
                curr_node = route.pop(0)
                if isinstance(curr_node, Fragment):
                    # Reset the arc
                    prev_node = route.pop(0)
                    continue
                waiting_arc_flows[
                    FrozenChargeDepotStore(start=prev_node, end=curr_node)
                ] += 1
                prev_node = curr_node

        for arc in self.recharge_arc_var_by_depot_store:
            self.model.addConstr(
                self.recharge_arc_var_by_depot_store[arc] == waiting_arc_flows.get(arc, 0), name=f"set_sol_{arc}"
            )     

    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("fragment_network.ilp")
        else:
            self.statistics.update(
                {
                    "objective": self.model.objval,
                    "runtime": self.model.Runtime,
                    "gap": self.model.MIPGap,
                }
            )

    def forward_label(
            self,
            solution_charge_fragments: list[tuple[ChargeFragment, float]],
            waiting_arcs: list[tuple[ChargeDepot, ChargeDepot, float]]
        ) -> list[Route]:
        """
        Sequences the solution routes using a forward labelling procedure.
        """

        label_heap: list[Label] = [] # min heap of Labels
        routes = []
        # Get the first and last ChargeDepot for each Depot 
        start_depots = {min(depot_list) for depot_list in self.charge_depots_by_depot.values()}
        end_depots = {max(depot_list) for depot_list in self.charge_depots_by_depot.values()}

        arcs_by_charge_depot: dict[ChargeDepot, list[Arc]] = defaultdict(list[Arc])
        for cf, flow in solution_charge_fragments:
            # Retrieve the timed depot it starts and ends at
            start_depot, end_depot = (
                self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
            )
            arcs_by_charge_depot[start_depot].append(
                Arc(start_depot=start_depot, end_depot=end_depot, f_id=cf.id, f_charge=cf.start_charge, flow=flow)
            )
                     
            if start_depot not in start_depots:
                continue
            # Decompress the network such that a min heap on time is real. 
            # To do this, we need to map these depots back to their true time.
            uncompressed_end_depot = ChargeDepot(time=cf.end_time, id=cf.end_depot_id, charge=end_depot.charge)
            
            start_label = Label(
                    flow=flow,
                    prev_label=Label(
                        uncompressed_end_depot=uncompressed_end_depot,
                        end_depot=start_depot,
                        prev_label=None,
                        flow=flow,
                        f_id=None,
                    ),
                    end_depot=end_depot,
                    f_id=cf.id,
                    f_charge=cf.start_charge,
                    uncompressed_end_depot=uncompressed_end_depot
                )
            heapq.heappush(
                label_heap,
                start_label 
            )

        # Add any start labels
        for start_depot, end_depot, flow in waiting_arcs:
            arcs_by_charge_depot[start_depot].append(
                Arc(start_depot=start_depot, end_depot=end_depot, f_id=None, flow=flow)
            )
            if start_depot not in start_depots:
                continue

            start_label = Label(uncompressed_end_depot=start_depot, flow=flow, end_depot=start_depot, prev_label=None, f_id=None)
            heapq.heappush(
                label_heap, 
                Label(uncompressed_end_depot=start_depot, flow=flow, prev_label=start_label, end_depot=end_depot, f_id=None)
            )

        while len(label_heap) != 0:
            label = heapq.heappop(label_heap)
            label_flow = label.flow
            arcs = arcs_by_charge_depot[label.end_depot]
            if len(arcs) == 0:
                if label.end_depot not in end_depots:
                    raise Exception("Route ends before the last depot time - very bad:(")
                # assemble the route backwards
                route = []
                while label.prev_label is not None:
                    route.append(label.end_depot)
                    if label.f_id is not None:
                        route.append(
                            ChargeFragment.from_fragment(
                                start_charge=label.f_charge, fragment=self.fragments_by_id[label.f_id]
                            )
                        )
                    label = label.prev_label
                route.append(label.end_depot)
                routes.append(Route(route_list=route[::-1]))

            arc_idx = 0
            while len(arcs) != 0 and label_flow > FORWARD_LABEL_EPS:
                arc = arcs[arc_idx]
                delta = min(arc.flow, label_flow)
                uncompressed_end_depot = arc.end_depot

                if arc.f_id is not None:
                    fragment = self.fragments_by_id[arc.f_id]
                    uncompressed_end_depot = ChargeDepot(
                        time=fragment.end_time, id=fragment.end_depot_id, charge=end_depot.charge
                    )
                heapq.heappush(
                    label_heap,
                    Label(
                        uncompressed_end_depot=uncompressed_end_depot,
                        end_depot=arc.end_depot,
                        prev_label=label,
                        flow=delta,
                        f_id=arc.f_id,
                        f_charge=arc.f_charge
                    )
                )

                # Decrement flow on the arc and the label
                arc.flow -= delta
                label_flow -= delta

                if arc.flow < FORWARD_LABEL_EPS:
                    arcs.pop(arc_idx)
                else:
                    arc_idx += 1

                if label_flow < FORWARD_LABEL_EPS:
                    break
        return routes                  

    def convert_solution_to_fragments(self, instance_type:str=None, sheet_name=None, charging_style='constant-time') -> tuple[list[list[int]], list[str]]:
        """Reads the solution given by the paper into a list of their routes which has waiting arcs and fragments."""
        solution_routes, string_solution_routes = self.read_solution(
            instance_type=instance_type, sheet_name=sheet_name, charging_style=charging_style
        )
        
        fragment_routes = []
        for current_route in solution_routes:
            route_fragment_ids = self.convert_route_to_fragments(current_route)
            fragment_routes.append(route_fragment_ids)
        return fragment_routes, string_solution_routes

    def get_fragment_id(self, candidate: dict):
        """Retrieves a fragment based on its jobs and start/end depot"""
        job_tuple = tuple(candidate["jobs"])
        for fragment in self.fragment_set:
            if (
                fragment.jobs == job_tuple
                and fragment.start_depot_id == candidate.get("start_depot_id")
                and fragment.end_depot_id == candidate.get("end_depot_id")
            ):
                return fragment.id
        else:
            fragments_with_same_jobs = [
                fragment
                for fragment in self.fragment_set
                if set(job_tuple).issubset(fragment.jobs)
            ]
            actual_charge = (
                self.config.MAX_CHARGE
                - self.depot_to_job_charge_matrix[candidate.get("start_depot_id")][job_tuple[0].id]
                - sum(job.charge for job in job_tuple[:-1])
                - sum(self.job_charge_matrix[j1.id][j2.id] for j1, j2 in zip(job_tuple, job_tuple[1:]))
            )
            reachable_jobs = self._get_jobs_reachable_from(actual_charge, job_tuple[0])
            assert job_tuple[1] in reachable_jobs
            print(
                (
                    f"no such fragment exists:\n"
                    f"jobs: {[j.id for j in candidate['jobs']]}\n"
                    f"start_depot_id: {candidate.get('start_depot_id')}\n"
                    f"end_depot_id: {candidate.get('end_depot_id')}"
                )
            )

    def convert_route_to_fragments(self, route: list[Job | Building]) -> list[int]:
        """Converts a route from the paper's data format into its fragments"""
        fragment_ids = []
        current_fragment = {
            "jobs": [],
            "start_depot_id": None,
            "start_time": None,
            "end_depot_id": None,
            "end_time": None,
        }

        fragments_in_order = []
        for i, location in enumerate(route):
            match location:
                case Building():
                    # Starting point
                    if current_fragment["start_depot_id"] is None:
                        current_fragment["start_depot_id"] = location.id
                    else:
                        # figure out if can recharge before the next task
                        current_fragment["end_depot_id"] = location.id
                        frag_id = self.get_fragment_id(current_fragment)
                        if frag_id is not None:
                            fragments_in_order.append(frag_id)
                            fragment_ids.append(frag_id)
                        else:
                            if len(current_fragment["jobs"]) == 0:
                                continue
                            self.validate_fragment(
                                Fragment(
                                    id=-1,
                                    jobs=tuple(current_fragment["jobs"]),
                                    start_time=(
                                        current_fragment["jobs"][0].start_time - self.depot_to_job_time_matrix[current_fragment["start_depot_id"]][current_fragment["jobs"][0].id]
                                    ),
                                    end_time=(
                                        current_fragment["jobs"][-1].end_time + self.job_to_depot_time_matrix[current_fragment["jobs"][-1].id][location.id] + self.config.CHARGE_BUFFER
                                    ),
                                    start_depot_id=current_fragment["start_depot_id"],
                                    end_depot_id=location.id,
                                    charge=(
                                        sum(job.charge for job in current_fragment["jobs"])
                                        + self.depot_to_job_charge_matrix[current_fragment["start_depot_id"]][current_fragment["jobs"][0].id]
                                        + self.job_to_depot_charge_matrix[current_fragment["jobs"][-1].id][location.id]
                                        + sum(
                                            self.job_charge_matrix[j1.id][j2.id]
                                            for j1, j2 in zip(current_fragment["jobs"], current_fragment["jobs"][1:])
                                        )
                                    )
                                ),
                                self.config.MAX_CHARGE,
                                0
                            )
                            raise AttributeError(f"Fragment {current_fragment} not found")
                        current_fragment = {
                            "jobs": [],
                            "start_depot_id": location.id,
                            "end_depot_id": None,
                        }
                case Job():
                    current_fragment["jobs"].append(location)
            
        return fragment_ids

    def validate_timed_network(self):
        """
        Validates the timed network to ensure it is feasible.
        For a timed network to be feasible:
        - No timed depot can have fragments which start after its time
        - Each timed depot cannot have a time earlier than the previous timed depot
        """
        for depot_id, charge_depots in self.charge_depots_by_depot.items():
            prev_td = None
            for td in sorted(charge_depots):
                if prev_td:
                    assert td.time >= prev_td.time
                    if td.time <= prev_td.time:
                        print(f"Depot {td} has a time earlier than the previous depot.")
                        return False
                fragments = self.timed_fragments_by_charge_depot[td]
                # Check the following:
                # min end time >= max start time
                # no start time later than the current depot time and no earlier than the previous time.
                departure_fragments = [f for f in fragments if f.direction == Flow.DEPARTURE]
                arrival_fragments = [f for f in fragments if f.direction == Flow.ARRIVAL]
                if len(arrival_fragments) != 0:
                    assert all(prev_td.time <= tf.time <= td.time for tf in departure_fragments)
                if len(departure_fragments) != 0:
                    assert all(prev_td.time <= tf.time <= td.time for tf in arrival_fragments)
                if len(departure_fragments) != 0 and len(arrival_fragments) != 0:
                   max_arr = max(tf.time for tf in arrival_fragments) 
                   min_dep = min(tf.time for tf in departure_fragments)
                   assert max_arr <= min_dep
                prev_td=td

    def validate_fragment(self, fragment: Fragment, charge: int, prev_time: int) -> tuple[int, int, bool]:
        """Validates a given fragment at a time-charge level is feasible."""
        is_valid = True
        if prev_time > fragment.start_time:
            print(f"Fragment {fragment} starts too late")
            raise Exception()
        cumulative_charge = (
            sum(job.charge for job in fragment.jobs) 
            + self.depot_to_job_charge_matrix[fragment.start_depot_id][fragment.jobs[0].id]
            + self.job_to_depot_charge_matrix[fragment.jobs[-1].id][fragment.end_depot_id]
            + sum(self.job_charge_matrix[j1.id][j2.id] for j1, j2 in zip(fragment.jobs, fragment.jobs[1:]))
            )
        if cumulative_charge != fragment.charge:
            raise ValueError(f"Fragment {fragment} has incorrect charge {cumulative_charge} != {fragment.charge}")
        if cumulative_charge > self.config.MAX_CHARGE:
            raise ValueError(f"Fragment {fragment} has charge exceeding maximum")
        time = fragment.start_time 
        for j, job in enumerate(fragment.jobs):
            if j == 0:
                charge -= self.depot_to_job_charge_matrix[fragment.start_depot_id][job.id]
                time += self.depot_to_job_time_matrix[fragment.start_depot_id][job.id]
            else:
                charge -= self.job_charge_matrix[fragment.jobs[j-1].id][job.id]
                time += self.job_time_matrix[fragment.jobs[j - 1].id][job.id]

            if j == len(fragment.jobs) - 1:
                charge -= self.job_to_depot_charge_matrix[job.id][fragment.end_depot_id]
                
            charge -= job.charge
            if charge < 0:
                # print(f"Fragment {fragment} has negative charge")
                return None, None, False
            if time > job.start_time:
                raise ValueError(f"Fragment {fragment} starts too late")
            time = job.end_time
            
        time += self.job_to_depot_time_matrix[job.id][fragment.end_depot_id] + self.config.CHARGE_BUFFER
        assert time  == fragment.end_time, "end time does not align"
        return charge, time, is_valid
      
    def validate_solution(self, routes: list[list[ChargeDepot | TimedFragment]], objective: int,  triangle_inequality: bool=True):
        """
        Validates the input solution routes to ensure they are feasible.
        For a solution to be feasible:
        - All routes must be feasible
        - All jobs must be served
        - Must have as many routes as objective value
        """
        if len(routes) != objective:
            print(f"Objective value {objective} does not match number of routes {len(routes)}")
            return False
        for route in routes:
            if not self.validate_route(route):
                return False
            
        if triangle_inequality:
            self.validate_job_sequences(routes)
        return True
    
    def write_statistics(self, file: str=None):
        """
        Writes the recorded statistics from the model run to a csv format.
        Records the following:
            - Objective value
        """
        # if stats is None:
        #     stats = self.statistics.keys()
        if file is not None:
            df = pd.DataFrame([self.statistics])
            # open current csv, add an entry and exit.
            print(df)
            if os.path.exists(file):
                # print(pd.read_csv(file))
                df = pd.concat([pd.read_csv(file, index_col=0), df], ignore_index=True)
                # print(df)
            else:
                file = open(file, 'w')
            # turn dictionary into df
            df.to_csv(file)
    
    def write_fragments(self) -> None:
        """Utility method to save the fragments in a json format."""

        if not os.path.exists(f"{self.base_dir}/fragments"):
            # If the directory doesn't exist, create it
            os.makedirs(f"{self.base_dir}/fragments")

        
        with open(f"{self.base_dir}/fragments/f-{self.data['label']}.json", "w") as f:
            json.dump(
                {
                    "label": self.data["label"],
                    "fragments": [
                        {
                            "id": fragment.id,
                            "jobs": [job.id for job in fragment.jobs],
                            "start_time": fragment.start_time,
                            "end_time": fragment.end_time,
                            "start_depot_id": fragment.start_depot_id,
                            "end_depot_id": fragment.end_depot_id,
                            "charge": fragment.charge
                        }
                        for fragment in self.fragment_set
                    ],
                },
                f,
            )

    def read_fragments(self, file):
        """Reads the fragments from a json file."""
        data = json.load(open(file))
        fragment_set = set()
        for fragment in data["fragments"]:
            fragment_set.add(
                Fragment(
                    id=fragment["id"],
                    jobs=tuple(self.jobs_by_id[job_id] for job_id in fragment["jobs"]),
                    start_time=fragment["start_time"],
                    end_time=fragment["end_time"],
                    start_depot_id=fragment["start_depot_id"],
                    end_depot_id=fragment["end_depot_id"],
                    charge=None,
                )
            )
        return fragment_set


    @abstractmethod
    def generate_timed_network(self) -> None:
        """Creates the compressed time network for the current instance."""
    
    @abstractmethod
    def add_flow_balance_constraints(self):
        """Adds a flow balance constraint at every node in the network."""

    @abstractmethod
    def get_validated_timed_solution(self, solution: list[set[int]], expected_vehicles: int=None) -> list[list[ChargeDepot | Fragment]]:
        """
        Validates the prior solution to ensure its feasbility.
        Converts a list of fragments which form a route into its consequent Timed Depot/Fragment, 
        then validates on that.
        """

    @abstractmethod
    def validate_job_sequences(self, routes: list[list[ChargeDepot | Fragment]]) -> bool:
        """
        Validates no job sequenecs A - B - C occur such that A - C is illegal.
        """

    @abstractmethod
    def validate_route(self, route: list[ChargeDepot | Fragment]) -> bool:
        """
        Validates a route to ensure it is feasible.
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time        
        """

    @abstractmethod
    def run(self):
        """Method to run a regular solve (no special solution)"""