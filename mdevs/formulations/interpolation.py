import json
import os
import time
from collections import defaultdict
import heapq
from gurobipy import Model, GRB, quicksum, Var
import pandas as pd
from typing import TypeVar
import glob
from abc import ABC, abstractmethod
from itertools import product, islice
from mdevs.utils.visualiser import visualise_timed_network, visualise_routes
import math
from mdevs.formulations.base import *
from mdevs.formulations.charge_functions import LinearChargeFunction

@dataclass
class JobChargeStore:
    """
    Stores information around the min/max charge required to execute one job after an another.
    Similarly, the end charge of those states also.
    # TODO: DEAL WITH NEGATIVE CHARGES / infeasible pairs of fragments
    """
    start: Location
    end: Location
    min_start_charge: int # Minimum charge starting from the END of start 
    max_start_charge: int # Maximum charge starting from the END of start
    min_end_charge: int # Minimum charge to reach the END of end
    max_end_charge: int # Maximum charge to reach the END of end

class InterpolationIP(BaseMDEVCalculator):
    TYPE = "interpolation"

    def __init__(
            self,
            file: str,
            charge_calculator_class=LinearChargeFunction,
            charge_calculator_kwargs={},
            config=CalculationConfig(), 
            **kwargs
        ):
        super().__init__(
            file, 
            charge_calculator_class, 
            config=config, 
            charge_calculator_kwargs=charge_calculator_kwargs, 
            **kwargs
        )
        self.location_charge_bounds: dict[tuple[Job, Job], JobChargeStore] = {}
        self.locations: list[Location] = self.depots + self.jobs
        self.location_by_id = {location.id if isinstance(location, Building) else location.offset_id: location for location in self.locations}
        self.pairs_by_start: dict[Location, set[tuple[Location, Location]]] = defaultdict(set)
        self.pairs_by_end: dict[Location, set[tuple[Location, Location]]] = defaultdict(set)
        self.generate_valid_location_lookups()

    def generate_cost_matrices(self) -> None:
        """Generates all cost related matrices for the problem"""
        self.generate_building_distance_matrix()

        self.distance_matrix = [[0 for _ in range(len(self.locations))] for _ in range(len(self.locations))]
        self.charge_matrix = [[0 for _ in range(len(self.locations))] for _ in range(len(self.locations))]
        self.time_matrix = [[0 for _ in range(len(self.locations))] for _ in range(len(self.locations))]
        for i, loc_1 in enumerate(self.locations):
            for j, loc_2 in enumerate(self.locations):
                if i == j:
                    continue
                match loc_1, loc_2:
                    #depot to depot
                    case Building(), Building():
                        distance = (
                            self.building_distance[loc_1.id][loc_2.id] 
                            + self.get_internal_distance(loc_1.location) 
                            + self.get_internal_distance(loc_2.location)
                        )
                    #job to job
                    case Job(), Job():
                        distance = self.get_job_distance(loc_1, loc_2)
                    #depot to job and vice versa
                    case Building(), Job():
                        distance = self.get_job_to_depot_distance(loc_2, loc_1, is_job_origin=False)
                    case Job(), Building():
                        distance = self.get_job_to_depot_distance(loc_1, loc_2, is_job_origin=True)
                self.distance_matrix[i][j] = distance
                self.charge_matrix[i][j] = self.distance_to_charge(distance)
                self.time_matrix[i][j] = self.distance_to_time(distance)

    def _generate_valid_location_pairs(self):
        """
        Generates all valid location pairs for the problem.

        Depots:
        - No depots go to another depot

        Jobs:
        - Cannot map to another job unless it can reach it in time.
        """
        self.valid_location_pairs: set[tuple[Location, Location]] = set()
        for start, end in product(self.locations, self.locations):
            match start, end:
                case Building(), Building():
                    continue
                case Job(), Job():
                    if start.end_time + self.time_matrix[start.offset_id][end.offset_id] > end.start_time:
                        continue                    
                    if self.calculate_minimum_charge_to_execute_jobs(start, end) > self.config.MAX_CHARGE:
                        continue
                    self.valid_location_pairs.add((start, end))

                case (Job(), Building()) | (Building(), Job()):
                    self.valid_location_pairs.add((start, end))

        for start, end in self.valid_location_pairs:
            self.pairs_by_start[start].add((start, end))
            self.pairs_by_end[end].add((start, end))

    def generate_valid_location_lookups(self):
        self.generate_cost_matrices()
        self._generate_valid_location_pairs()
        self.calculate_charge_bounds_for_location_pairs()

    def calculate_minimum_charge_to_execute_jobs(self, start: Location, end: Location):
        """
        Calculates the minimum charge required to execute end_job from start job.
        This takes into account the recharge time that could lie between the two jobs.
        """
        match start, end:
            case Building(), Job():
                min_charge = (
                    self.charge_matrix[start.offset_id][end.offset_id]
                    + end.charge
                    + min(self.charge_matrix[end.offset_id][depot.offset_id] for depot in self.depots)
                )
            case Job(), Job():
                fixed_end_cost = (
                    end.charge 
                    + min(self.charge_matrix[end.offset_id][depot.offset_id] for depot in self.depots)
                )

                # Determine whether there is a possible detour.               
                min_depot = min(
                    (
                        depot for depot in self.depots
                        if self.charge_matrix[start.offset_id][depot.offset_id]
                    ),
                    key=lambda depot: (
                        self.time_matrix[start.offset_id][depot.offset_id]
                        + self.time_matrix[depot.offset_id][end.offset_id]
                    )
                ) 

                recharge_time = (
                    end.start_time 
                    - start.end_time
                    - self.time_matrix[start.offset_id][min_depot.offset_id]
                    - self.time_matrix[min_depot.offset_id][end.offset_id]
                    - self.config.CHARGE_BUFFER
                )
                # TODO: ASSUMES LINEAR CHARGING, will need to be based on current charge level as for non linear
                break_even_recharge_time = self.charge_inverse(
                    self.charge_matrix[start.offset_id][min_depot.offset_id]
                    + self.charge_matrix[min_depot.offset_id][end.offset_id]
                    - self.charge_matrix[start.offset_id][end.offset_id]
                ) 
                if recharge_time > break_even_recharge_time:
                    charge_cost_after_depot = fixed_end_cost + self.charge_matrix[min_depot.offset_id][end.offset_id]
                    possible_recharge = self.get_charge(0, recharge_time)
                    min_charge = (
                        max(charge_cost_after_depot - possible_recharge, 0)  
                        + self.charge_matrix[start.offset_id][min_depot.offset_id] # Charge cost to reach depot from start
                    )
                else:
                    min_charge = (
                        fixed_end_cost
                        + self.charge_matrix[start.offset_id][end.offset_id]
                    )

            case Job(), Building():
                min_charge = self.charge_matrix[start.offset_id][end.offset_id]
        return min_charge

    def calculate_maximum_charge_to_execute_jobs(self, start: Location, end: Location):
        """
        Calculates the maximum charge possible to execute end_job from start job.
        For Job to Job, this is given by:
            Max charge
            - charge to execute start if a job
            - min travel charge to reach start job
            + possible recharge time between jobs, capped at max charge.
        """

        match start, end:
            case Building(), Job():
                max_charge = self.config.MAX_CHARGE
            case Job(), Job():
                # Max charge completing the start job
                max_charge = (
                    self.config.MAX_CHARGE
                    - start.charge
                    - min(self.charge_matrix[depot.offset_id][start.offset_id] for depot in self.depots)
                )
            case Job(), Building():
                max_charge = (
                    self.config.MAX_CHARGE
                    - start.charge
                    - min(self.charge_matrix[depot.offset_id][start.offset_id] for depot in self.depots)
                )
        return max_charge

    def calculate_minimum_completion_charge_for_jobs(self, start: Location, end: Location):
        """
        Calculates the minimum charge that can be attained from completing end_job starting from start_job.
        """
        start_charge = self.calculate_minimum_charge_to_execute_jobs(start, end)

        match start, end:
            case Building(), Job():
                min_charge = start_charge - end.charge - self.charge_matrix[start.offset_id][end.offset_id]

            case Job(), Job(): 
                min_charge = self._calculate_end_charge_from_charge_level(start, end, start_charge)                     

            case Job(), Building():
                min_charge = start_charge - self.charge_matrix[start.offset_id][end.offset_id]

        return min_charge

    def calculate_maximum_completion_charge_for_jobs(self, start: Location, end: Location):
        """
        Calculates the maximum charge that can be attained from completing end_job starting from start_job.
        """
        start_charge = self.calculate_maximum_charge_to_execute_jobs(start, end)
        match start, end:
            case Building(), Job():
                max_charge = start_charge - end.charge - self.charge_matrix[start.offset_id][end.offset_id]

            case Job(), Job():
                max_charge = self._calculate_end_charge_from_charge_level(start, end, start_charge)

            case Job(), Building():
                max_charge = start_charge

        return max_charge

    def _calculate_end_charge_from_charge_level(self, start: Job, end: Job, start_charge: int):
        """
        Calculates the best case end charge level from the start job leaving at start_charge.
        """
        min_depot = self.get_closest_detour_depot(start, end, start_charge=start_charge) 
        recharge_time = (
            end.start_time 
            - start.end_time
            - self.time_matrix[start.offset_id][min_depot.offset_id]
            - self.time_matrix[min_depot.offset_id][end.offset_id]
            - self.config.CHARGE_BUFFER
        )
 
        charge_at_end_of_detour = (
            min(
                (
                    start_charge
                    - self.charge_matrix[start.offset_id][min_depot.offset_id]
                    + self.get_charge(0, recharge_time)
                ),
                self.config.MAX_CHARGE
            )
            - self.charge_matrix[min_depot.offset_id][end.offset_id]
        )

        charge_without_detour = (
            start_charge
            - self.charge_matrix[start.offset_id][end.offset_id]
        )

        end_charge = max(charge_at_end_of_detour, charge_without_detour)
        end_charge -= end.charge
        return end_charge

    def get_closest_detour_depot(self, start: Job, end: Job, start_charge: int | None=None) -> Building:
        if start_charge is None:
            start_charge = self.config.MAX_CHARGE
        min_depot = min(
            (
                depot for depot in self.depots
                if self.charge_matrix[start.offset_id][depot.offset_id] <= start_charge
            ),
            key=lambda depot: (
                self.time_matrix[start.offset_id][depot.offset_id]
                + self.time_matrix[depot.offset_id][end.offset_id]
            )
        )
        
        return min_depot

    def calculate_charge_bounds_for_location_pairs(self):
        """
        Calculates the charge bounds for all pairs of jobs which are feasibly linked.
        Feasible linking requires:
            - job start times bering feasibly reached.

        """
        for start, end in self.valid_location_pairs:
            self.location_charge_bounds[(start, end)] = JobChargeStore(
                start=start,
                end=end,
                min_start_charge=self.calculate_minimum_charge_to_execute_jobs(start, end),
                max_start_charge=self.calculate_maximum_charge_to_execute_jobs(start, end),
                min_end_charge=self.calculate_minimum_completion_charge_for_jobs(start, end),
                max_end_charge=self.calculate_maximum_completion_charge_for_jobs(start, end),
            )

    def build_model(self):
        self.model = Model("Interpolation")
        self.job_arc_variables = {
            (start, end): self.model.addVar(
                vtype=GRB.BINARY,
                name=f"arc_{start.offset_id}_{end.offset_id}"
            )
            for start, end in self.valid_location_pairs
        }

        self.arc_charge_lower = {
            (start, end): self.model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"arc_charge_lower_{start.offset_id}_{end.offset_id}"
            )
            for start, end in self.valid_location_pairs
        }

        self.arc_charge_upper = {
            (start, end): self.model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"arc_charge_upper_{start.offset_id}_{end.offset_id}"
            )
            for start, end in self.valid_location_pairs
        }

        self.charge_bound_linking = {
            pair: self.model.addConstr(
                self.arc_charge_lower[pair] + self.arc_charge_upper[pair] == self.job_arc_variables[pair],
                name=f"charge_linking_{pair[0].offset_id}_{pair[1].offset_id}"
            )
            for pair in self.valid_location_pairs
        }

        self.coverage = {
            job: self.model.addConstr(
                quicksum(self.job_arc_variables[pair] for pair in self.pairs_by_start[job]) == 1,
                name=f"coverage_{job.offset_id}"
            )
            for job in self.jobs
        }
    
        self.flow_balance = self.model.addConstr(
            (
                quicksum(self.job_arc_variables[pair] for depot in self.depots for pair in self.pairs_by_start[depot])
                == quicksum(self.job_arc_variables[pair] for depot in self.depots for pair in self.pairs_by_end[depot])
            ),
            name="flow_balance"
        )
        
        self.job_flow_balance = {
            job: self.model.addConstr(
                quicksum(self.job_arc_variables[pair] for pair in self.pairs_by_start[job])
                == quicksum(self.job_arc_variables[pair] for pair in self.pairs_by_end[job])
            )
            for job in self.jobs
        }

        self.charge_balance = {
            job: self.model.addConstr(
                quicksum(
                    self.location_charge_bounds[pair].min_end_charge * self.arc_charge_lower[pair]
                    + self.location_charge_bounds[pair].max_end_charge * self.arc_charge_upper[pair]
                    for pair in self.pairs_by_end[job]
                )
                >=
                quicksum(
                    self.location_charge_bounds[pair].min_start_charge * self.arc_charge_lower[pair]
                    + self.location_charge_bounds[pair].max_start_charge * self.arc_charge_upper[pair]
                    for pair in self.pairs_by_start[job]
                )
            )            
            for job in self.jobs
        }

        self.model.setObjective(
            quicksum(
                self.job_arc_variables[pair] for depot in self.depots for pair in self.pairs_by_start[depot]
            ),
            GRB.MINIMIZE
        )

    def solve(self):
        self.model.setParam("TimeLimit", 300)
        for job in self.jobs:
            for pair in self.pairs_by_start[job]:
                self.job_arc_variables[pair].BranchPriority = math.ceil(max(self.jobs, key=lambda j: j.start_time).start_time / job.start_time * 1000)
        
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            print("Optimal solution found")
        elif self.model.Status == GRB.INFEASIBLE:
            print("Infeasible solution found")
            self.model.computeIIS()
            self.model.write("interpolation.ilp")
        

    def sequence_routes(self) -> list[list[Location]]:
        solution_arcs = [a for a, x in self.job_arc_variables.items() if x.x > 0.9]
        incident_jobs = defaultdict(list)
        for s, e in solution_arcs:
            incident_jobs[s].append(e)
        # Doesn't include dead hanging
        job_sequences = []
        for start_depot in self.depots:
            while len(incident_jobs[start_depot]) != 0:
                curr = start_depot
                current_route = [curr]
                while len(incident_jobs[curr]) != 0:
                    curr = incident_jobs[curr].pop()
                    current_route.append(curr)
                    if isinstance(curr, Building):
                        job_sequences.append(current_route)
                        break
        
        complete_routes = []
        
        return job_sequences 

    def validate_routes(self, job_sequences: list[list[Location]]) -> bool:
        for seq in job_sequences:
            # Include start depot
            # curr_route = seq[:1]
            charge = prev_charge = (
                self.config.MAX_CHARGE 
                - self.charge_matrix[seq[0].offset_id][seq[1].offset_id]
                - seq[1].charge    
            )
            assert isinstance(seq[0], Building) and isinstance(seq[-1], Building)
            for start, end in zip(seq[1:], seq[2:-1]): # Exclude last as it is a Building
                # Check if can do a recharge and reach it in time
                charge = self._calculate_end_charge_from_charge_level(start, end, charge)
                print(
                    charge,
                    self.arc_charge_lower[(start, end)].x * self.location_charge_bounds[start, end].min_end_charge + self.arc_charge_upper[(start, end)].x * self.location_charge_bounds[start,end].max_end_charge
                )
                
                prev_charge = charge
                assert charge >= 0
                

    def run(self):
        self.build_model()
        self.solve()
        routes = self.sequence_routes()
        self.validate_routes(routes)
        return routes