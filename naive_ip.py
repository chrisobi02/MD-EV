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
from visualiser import visualise_timed_network, visualise_routes
import math
from fragment_generation import BaseMDEVCalculator
from constants import *

T = TypeVar("T")



class NaiveIP(BaseMDEVCalculator):
    """
    Implements the naive MIP in section 4.

    This is used for validation of fragment solutions
    For parity with the paper, the job indices are offset by the number of depots.

    TODO:
    - Implement cost matrices with the indices in the paper
    i.e. depots, jobs in one matrix
    give them new id's.

    - Calculate max/min charge for each state required
    
    - add validation / solution transfers between the two styles.
    """
    def __init__(self, file: str):
        super().__init__(file)
        self.locations = self.depots + self.jobs
        self.location_by_id = {location.id if isinstance(location, Building) else location.offset_id: location for location in self.locations}
        self.charge_levels = range(CHARGE_MAX + 1)
    
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
                        distance = self.building_distance[loc_1.id][loc_2.id]
                    #job to job
                    case Job(), Job():
                        distance = self.get_job_distance(loc_1, loc_2)
                    #depot to job and vice versa
                    case Building(), Job() | Job(), Building():
                        distance = self.get_job_to_depot_distance(loc_1, loc_2, is_job_origin=(isinstance(loc_1, Job)))
                self.distance_matrix[i][j] = distance
                self.charge_matrix[i][j] = self.distance_to_charge(distance)
                self.time_matrix[i][j] = self.distance_to_time(distance)

    def generate_valid_charge_levels(self) -> None:
        """
        Calculates the valid charge levels for each location pair.
        starting from a location at a charge level, 
        we compute:
        - state of charge from each depot, if unreachable is negatve
        - state of charge skipping a depot.
        We then store the maximum for each charge level.
        """
        self.max_charge_by_loc_pair_charge: dict[tuple[int, int, int], int] = {}
        self.min_charge_by_loc_pair: dict[tuple[int, int], int] = {}
        for start, end in product(self.locations, self.locations):
            # compute the largest charge achievable between these two locations
            if isinstance(end, Building):
                recharge_cost = self.charge_matrix[start.offset_id][end.offset_id]
                for charge in self.charge_levels:
                    self.max_charge_by_loc_pair_charge[start, end, charge] = CHARGE_MAX if charge >= recharge_cost else -1
                continue

            if isinstance(start, Building):
                for charge in self.charge_levels:
                    self.max_charge_by_loc_pair_charge[start, end, charge] = CHARGE_MAX - self.charge_matrix[start.offset_id][end.offset_id] - end.charge
                continue
        
            time_feasible_detours = [
                detour for detour in self.depots 
                if (
                    start.end_time + self.time_matrix[start.offset_id][detour.offset_id] 
                    + RECHARGE_TIME + self.time_matrix[detour.offset_id][end.offset_id] 
                    <= end.start_time
                )
            ]
            for charge in self.charge_levels:
                # All detours which can be reached before the next job with non-zero charge
                detour_cost = min(
                    (
                        self.charge_matrix[detour.offset_id][end.offset_id] + end.charge for detour in time_feasible_detours 
                        if charge - self.charge_matrix[start.offset_id][detour.offset_id] >= 0
                    ),
                    default=math.inf
                )
                straight_path = (
                    charge - self.charge_matrix[start.offset_id][end.offset_id] - end.charge
                    if self.time_matrix[start.offset_id][end.offset_id] + start.end_time <= end.start_time
                    else - math.inf
                    )
                max_charge = max(straight_path, CHARGE_MAX - detour_cost, -1)
                # charge left at end of job end
                self.max_charge_by_loc_pair_charge[start, end, charge] = max_charge


    def generate_model(self) -> Model:
        """Generates the MIP model."""
        self.model = Model("IP")
        print("Creating variables...")
        self.job_arcs: dict[tuple, Var] = {
            (start, end): self.model.addVar(vtype=GRB.BINARY, name=f"s_{start.offset_id}_e_{end.offset_id}") 
            for start, end in product(self.locations, repeat=2) 
            if not (isinstance(start, Building) and isinstance(end, Building))
        }

        reachable_charge_levels_by_location = {
            loc:
            {
                self.max_charge_by_loc_pair_charge[start, loc, c]
                for start in self.locations for c in self.charge_levels
                if self.max_charge_by_loc_pair_charge[start, loc, c] >= 0
            }
            for loc in self.locations
        }

        self.job_charges: dict[tuple, Var] = {
            (loc, charge): self.model.addVar(vtype=GRB.BINARY, name=f"j_{loc.offset_id}_c_{charge}")
            for loc in self.locations
            for charge in reachable_charge_levels_by_location[loc]
        }

        print("Creating flow balance...")
        forward_balance = {
            start: self.model.addConstr(
                quicksum( 
                    self.job_arcs[start, end] 
                    for end in self.locations 
                    if end.offset_id > start.offset_id or isinstance(end, Building)
                ) == 1,
                f"fb_{start.offset_id}"
            )
            for start in self.jobs
        } 

        backward_balance = {
            end: self.model.addConstr(
                quicksum(self.job_arcs[start, end] for start in self.locations if start.offset_id < end.offset_id) == 1,
                f"bb_{end.offset_id}"
            )
            for end in self.jobs
        }
        print("Creating coverage...")
        charge_coverage = {
            job: self.model.addConstr(
                quicksum(self.job_charges[job, charge] for charge in reachable_charge_levels_by_location[job]) == 1,
                f"cv_{job.offset_id}"
            )
            for job in self.jobs
        }
        
        print("Incompatible charge...")
        incompatible_charge = {
            (start, end, charge):
            self.model.addConstr(
               1 >= self.job_arcs[start, end] + self.job_charges[start, charge],
                f"ic_{start.offset_id}_{end.offset_id}_{charge}"
            )
            for start in self.jobs for end in self.locations 
            if start.offset_id < end.offset_id or isinstance(end, Building)
            for charge in reachable_charge_levels_by_location[start]
            if self.max_charge_by_loc_pair_charge[start, end, charge] < 0
        }

        print("Propagating charge...")
        self.propagate_charge = {
            (start, end, charge): self.model.addConstr(
                self.job_charges[end, self.max_charge_by_loc_pair_charge[start, end, charge]]
                >= self.job_arcs[start, end] + self.job_charges[start, charge] - 1,
                f"pc_s_{start.offset_id}_e_{end.offset_id}_c_{charge}"
            )
            for start in self.jobs for end in self.locations 
            if start.offset_id < end.offset_id or isinstance(end, Building)
            for charge in reachable_charge_levels_by_location[start]
            if self.max_charge_by_loc_pair_charge[start, end, charge] >= 0
        }

        print("Enforcing charge...")
        self.enforce_charge = {
            (start, end): self.model.addConstr(
                self.job_charges[end, self.max_charge_by_loc_pair_charge[start, end, CHARGE_MAX]] >= self.job_arcs[start, end],
                f"ec_s_{start.offset_id}_e_{end.offset_id}"
            )
            for start, end in product(self.depots, self.jobs) if self.max_charge_by_loc_pair_charge[start, end, CHARGE_MAX] >= 0
        }

        self.model.setObjective(quicksum(self.job_arcs[start, end] for start, end in product(self.depots, self.jobs)))

    def solve(self) -> None:
        print("Optimizing...")
        self.model.optimize()

        if self.model.status == GRB.INFEASIBLE:
            print("Infeasible")
            self.model.computeIIS()
            self.model.write("IP.ilp")

    def run(self):
        self.model = Model("naive")
        self.generate_cost_matrices()
        self.generate_valid_charge_levels()
        self.generate_model()
        self.solve()