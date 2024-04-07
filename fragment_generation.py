import json
import os
import time
from collections import defaultdict
import heapq
from gurobipy import Model, GRB, quicksum
import pandas as pd
from typing import TypeVar
import glob
from abc import ABC, abstractmethod
import re
from visualiser import visualise_timed_network, visualise_routes

from constants import *

T = TypeVar("T")


class BaseFragmentGenerator(ABC):
    def __init__(self, file: str) -> None:
        self.base_dir = os.path.dirname(file)
        self.data = json.load(open(file))
        self.buildings_by_id: dict[int, Building] = self.to_dataclass_by_id(
            self.data["buildings"], Building
        )
        self.buildings: list[Building] = list(self.buildings_by_id.values())
        self.depots_by_id: dict[int, Building] = {
            building.id: building
            for building in self.buildings
            if building.type == "depot"
        }
        self.depots: list[Building] = list(self.depots_by_id.values())
        self.jobs_by_id: dict[int, Job] = self.to_dataclass_by_id(
            self.data["jobs"], Job
        )
        self.jobs: list[Job] = list(self.jobs_by_id.values())
        self.generate_all_cost_matrices()
        self.fragment_set: set[Fragment] = set()
        self.fragments_by_id: dict[int, Fragment] = {}
        self.fragment_vars_by_id: dict[int, Fragment] = {}
        self.contracted_fragments: set[ContractedFragment] = set()
        self.timed_depots_by_depot: dict[int, list[TimedDepot]] = defaultdict(list)
        self.timed_fragments_by_timed_depot: dict[TimedDepot, list[TimedFragment]] = defaultdict(list)
        self.waiting_arcs: dict[TimedDepot, dict[TimedDepot, GRB.Var]] = {}
        self.model = Model("fragment_network")
        self.statistics: dict[str, int | float] = {}

    def generate_all_cost_matrices(self):
        self.generate_building_distance_matrix()
        self.generate_job_cost_matrix()
        self.generate_job_to_depot_matrices()
        self.generate_depot_to_job_matrices()

    def to_dataclass_by_id(
        self, json_data: list[dict], dataclass: T, id_field: str = "id"
    ) -> dict[int, T]:
        return {
            data[id_field] if id_field else i: dataclass(
                **{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
            )
            for i, data in enumerate(json_data)
        }

    def distance(self, start: list, end: list) -> int:
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def generate_building_distance_matrix(self) -> list[list[int]]:
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

    def generate_job_to_depot_matrices(self) -> list[list[int]]:
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
                if job.building_end_id == depot.id:
                    distance = self.distance(job.end_location, depot.location)
                else:
                    distance = (
                        self.building_distance[job.building_end_id][depot.id]
                        + self.get_internal_job_distance(job, start=False)
                        + self.get_internal_distance(depot.location)
                    )
                self.job_to_depot_distance_matrix[i][j] = distance
                self.job_to_depot_charge_matrix[i][j] = self.distance_to_charge(
                    distance
                )
                self.job_to_depot_time_matrix[i][j] = self.distance_to_time(distance)

        return (
            self.job_to_depot_distance_matrix,
            self.job_to_depot_charge_matrix,
            self.job_to_depot_time_matrix,
        )

    def generate_depot_to_job_matrices(self) -> list[list[int]]:
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
                if job.building_start_id == depot.id:
                    distance = self.distance(job.start_location, depot.location)
                else:
                    distance = (
                        self.building_distance[job.building_start_id][depot.id]
                        + self.get_internal_job_distance(job, start=True)
                        + self.get_internal_distance(depot.location)
                    )
                self.depot_to_job_distance_matrix[i][j] = distance
                self.depot_to_job_charge_matrix[i][j] = self.distance_to_charge(
                    distance
                )
                self.depot_to_job_time_matrix[i][j] = self.distance_to_time(distance)

        return (
            self.depot_to_job_distance_matrix,
            self.depot_to_job_charge_matrix,
            self.depot_to_job_time_matrix,
        )

    def get_internal_distance(self, location: tuple) -> int:
        return sum(location)

    def get_internal_job_distance(self, job: Job, start=True):
        """Returns the distance of the job to the"""
        return self.get_internal_distance(
            job.start_location if start else job.end_location
        )
       
    def generate_job_cost_matrix(self) -> list[list[int]]:
        """Generates the charge and time cost"""
        self.job_distance_matrix = [
            [0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))
        ]
        self.job_charge_matrix = [
            [0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))
        ]
        self.job_time_matrix = [
            [0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))
        ]
        for i, job in enumerate(self.jobs):
            for j, other_job in enumerate(self.jobs):
                # NAIVE: if slow, filter to jobs that can feasibly be connected
                if job.id == other_job.id:
                    continue
                if job.building_end_id == other_job.building_start_id:
                    distance = self.distance(job.end_location, other_job.start_location)
                else:
                    # Move between buildings
                    building_distance = self.building_distance[job.building_end_id][
                        other_job.building_start_id
                    ]
                    # Move to entrance of buildings
                    travel_distance = self.get_internal_job_distance(
                        job, start=False
                    ) + self.get_internal_job_distance(other_job, start=True)
                    distance = building_distance + travel_distance

                self.job_distance_matrix[i][j] = distance
                self.job_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.job_time_matrix[i][j] = self.distance_to_time(distance)

        return self.job_distance_matrix

    def distance_to_time(self, distance: int) -> int:
        return round(distance / VEHICLE_MOVE_SPEED_PER_UNIT)

    def distance_to_charge(self, distance: int) -> int:
        return round(distance * CHARGE_PER_METRE / CHARGE_PER_UNIT)

    def generate_contracted_fragments(self):
        """Contracts the fragments by finding all fragments with the same jobs covered and combining the ids."""
        if len(self.fragment_set) == 0:
            self.generate_fragments()
        depots_by_job_list = defaultdict(lambda: defaultdict(set))
        for fragment in self.fragment_set:
            depots_by_job_list[tuple(fragment.jobs)]["start_depot_ids"].add(
                fragment.start_depot_id
            )
            depots_by_job_list[tuple(fragment.jobs)]["end_depot_ids"].add(
                fragment.end_depot_id
            )

        for jobs, depot_info in depots_by_job_list.items():
            self.contracted_fragments.add(
                ContractedFragment(
                    jobs=jobs,
                    start_depot_ids=tuple(depot_info["start_depot_ids"]),
                    end_depot_ids=tuple(depot_info["end_depot_ids"]),
                )
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
                    "node_count": self.model.NodeCount,
                    "gap": self.model.MIPGap,
                }
            )
            print(self.model.objval)

    def get_fragments_from(
            self, 
            start: TimedDepot,
            current_route: list, 
            fragments_by_timed_depot: dict[TimedDepot, set[tuple[TimedDepot, int]]], 
            flow_by_waiting_arc_start: dict[TimedDepot, dict[TimedDepot, int]]
        ) -> list[Fragment | TimedDepot]:
        """Traverse the waiting arcs and fragments to get the route from a given start depot."""

        if len(fragments_by_timed_depot[start]) != 0:
            next_depot, f_id = fragments_by_timed_depot[start].pop()
            current_route.extend([self.fragments_by_id[f_id], next_depot])

        elif len(flow_by_waiting_arc_start[start]) != 0:
            next_depot = min(flow_by_waiting_arc_start[start])
            flow_by_waiting_arc_start[start][next_depot] -= 1
            if flow_by_waiting_arc_start[start][next_depot] == 0:
                flow_by_waiting_arc_start[start].pop(next_depot)
        else:
            print(self.stringify_route(current_route))
            # print("return", current_route)
            return current_route
        self.get_fragments_from(next_depot, current_route, fragments_by_timed_depot, flow_by_waiting_arc_start)
        return current_route

    def stringify_route(self, route: list) -> list[str]:
        """Converts a route into a string"""
        return " -> ".join([loc.route_str for loc in route])

    def create_routes(self):#, fragments: set[Fragment], waiting_arcs: set[tuple[TimedDepot, TimedDepot, int]] = None, expected_route_count: int=None) -> list[Route]:
        """
        Sequences a set of fragments into complete routes for the job horizon.
        If it is unable to sequence them, it says something
        """
        # Sequence fragments by their start / end depots
        solution_fragment_ids = {f for f in self.fragment_vars_by_id if self.fragment_vars_by_id[f].x > 0.5}
        waiting_arcs = [(start, end, round(self.waiting_arcs[start, end].x)) for start, end in self.waiting_arcs if self.waiting_arcs[start, end].x > 0.5]

        routes = []
        fragments_by_timed_depot: dict[TimedDepot, set[tuple[TimedDepot, int]]] = defaultdict(set[tuple])
        for f_id in solution_fragment_ids:
            # Retrieve the timeddepot it starts and ends at
            for timed_depot in self.timed_fragments_by_timed_depot:
                if f_id in [tf.id for tf in self.timed_fragments_by_timed_depot[timed_depot] if tf.direction == Flow.DEPARTURE]:
                    fragment = self.fragments_by_id[f_id] 
                    destination_depot = None
                    for td in self.timed_depots_by_depot[fragment.end_depot_id]:
                        for tf in self.timed_fragments_by_timed_depot[td]:
                            if tf.id == fragment.id:
                                destination_depot = td
                                break
                    # destination_depot = min(td for td in self.timed_depots_by_depot[fragment.end_depot_id] if any() >= fragment.end_time)
                    fragments_by_timed_depot[timed_depot].add((destination_depot, f_id))
                    break


        # Set up the waiting arc counts
        flow_by_waiting_arc_start = defaultdict(lambda : defaultdict(int))

        for start, end, count in waiting_arcs:
            flow_by_waiting_arc_start[start][end] = count

        # get the starting points for all routes: the first timed depot of each type (+ waiting arcs?)
        # for each depot's start point, start tracing a path until you hit a timed depot with no further connections.
        # if you hit a depot with no further connections, start a new route.
        for depot_id, timed_depots in self.timed_depots_by_depot.items():
            start_depot = min(timed_depots)
            # Following fragments
            while len(fragments_by_timed_depot[start_depot]) != 0 or len(flow_by_waiting_arc_start[start_depot]) != 0:
                route = self.get_fragments_from(start_depot, [start_depot], fragments_by_timed_depot, flow_by_waiting_arc_start)
                routes.append(route)
        return routes

    def read_solution(self, instance_type:str=None) -> set[int]:
        """Reads the solution given by the paper into a list of their routes which has waiting arcs and fragments."""
        data = pd.read_excel(r"data/mdevs_solutions.xlsx", sheet_name=None)
        for sheet in data:
            if instance_type in sheet:
                data = data[sheet]
                break

        curr_sol_str = data.query(
            f"ID_instance == {self.data['ID']} and battery_charging == 'constant-time'"
        )["solution"].values[0]
        solution_routes = []
        string_solution_routes=[]
        # Convert the matches to integers and find the maximum since sometimes there is a weird offset
        # job_offset = min(map(int, re.findall(r'>(\d+)>', curr_sol_str))) - 1
        num_depots = len(self.depots_by_id)
        num_jobs = len(self.jobs_by_id)
        print(len(curr_sol_str.split(",")))
        print(curr_sol_str)
        for route in curr_sol_str.split(","):
            print(route)
            current_route = []
            # Remove []
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

            string_solution_routes.append(self.stringify_route(current_route))
            print(string_solution_routes[-1])
            route_fragment_ids = self.convert_route_to_fragments(current_route)
            solution_routes.append(route_fragment_ids)

        print(string_solution_routes)
        return solution_routes, string_solution_routes

    def convert_route_to_fragments(self, route: list) -> list[int]:
        """Converts a route into its fragments"""
        fragment_ids = []
        prev_loc = None
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
                            print(
                                f"Fragment {current_fragment} not found, interpolating..."
                            )
                            interpolated_fragments = self.interpolate_fragments(
                                current_fragment
                            )
                            if interpolated_fragments:
                                fragment_ids.extend(interpolated_fragments)
                        current_fragment = {
                            "jobs": [],
                            "start_depot_id": location.id,
                            "end_depot_id": None,
                        }

                case Job():
                    current_fragment["jobs"].append(location)

            prev_loc = location
            
        return fragment_ids

    def validate_timed_network(self):
        """
        Validates the timed network to ensure it is feasible.
        For a timed network to be feasible:
        - No timed depot can have fragments which start after its time
        - Each timed depot cannot have a time earlier than the previous timed depot
        """
        for depot_id, timed_depots in self.timed_depots_by_depot.items():
            prev_td = None
            for td in timed_depots:
                if prev_td:
                    assert td.time > prev_td.time
                    if td.time <= prev_td.time:
                        print(f"Depot {td} has a time earlier than the previous depot.")
                        return False
                fragments = self.timed_fragments_by_timed_depot[td]
                # Check the following:
                # min end time >= max start time
                # no start time later than the current depot time and no earlier than the previous time.
                start_fragments = [f for f in fragments if f.direction == Flow.DEPARTURE]
                end_fragments = [f for f in fragments if f.direction == Flow.ARRIVAL]
                if len(end_fragments) != 0:
                    assert all(prev_td.time <= tf.time <= td.time for tf in start_fragments)
                if len(start_fragments) != 0:
                    assert all(prev_td.time <= tf.time <= td.time for tf in end_fragments)
                if len(start_fragments) != 0 and len(end_fragments) != 0:
                   max_arr = max(tf.time for tf in end_fragments) 
                   min_dep = min(tf.time for tf in start_fragments)
                   assert max_arr <= min_dep
                prev_td=td

    def validate_solution(self, routes: list[list[TimedDepot | TimedFragment]], objective: int):
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
        return True

    def get_fragment_id(self, candidate: dict):
        job_tuple = tuple(candidate["jobs"])
        for fragment in self.fragment_set:
            if (
                fragment.jobs == job_tuple
                and fragment.start_depot_id == candidate.get("start_depot_id")
                and fragment.end_depot_id == candidate.get("end_depot_id")
            ):
                return fragment.id
        else:
            print(
                (
                    f"no such fragment exists:\n"
                    f"jobs: {[j.id for j in candidate['jobs']]}\n"
                    f"start_depot_id: {candidate.get('start_depot_id')}\n"
                    f"end_depot_id: {candidate.get('end_depot_id')}"
                )
            )

    def visualise_solution(self, sol=None):
        """Visualises the solution found by gurobi in the network"""

        sol_fragments = {f for f in self.fragment_vars_by_id if self.fragment_vars_by_id[f].x > 0.5} 

        vehicle_arcs = []
        for f in sol_fragments:
            used_depots = []
            for depot in self.timed_depots_by_depot:
                for td in self.timed_depots_by_depot[depot]:
                    if f in [t.id for t in self.timed_fragments_by_timed_depot[td]]:
                        used_depots.append(td)
            vehicle_arcs.append(sorted(used_depots))
        waiting_arcs = [{"timed_depot": arc, "flow": round(self.waiting_arcs[arc].x)} for arc in self.waiting_arcs]# if self.waiting_arcs[arc].x > 0.5]
        timed_depots = [td for depot in self.timed_depots_by_depot for td in self.timed_depots_by_depot[depot]]

        fig = visualise_timed_network(
            timed_depots,
            vehicle_arcs,
            waiting_arcs,
            instance_label=self.data["label"],
            charge_type="constant-time",
        )
        for f_id in sol:
            #annotate f (Fragment)'s start location and end location, the start arrow should point towards the start location, similar for the end
            f = self.fragments_by_id[f_id]
            start_time = f.start_time
            end_time = f.end_time
            start_depot_id =f.start_depot_id
            end_depot_id = f.end_depot_id

            fig.add_annotation(
                x=start_time,
                y=start_depot_id,
                text="S",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=0,
                ay=-40,
            )
            fig.add_annotation(
                x=end_time,
                y=end_depot_id,
                text="E",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=0,
                ay=40,
            )

            fig.show()

    def visualise_routes(self, routes: list):
        """Visualises the routes found by the model"""
        fig = visualise_routes(
            routes,
            [td for depot in self.timed_depots_by_depot for td in self.timed_depots_by_depot[depot]],
            # instance_label=self.data["label"],
            # charge_type="constant-time",
        )
        fig.show()

    def write_statistics(self, folder: str=None, file: str=None):
        """
        Writes the recorded statistics from the model run to a csv format.
        Records the following:
            - Objective value
        """
        # if stats is None:
        #     stats = self.statistics.keys()
        if file:
            df = pd.DataFrame([self.statistics])
            # open current csv, add an entry and exit.
            print(df)
            if os.path.exists(file):
                print(pd.read_csv(file))
                df = pd.concat([pd.read_csv(file, index_col=0), df], ignore_index=True)
                print(df)
            # turn dictionary into df
            df.to_csv(file)

    def display_stats(self):
        """Gives some info on the model"""

        # Reduction in size from time/space compression
        uncompressed_network_timed_depots = set()
        for f in self.fragment_set:
            uncompressed_network_timed_depots.update(
                {TimedDepot(id=f.start_depot_id, time=f.start_time), TimedDepot(id=f.end_depot_id, time=f.end_time)}
                )
        compressed_timed_depot_nodes = sum(len(self.timed_depots_by_depot[depot]) for depot in self.timed_depots_by_depot)
        print(
            f"Reduced from {len(uncompressed_network_timed_depots)} to {compressed_timed_depot_nodes} ({round(compressed_timed_depot_nodes/len(uncompressed_network_timed_depots) * 100)} % of uncompressed) Timed Depots"
        )

        # Number of fragments. contracted fragments
        print(
            f"Fragment count: {len(self.fragment_set)}\n Contracted: {len(self.contracted_fragments)}  ({round(len(self.contracted_fragments)/len(self.fragment_set) * 100)} % of uncompressed)"
            )

    @abstractmethod
    def get_jobs_reachable_from(self, charge: int, job: Job) -> list[Job]:
        """
        Takes either a job or a depot and returns a set of jobs which can be reached from the input location.
        """
      
    @abstractmethod
    def generate_fragments(self):
        """
        Enumerates all possible fragments which satisfy the following requirements:
        """
    
    @abstractmethod
    def _generate_fragment_starting_at(
        self,
        fragment_set: set[Fragment],
        current_fragment: dict,
        job: Job,
        charge: int,
        id_counter: list[int],
    ) -> set[Fragment]:
        """
        Generates a fragment starting at a given job and depot with a given charge at a given time.
        """
        current_jobs: list[Job] = current_fragment["jobs"]
        # Get all jobs which can be reached from the current job
        # Add this partial part of the journey as a fragment.
        for id, depot in self.depots_by_id.items():
            if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge:
                finish_time = (
                    job.end_time
                    + self.job_to_depot_time_matrix[job.id][depot.id]
                    + RECHARGE_TIME
                )
                fragment_set.add(
                    Fragment(
                        id=id_counter[0],
                        jobs=tuple(current_jobs),
                        start_time=current_fragment["start_time"],
                        end_time=finish_time,
                        start_depot_id=current_fragment["start_depot_id"],
                        end_depot_id=depot.id,
                        charge=current_fragment["charge"] + self.job_to_depot_charge_matrix[job.id][depot.id],
                    )
                )
                id_counter[0] += 1

        reachable_jobs = self.get_jobs_reachable_from(charge, job)
        next_fragment = current_fragment.copy()
        for next_job in reachable_jobs:
            # manage memory
            next_fragment["jobs"] = current_jobs.copy() + [next_job]
            next_fragment["charge"] = charge + self.job_charge_matrix[job.id][next_job.id]
            # Otherwise, generate a fragment starting at the next job
            self._generate_fragment_starting_at(
                fragment_set,
                next_fragment,
                next_job,
                charge - self.job_charge_matrix[job.id][next_job.id],
                id_counter,
            )

        return fragment_set

    @abstractmethod
    def write_fragments(self) -> None:
        """Utility method to save the fragments in a json format."""

    @abstractmethod
    def generate_timed_network(self) -> None:
        """Creates the compressed time network for the current instance."""
        
    @abstractmethod
    def build_model(self):
        """Solves the network flow mip"""
        
    @abstractmethod
    def set_solution(self, solution: list[set[int]], n_vehicles=None):
        """Sets the solution to the model to be the input values.0"""
    
    @abstractmethod
    def get_validated_timed_solution(self, solution: list[set[int]], expected_vehicles: int=None) -> list[list[TimedDepot | Fragment]]:
        """
        Validates the prior solution to ensure its feasbility.
        Converts a list of fragments which form a route into its consequent Timed Depot/Fragment, 
        then validates on that.
        """

    @abstractmethod
    def validate_route(self, route: list[TimedDepot | Fragment]) -> bool:
        """
        Validates a route to ensure it is feasible.
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time        
        """

    @abstractmethod
    def interpolate_fragments(self, candidate):
        """Interpolates a series of fragments from sequential jobs in a solution read externally."""

    @abstractmethod
    def run(self):
        """Method to run a regular solve (no special solution)"""
class ConstantFragmentGenerator(BaseFragmentGenerator):
    def __init__(self, file: str) -> None:
        super().__init__(file)
        self.type = "constant-charging"
        self.statistics.update(
            {
                "type": self.type,
                "label": self.data["label"],
            }
        )


    def get_jobs_reachable_from(self, charge: int, job: Job) -> list[Job]:
        """
        Takes either a job or a depot and returns a set of jobs which can be reached from the input location.
        Filters job which:
            1. Are in the past
            2. Can be executed with a battery swap (depot visit) in between
            3. Cannot be executed with the current charge and reach a depot after (i.e. goes flat).
        """
        reachable_jobs = []
        t = job.end_time
        for next_job in self.jobs:
            # 1
            arrival_time = t + self.job_time_matrix[job.id][next_job.id]
            if next_job.start_time < arrival_time:
                # Cannot reach job at start time
                continue

            recharge_time = min(
                self.job_to_depot_time_matrix[job.id][depot.id]
                + RECHARGE_TIME
                + self.depot_to_job_time_matrix[depot.id][next_job.id]
                for depot in self.depots
                if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge
            )
            # 2.
            if t + recharge_time <= next_job.start_time:
                continue

            charge_cost = self.job_charge_matrix[job.id][next_job.id] + min(
                self.job_to_depot_charge_matrix[next_job.id][depot.id]
                for depot in self.depots_by_id.values()
            )
            # 3.
            if charge < charge_cost:
                # Cannot reach job and recharge.
                continue

            reachable_jobs.append(next_job)
        return reachable_jobs

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

    def generate_fragments(self, file: str=None):
        """
        Enumerates all possible fragments which satisfy the following requirements:
        1. The fragment leaves from a depot and returns to a depot with non-zero charge
        2. All jobs included within the fragment can be executed in sequence
        3. Any job which takes longer to reach than the time to return to a depot, recharge and reach the job before its start is ignored
        """
        # Only starting points for any fragment: any building in depots
        # only generate starting at the latest possible leaving time from depot (deadhanging is allowed)
        # keep generating until out of charge etc.
        # To make ALL FRAGMENTS, we need:
        # each depot -> each job -> each depot + those others which could fit
        # Each time hitting a new fragment, can cap it off with a depot and add to fragments.
        if file is not None:
            try:
                self.fragment_set = self.read_fragments(file)
                self.fragments_by_id = {fragment.id: fragment for fragment in self.fragment_set}
                self.generate_contracted_fragments()
                return self.fragment_set
            except FileNotFoundError:
                print(f"File {file} not found. Generating fragments...")

        fragment_set: set[Fragment] = set()
        id_counter = [0]
        time0 = time.time()
        for depot_id, depot in self.depots_by_id.items():
            # starting job
            for job in self.jobs:
                charge = CHARGE_MAX - self.depot_to_job_charge_matrix[depot.id][job.id]
                start_time = (
                    job.start_time - self.depot_to_job_time_matrix[depot.id][job.id]
                )
                
                if start_time < 0:
                    continue
                current_fragment = {
                    "jobs": [job],
                    "start_depot_id": depot.id,
                    "end_depot_id": None,
                    "start_time": start_time,
                    "end_time": None,
                    "charge": self.depot_to_job_charge_matrix[depot.id][job.id],
                }
                self._generate_fragment_starting_at(
                    fragment_set, current_fragment, job, charge, id_counter
                )
        self.fragment_set = fragment_set
        self.fragments_by_id = {fragment.id: fragment for fragment in fragment_set}
        self.statistics["fragment_generation_time"] = time.time() - time0
        self.generate_contracted_fragments()
        return fragment_set

    def _generate_fragment_starting_at(
        self,
        fragment_set: set[Fragment],
        current_fragment: dict,
        job: Job,
        charge: int,
        id_counter: list[int],
    ) -> set[Fragment]:
        """
        Generates a fragment starting at a given job and depot with a given charge at a given time.
        """
        current_jobs: list[Job] = current_fragment["jobs"]
        # Get all jobs which can be reached from the current job
        # Add this partial part of the journey as a fragment.
        for id, depot in self.depots_by_id.items():
            if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge:
                finish_time = (
                    job.end_time
                    + self.job_to_depot_time_matrix[job.id][depot.id]
                    + RECHARGE_TIME
                )
                fragment_set.add(
                    Fragment(
                        id=id_counter[0],
                        jobs=tuple(current_jobs),
                        start_time=current_fragment["start_time"],
                        end_time=finish_time,
                        start_depot_id=current_fragment["start_depot_id"],
                        end_depot_id=depot.id,
                        charge=current_fragment["charge"] + self.job_to_depot_charge_matrix[job.id][depot.id],
                    )
                )
                id_counter[0] += 1

        reachable_jobs = self.get_jobs_reachable_from(charge, job)
        next_fragment = current_fragment.copy()
        for next_job in reachable_jobs:
            # manage memory
            next_fragment["jobs"] = current_jobs.copy() + [next_job]
            next_fragment["charge"] += self.job_charge_matrix[job.id][next_job.id]
            # Otherwise, generate a fragment starting at the next job
            self._generate_fragment_starting_at(
                fragment_set,
                next_fragment,
                next_job,
                charge - self.job_charge_matrix[job.id][next_job.id],
                id_counter,
            )

        return fragment_set

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
                        }
                        for fragment in self.fragment_set
                    ],
                    "contracted_fragments": [
                        {
                            "jobs": [job.id for job in fragment.jobs],
                            "start_depot_ids": fragment.start_depot_ids,
                            "end_depot_ids": fragment.end_depot_ids,
                        }
                        for fragment in self.contracted_fragments
                    ],
                },
                f,
            )

    def generate_timed_network(self) -> None:
        """Creates the compressed time network for the current instance."""
        time0 = time.time()
        self.arrivals_by_depot_by_time = defaultdict(
            lambda: defaultdict(set[TimedFragment])
        )
        self.departures_by_depot_by_time = defaultdict(
            lambda: defaultdict(set[TimedFragment])
        )
        self.timed_fragment_by_id: dict[int, set[TimedFragment]] = {}
        self.timed_depots_by_fragment_id: dict[int, set[TimedDepot]] = defaultdict(set)
        for fragment in self.fragment_set:
            arrival_frag = TimedFragment(id=fragment.id, time=fragment.end_time, direction=Flow.ARRIVAL)
            departure_frag = TimedFragment(id=fragment.id, time=fragment.start_time, direction=Flow.DEPARTURE)
            self.arrivals_by_depot_by_time[fragment.end_depot_id][fragment.end_time].add(arrival_frag)
            self.departures_by_depot_by_time[fragment.start_depot_id][fragment.start_time].add(departure_frag)
            self.timed_fragment_by_id[fragment.id] = {arrival_frag, departure_frag}
        # Union of the above sets
        self.timed_fragments_by_depot_by_time: dict[int, dict[int, set[TimedFragment]]] = {
            depot_id: {
                time: self.arrivals_by_depot_by_time[depot_id][time]
                | self.departures_by_depot_by_time[depot_id][time]
                for time in list(self.departures_by_depot_by_time[depot_id].keys())
                + list(self.arrivals_by_depot_by_time[depot_id].keys())
            }
            for depot_id in self.depots_by_id
        }
        self.statistics.update(
            {
                "timed_network_generation": time.time() - time0,
                "timed_network_size": sum(len(v) for v in self.timed_fragments_by_depot_by_time.values())
            }
            )
        
        time0 = time.time()
        self.timed_nodes: set[TimedDepot] = set()
        timed_depots_by_depot = defaultdict(set[TimedDepot])
        self.timed_fragments_by_timed_depot = defaultdict(set[TimedFragment])

        for depot_id in self.depots_by_id:
            times_for_depot: list[int] = list(self.timed_fragments_by_depot_by_time[depot_id].keys())
            heapq.heapify(times_for_depot)
            previous_direction = Flow.DEPARTURE
            current_fragments = set()
            # Current fragments in the current 'block' of time
            while len(times_for_depot) != 0:
                curr_time = heapq.heappop(times_for_depot)
                timed_fragments = self.timed_fragments_by_depot_by_time[depot_id][curr_time]
                # Check if all fragments have the same type, if so can add them all and move to the next time
                current_direction = Flow.ARRIVAL if any(tf.direction == Flow.ARRIVAL for tf in timed_fragments) else Flow.DEPARTURE
                has_both_directions = any(tf.direction == Flow.ARRIVAL for tf in timed_fragments) and any(tf.direction == Flow.DEPARTURE for tf in timed_fragments)

                match previous_direction, current_direction:
                    case (
                        (Flow.DEPARTURE, Flow.DEPARTURE) 
                        | (Flow.ARRIVAL, Flow.ARRIVAL)
                        | (Flow.ARRIVAL, Flow.DEPARTURE)
                        ):
                        current_fragments.update(timed_fragments)

                    case (Flow.DEPARTURE, Flow.ARRIVAL):
                        timed_depot = TimedDepot(id=depot_id, time=curr_time)
                        
                        timed_depots_by_depot[depot_id].add(timed_depot)
                        for tf in current_fragments:
                            self.timed_depots_by_fragment_id[tf.id].add(timed_depot) 

                        self.timed_fragments_by_timed_depot[timed_depot].update(current_fragments)
                        current_fragments = timed_fragments
                
                previous_direction = current_direction if not has_both_directions else Flow.DEPARTURE

            # Add the last nodes into the model
            timed_depot = TimedDepot(id=depot_id, time=curr_time)
            timed_depots_by_depot[depot_id].add(timed_depot)
            for tf in current_fragments:
                self.timed_depots_by_fragment_id[tf.id].add(timed_depot) 
            self.timed_fragments_by_timed_depot[timed_depot].update(current_fragments)
            self.timed_depots_by_depot = {
                depot_id: sorted(timed_depots)
                for depot_id, timed_depots in timed_depots_by_depot.items()
            }
        self.statistics.update(
            {
                "timed_network_compression": time.time() - time0,
                "compressed_network_nodes": sum(len(v) for v in self.timed_depots_by_depot.values()),
            } 
        )

    def build_model(self):
        """Solves the network flow mip"""
        self.model = Model("fragment_network")
        self.fragment_vars_by_id = {
            i: self.model.addVar(vtype=GRB.BINARY, name=f"f_{i}")
            for i in self.fragments_by_id
        }
        self.waiting_arcs = {
            (d_1, d_2): self.model.addVar(vtype=GRB.INTEGER, name=f"w_{d_1}_{d_2}")
            for depot in self.timed_depots_by_depot
            for d_1, d_2 in zip(
                self.timed_depots_by_depot[depot][:-1],
                self.timed_depots_by_depot[depot][1:],
            )
        }
        self.starting_counts = {
            depot: self.model.addVar(vtype=GRB.INTEGER, name=f"sc_{depot}")
            for depot in self.timed_depots_by_depot
        }
        self.finishing_counts = {
            depot: self.model.addVar(vtype=GRB.INTEGER, name=f"fc_{depot}")
            for depot in self.timed_depots_by_depot
        }
        self.flow_balance = {}

        for depot in self.timed_depots_by_depot:
            for idx, timed_depot in enumerate(self.timed_depots_by_depot[depot]):
                name = f"flow_{str(timed_depot)}"
                if idx == 0:
                    constr = self.model.addConstr(
                        self.starting_counts[depot]
                        == quicksum(
                            self.fragment_vars_by_id[tf.id]
                            for tf in self.timed_fragments_by_timed_depot[timed_depot]
                        )
                        + self.waiting_arcs[
                            timed_depot, self.timed_depots_by_depot[depot][1]
                        ],
                        name=name,
                    )

                elif idx == len(self.timed_depots_by_depot[depot]) - 1:
                    constr = self.model.addConstr(
                        self.finishing_counts[depot]
                        == quicksum(
                            self.fragment_vars_by_id[tf.id]
                            for tf in self.timed_fragments_by_timed_depot[timed_depot]
                        )
                        + self.waiting_arcs[
                            self.timed_depots_by_depot[depot][-2], timed_depot
                        ],
                        name=name,
                    )
                else:
                    next_timed_depot = self.timed_depots_by_depot[depot][idx + 1]
                    previous_timed_depot = self.timed_depots_by_depot[depot][idx - 1]
                    constr = self.model.addConstr(
                        quicksum(
                            (1 - 2*(tf.direction == Flow.DEPARTURE))*self.fragment_vars_by_id[tf.id]
                            for tf in self.timed_fragments_by_timed_depot[timed_depot]
                        )
                        + self.waiting_arcs[(previous_timed_depot, timed_depot)] 
                        - self.waiting_arcs[timed_depot, next_timed_depot]
                        == 0,
                        name=name,
                    )

                self.flow_balance[timed_depot] = constr

        # Coverage
        fragments_by_job = defaultdict(list[int])
        for f in self.fragment_set:
            for job in f.jobs:
                fragments_by_job[job].append(f.id)

        self.coverage = {
            job: self.model.addConstr(
                quicksum(
                    self.fragment_vars_by_id[i]
                    for i in fragments_by_job[job]
                )
                == 1,
                name=f"job_{job.id}",
            )
            for job in self.jobs
        }
        # end same as start
        self.vehicle_conservation = self.model.addConstr(
            quicksum(self.starting_counts.values())
            == quicksum(self.finishing_counts.values())
        )
        self.model.setObjective(quicksum(self.starting_counts.values()))

    def set_solution(self, solution: list[set[int]], n_vehicles=None):
        all_solution_fragments = set(f for s in solution for f in s)
        if n_vehicles is not None:
            self.model.addConstr(
                quicksum(self.starting_counts.values()) == n_vehicles,
                name="vehicle_count",
            )

        for fragment_id in sorted(self.fragments_by_id):
            val = fragment_id in all_solution_fragments
            self.model.addConstr(
                self.fragment_vars_by_id[fragment_id] == int(val), name=f"set_sol_f_{fragment_id}"
            )
        
        timed_routes = self.get_validated_timed_solution(solution, n_vehicles)

        waiting_arc_flows = defaultdict(int)
        # Find the waiting arcs for each route and add those in too
        for route in timed_routes:
            route = route.copy()
            # Iterate until we hit a TimedFragment, then we reset
            prev_node = route.pop(0)
            while len(route) != 0:
                curr_node = route.pop(0)
                if isinstance(curr_node, TimedFragment):
                    # Reset the arc
                    prev_node = route.pop(0)
                    continue
                waiting_arc_flows[(prev_node, curr_node)] += 1
                prev_node = curr_node

        for arc in self.waiting_arcs:
            self.model.addConstr(
                self.waiting_arcs[arc] == waiting_arc_flows.get(arc, 0), name=f"set_sol_{arc}"
            )            
    
    def get_validated_timed_solution(self, solution: list[set[int]], expected_vehicles: int=None) -> list[list[TimedDepot | Fragment]]:
        """
        Validates the prior solution to ensure its feasbility.
        Converts a list of fragments which form a route into its consequent Timed Depot/Fragment, 
        then validates on that.
        """
        if expected_vehicles is None:
            expected_vehicles = len(solution)
        final_routes = []
        # Convert into a timed network solution.
        for route_fragments in solution:
            converted_route = []
            fragments = list(route_fragments)
            for i, f_id in enumerate(fragments):
                fragment = self.fragments_by_id[f_id]
                new_arcs = []
                # Get start/end 
                s_td, e_td = tuple(sorted(self.timed_depots_by_fragment_id[f_id]))
                if i == 0:
                    # Add start -> first fragment waiting arcs, then between each pair:
                    waiting_gaps = sorted([td for td in self.timed_depots_by_depot[s_td.id] if td < s_td])
                else:
                    # Fill any waiting gaps between the previous and now
                    waiting_gaps = [td for td in self.timed_depots_by_depot[s_td.id] if prev_td < td < s_td]
                converted_route.extend(waiting_gaps)

                if s_td not in converted_route:
                    converted_route.append(s_td)
                converted_route.extend([fragment, e_td])
                
                new_depots = []
                if i == len(fragments) - 1:
                    # Fill until end
                    new_depots = [td for td in self.timed_depots_by_depot[fragment.end_depot_id] if e_td < td]                
                    converted_route.extend(new_depots)
                prev_td = e_td
            final_routes.append(converted_route)
        
        #Validate
        if not self.validate_solution(final_routes, expected_vehicles):
            raise(ValueError("Input solution is cooked mate."))
        return final_routes

    def get_fragments_from(
            self, 
            start: TimedDepot,
            current_route: list, 
            fragments_by_timed_depot: dict[TimedDepot, set[tuple[TimedDepot, int]]], 
            flow_by_waiting_arc_start: dict[TimedDepot, dict[TimedDepot, int]]
        ) -> list[Fragment | TimedDepot]:
        """Traverse the waiting arcs and fragments to get the route from a given start depot."""

        if len(fragments_by_timed_depot[start]) != 0:
            next_depot, f_id = fragments_by_timed_depot[start].pop()
            current_route.extend([self.fragments_by_id[f_id], next_depot])

        elif len(flow_by_waiting_arc_start[start]) != 0:
            next_depot = min(flow_by_waiting_arc_start[start])
            flow_by_waiting_arc_start[start][next_depot] -= 1
            if flow_by_waiting_arc_start[start][next_depot] == 0:
                flow_by_waiting_arc_start[start].pop(next_depot)
        else:
            print(self.stringify_route(current_route))
            # print("return", current_route)
            return current_route
        self.get_fragments_from(next_depot, current_route, fragments_by_timed_depot, flow_by_waiting_arc_start)
        return current_route

    def stringify_route(self, route: list) -> list[str]:
        """Converts a route into a string"""
        return " -> ".join([loc.route_str for loc in route])

    def create_routes(self):
        """
        Sequences a set of fragments into complete routes for the job horizon.
        If it is unable to sequence them, it says something
        """
        # Sequence fragments by their start / end depots
        solution_fragment_ids = {f for f in self.fragment_vars_by_id if self.fragment_vars_by_id[f].x > 0.5}
        waiting_arcs = [(start, end, round(self.waiting_arcs[start, end].x)) for start, end in self.waiting_arcs if self.waiting_arcs[start, end].x > 0.5]

        routes = []
        fragments_by_timed_depot: dict[TimedDepot, set[tuple[TimedDepot, int]]] = defaultdict(set[tuple])
        for f_id in solution_fragment_ids:
            # Retrieve the timeddepot it starts and ends at
            start_depot, end_depot = tuple(sorted(self.timed_depots_by_fragment_id[f_id]))
            fragments_by_timed_depot[start_depot].add((end_depot, f_id))

        # Set up the waiting arc counts
        flow_by_waiting_arc_start = defaultdict(lambda : defaultdict(int))

        for start, end, count in waiting_arcs:
            flow_by_waiting_arc_start[start][end] = count

        # get the starting points for all routes: the first timed depot of each type (+ waiting arcs?)
        # for each depot's start point, start tracing a path until you hit a timed depot with no further connections.
        # if you hit a depot with no further connections, start a new route.
        for depot_id, timed_depots in self.timed_depots_by_depot.items():
            start_depot = min(timed_depots)
            # Following fragments
            while len(fragments_by_timed_depot[start_depot]) != 0 or len(flow_by_waiting_arc_start[start_depot]) != 0:
                route = self.get_fragments_from(start_depot, [start_depot], fragments_by_timed_depot, flow_by_waiting_arc_start)
                routes.append(route)
        return routes                  
            
    def convert_route_to_fragments(self, route: list) -> list[int]:
        """Converts a route into its fragments"""
        fragment_ids = []
        prev_loc = None
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
                            print(
                                f"Fragment {current_fragment} not found, interpolating..."
                            )
                            interpolated_fragments = self.interpolate_fragments(
                                current_fragment
                            )
                            if interpolated_fragments:
                                fragment_ids.extend(interpolated_fragments)
                        current_fragment = {
                            "jobs": [],
                            "start_depot_id": location.id,
                            "end_depot_id": None,
                        }

                case Job():
                    current_fragment["jobs"].append(location)

            prev_loc = location
            
        return fragment_ids

    def validate_timed_network(self):
        """
        Validates the timed network to ensure it is feasible.
        For a timed network to be feasible:
        - No timed depot can have fragments which start after its time
        - Each timed depot cannot have a time earlier than the previous timed depot
        """
        for depot_id, timed_depots in self.timed_depots_by_depot.items():
            prev_td = None
            for td in timed_depots:
                if prev_td:
                    assert td.time > prev_td.time
                    if td.time <= prev_td.time:
                        print(f"Depot {td} has a time earlier than the previous depot.")
                        return False
                fragments = self.timed_fragments_by_timed_depot[td]
                # Check the following:
                # min end time >= max start time
                # no start time later than the current depot time and no earlier than the previous time.
                start_fragments = [f for f in fragments if f.direction == Flow.DEPARTURE]
                end_fragments = [f for f in fragments if f.direction == Flow.ARRIVAL]
                if len(end_fragments) != 0:
                    assert all(prev_td.time <= tf.time <= td.time for tf in start_fragments)
                if len(start_fragments) != 0:
                    assert all(prev_td.time <= tf.time <= td.time for tf in end_fragments)
                if len(start_fragments) != 0 and len(end_fragments) != 0:
                   max_arr = max(tf.time for tf in end_fragments) 
                   min_dep = min(tf.time for tf in start_fragments)
                   assert max_arr <= min_dep
                prev_td=td

    def validate_fragment(self, fragment: Fragment, charge: int, prev_time: int) -> bool:

        if prev_time > fragment.start_time:
            print(f"Fragment {fragment} starts too late")
            raise Exception()
        time = fragment.start_time + self.depot_to_job_time_matrix[fragment.start_depot_id][fragment.jobs[0].id]

        for j, job in enumerate(fragment.jobs):
            if j != 0:
                charge -= self.job_charge_matrix[job.id][fragment.jobs[j-1].id]
                time += self.job_time_matrix[job.id][fragment.jobs[j - 1].id]
            else:
                charge -= self.depot_to_job_charge_matrix[fragment.start_depot_id][job.id]
            if time > job.start_time:
                print(f"Fragment {fragment} starts too late")
                raise Exception()
            elif charge < 0:
                print(f"Fragment {fragment} has negative charge")
                raise Exception()
        prev_time = fragment.end_time
        charge -= self.job_to_depot_charge_matrix[fragment.jobs[-1].id][fragment.end_depot_id]
        if charge < 0:
            raise Exception(f"Fragment {fragment}\n{charge} failed at the finish line.")
        return charge, prev_time

    def validate_route(self, route: list[TimedDepot | Fragment]) -> bool:
        """
        Validates a route to ensure it is feasible.
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time        
        """
        infractions = []
        charge = CHARGE_MAX
        time = prev_time = 0
        for i, location in enumerate(route):
            if isinstance(location, TimedDepot):
                # ensure timed depots connect to the correct start fragment
                charge=CHARGE_MAX
                if prev_time > location.time:
                    print(f"depot {location} is after the fragment returns")
                    raise Exception()
                # prev_time= min(td for td in self.timed_depots_by_depot[location.id] if td.time >= location.time).time

            elif isinstance(location, Fragment):
                # subtract time
                charge, prev_time = self.validate_fragment(location, charge, prev_time)

                

        return True

    def validate_solution(self, routes: list[list[TimedDepot | TimedFragment]], objective: int):
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
        return True

    def interpolate_fragments(self, candidate):
        """Interpolates a series of fragments from sequential jobs in a solution read externally."""       
        if len(candidate["jobs"]) == 0:
            print("No jobs in candidate")
            return
        # Define its start and end time
        candidate["start_time"] = (
            candidate["jobs"][0].start_time - self.depot_to_job_time_matrix[candidate["start_depot_id"]][candidate["jobs"][0].id]
            )
        candidate["end_time"] = (
            candidate["jobs"][-1].end_time + self.job_to_depot_time_matrix[candidate["jobs"][-1].id][candidate["end_depot_id"]]
        )
        self.validate_fragment(Fragment(**candidate, id=None, charge=None), charge=CHARGE_MAX, prev_time=candidate["start_time"])

        # calculate the cumulative charge
        charge_cost = (
            self.depot_to_job_charge_matrix[candidate["start_depot_id"]][
                candidate["jobs"][0].id
            ]
            + sum(
                self.job_charge_matrix[candidate["jobs"][i].id][
                    candidate["jobs"][i + 1].id
                ]
                for i in range(len(candidate["jobs"]) - 1)
            )
            + self.job_to_depot_charge_matrix[candidate["jobs"][-1].id][
                candidate["end_depot_id"]
            ]
        )
        print(f"Failed fragment costs: {charge_cost}")

        recharge_gaps = []
        interpolated_fragment_ids = []
        # calculate if there can be any recharge gaps - skip last job in array since it cannot detour before hitting a depot
        for j in range(len(candidate["jobs"]) - 2):
            if (
               (recharge_time:= candidate["jobs"][j].end_time + min(
                    self.job_to_depot_time_matrix[candidate["jobs"][j].id][depot_id]
                    + self.depot_to_job_time_matrix[depot_id][
                        candidate["jobs"][j + 1].id
                    ]
                    for depot_id in self.depots_by_id
                )
                + RECHARGE_TIME)
                <= candidate["jobs"][j + 1].start_time
            ):
                print(f"Recharge gap at {j}")
                recharge_gaps.append(j)
        # create new fragments for each recharge gap
        start_depot_id = candidate["start_depot_id"]
        previous_stop = 0
        for j in recharge_gaps:
            fragment = {
                "jobs": candidate["jobs"][previous_stop : j + 1],
                "start_depot_id": start_depot_id,
                "end_depot_id": None,
            }
            previous_stop = j + 1
            # find a valid end point such that recharge can be done, then see if that fragment exists
            for depot_id in self.depots_by_id:
                if (
                    self.job_to_depot_time_matrix[candidate["jobs"][j].id][depot_id]
                    + self.depot_to_job_time_matrix[depot_id][
                        candidate["jobs"][j + 1].id
                    ]
                    + RECHARGE_TIME
                    <= candidate["jobs"][j + 1].start_time
                ):
                    fragment["end_depot_id"] = depot_id
                    frag_id = self.get_fragment_id(fragment)
                    if frag_id:
                        print(f"Fragment {fragment} found!")
                        interpolated_fragment_ids.append(frag_id)
                        start_depot_id = depot_id
                        break

        # Create fragment out of what remains
        fragment = {
            "jobs": candidate["jobs"][previous_stop:],
            "start_depot_id": start_depot_id,
            "end_depot_id": candidate["end_depot_id"],
        }
        frag_id = self.get_fragment_id(fragment)
        if frag_id:
            print("finishing fragment found!", fragment)
            interpolated_fragment_ids.append(frag_id)
        else:
            raise Exception(f"No fragment found :( stop being bad at your job, {fragment}")
        return interpolated_fragment_ids

    def run(self):
        pass

def main():
    # file = r"data/instances_regular/I-5-5-200-10.json"
    # file = r"data/instances_large/I-7-7-1000-01.json"
    file = r"data/instances_regular/I-1-1-50-02.json"


    # Specify the directory you want to search
    directory = "data/instances_regular/"

    # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)

    # Iterate over the list of filepaths & open each file
    for json_file in json_files:     
        if "fragments" in str(json_file):
            continue
        # if "50" not in str(json_file):
        #     continue
        print(f"Solving {json_file}...")
        generator = ConstantFragmentGenerator(json_file)
        print("generating fragments...")
        # remove the last /, append fragments and then the part on the other side f the slice

        frag_file = json_file.split("/")
        prev_runs = pd.read_csv("results.csv")
        if frag_file[-1].split(".")[0] in prev_runs["label"].values:
            continue
        str_frag_file = "/".join(frag_file[:-1]) + "/fragments/" + "f-" + frag_file[-1]
        fragments = generator.generate_fragments()#file=str_frag_file)
        
        print("generating timed network...")
        generator.generate_timed_network()
        generator.validate_timed_network()
        # visualise_timed_network(generator.timed_depots_by_depot, generator.fragment_set, set(
        #     a for d in generator.timed_depots_by_depot for a in zip(generator.timed_depots_by_depot[d][:-1], generator.timed_depots_by_depot[d][1:])
        # ),
        # generator.timed_fragments_by_timed_node)
        # print("writing fragments...")
        # generator.write_fragments()
        print("building model...")
        generator.build_model()
        # print("incumbent solution")
        # prior_solution,solution_routes = generator.read_solution(instance_type=frag_file[-2].split("instances_")[-1])
        # for s in solution_routes:
        #     print(s)
        # generator.get_validated_timed_solution(prior_solution)
        # generator.set_solution(prior_solution, n_vehicles=len(prior_solution))
        # print("solving...")
        generator.solve()
        # print(f"Prior Solution: {len(solution_routes)}")
        # for s in solution_routes:
        #     print(s)
        print("sequencing routes...")
        routes = generator.create_routes()
        print(f"Fragment routes: {len(routes)}")
        generator.validate_solution(routes, generator.model.objval)
        generator.write_statistics(file="results.csv")

if __name__ == "__main__":
    main()
