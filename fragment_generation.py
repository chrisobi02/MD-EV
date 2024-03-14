import json
import os
from collections import defaultdict
import heapq
from gurobipy import Model, GRB, quicksum
import pandas as pd
from typing import TypeVar

from visualiser import visualise_timed_network

from constants import *

T = TypeVar("T")


class ConstantFragmentGenerator:
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
        self.contracted_fragments: set[ContractedFragment] = set()

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
        # TODO: BROKEN
        for i, depot in self.depots_by_id.items():
            for j, job in enumerate(self.jobs):
                if job.building_start_id == depot.id:
                    distance = self.distance(job.start_location, depot.location)
                else:
                    distance = (
                        self.building_distance[job.building_end_id][depot.id]
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
        # if start:
        #     return self.distance(self.buildings_by_id[job.building_start_id].entrance, job.start_location)
        # else:
        #     return self.distance(job.end_location, self.buildings_by_id[job.building_end_id].entrance)

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
                + self.depot_to_job_charge_matrix[depot.id][next_job.id]
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

    def generate_fragments(self):
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
        fragment_set: set[Fragment] = set()
        id_counter = [0]
        for depot_id, depot in self.depots_by_id.items():
            # starting job
            for job in self.jobs:
                charge = CHARGE_MAX - self.depot_to_job_charge_matrix[depot.id][job.id]
                start_time = (
                    job.start_time - self.depot_to_job_time_matrix[depot.id][job.id]
                )
                # if job.id == 0:
                #     print("\n\n\nJOB 0", start_time)
                #     pass
                # Commented since job 0 on I-1-1-50-01.json cannot be reached from time 0
                if start_time < 0:
                    continue
                current_fragment = {
                    "jobs": [job],
                    "start_depot_id": depot.id,
                    "end_depot_id": None,
                    "start_time": start_time,
                    "end_time": None,
                }
                self._generate_fragment_starting_at(
                    fragment_set, current_fragment, job, charge, id_counter
                )
        self.fragment_set = fragment_set
        self.fragments_by_id = {fragment.id: fragment for fragment in fragment_set}
        self.generate_contracted_fragments()
        return fragment_set

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
                    )
                )
                id_counter[0] += 1

        reachable_jobs = self.get_jobs_reachable_from(charge, job)
        next_fragment = current_fragment.copy()
        for next_job in reachable_jobs:
            # manage memory
            next_fragment["jobs"] = current_jobs.copy() + [next_job]
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

    def create_timed_lookups(self) -> None:
        self.arrivals_by_depot_by_time = defaultdict(
            lambda: defaultdict(set[TimedFragment])
        )
        self.departures_by_depot_by_time = defaultdict(
            lambda: defaultdict(set[TimedFragment])
        )

        for fragment in self.fragment_set:
            self.arrivals_by_depot_by_time[fragment.end_depot_id][
                fragment.end_time
            ].add(
                TimedFragment(
                    id=fragment.id, time=fragment.end_time, direction=Flow.ARRIVAL
                )
            )
            self.departures_by_depot_by_time[fragment.start_depot_id][
                fragment.start_time
            ].add(
                TimedFragment(
                    id=fragment.id, time=fragment.start_time, direction=Flow.DEPARTURE
                )
            )
        # Union of the above sets
        self.timed_fragments_by_depot_by_time: dict[
            int, dict[int, set[TimedFragment]]
        ] = {
            depot_id: {
                time: self.arrivals_by_depot_by_time[depot_id][time]
                | self.departures_by_depot_by_time[depot_id][time]
                for time in list(self.departures_by_depot_by_time[depot_id].keys())
                + list(self.arrivals_by_depot_by_time[depot_id].keys())
            }
            for depot_id in self.depots_by_id
        }
        return

    def generate_timed_network(self) -> None:
        """Creates the compressed time network for the current instance."""
        self.create_timed_lookups()
        self.timed_nodes: set[TimedDepot] = set()
        timed_depots_by_depot = defaultdict(set[TimedDepot])
        self.timed_fragments_by_timed_node = defaultdict(set[TimedFragment])

        for depot_id in self.depots_by_id:
            times_for_depot: list[int] = list(
                self.timed_fragments_by_depot_by_time[depot_id].keys()
            )
            heapq.heapify(times_for_depot)
            previous_direction = None
            current_fragments = set()
            # Current fragments in the current 'block' of time
            while len(times_for_depot) != 0:
                curr_time = heapq.heappop(times_for_depot)
                timed_fragments = self.timed_fragments_by_depot_by_time[depot_id][
                    curr_time
                ]
                # Check if all fragments have the same type, if so can add them all and move to the next time
                if all(
                    tf.direction == list(timed_fragments)[0].direction
                    for tf in timed_fragments
                ):
                    current_direction = list(timed_fragments)[0].direction
                else:
                    current_direction = None

                if current_direction is None:
                    # if previous direction is arrival, this is ok
                    # if it's a departure, then we dont maintain flow balance, so we need a node before it.
                    if previous_direction == Flow.DEPARTURE:
                        timed_depot = TimedDepot(id=depot_id, time=curr_time - 1)
                        timed_depots_by_depot[depot_id].add(timed_depot)
                        self.timed_fragments_by_timed_node[timed_depot].update(
                            current_fragments
                        )
                        current_fragments = set()

                    timed_depot = TimedDepot(id=depot_id, time=curr_time)
                    timed_depots_by_depot[depot_id].add(timed_depot)
                    self.timed_fragments_by_timed_node[timed_depot].update(
                        current_fragments | timed_fragments
                    )

                    previous_direction = current_direction
                    current_fragments = set()
                elif (
                    current_direction == previous_direction
                    or previous_direction is None
                ):
                    # No change in flow -> add to current fragments and continue
                    current_fragments.update(timed_fragments)
                    previous_direction = current_direction
                else:
                    # Change in flow -> encapsulate current fragments and continue
                    timed_depot = TimedDepot(id=depot_id, time=curr_time)
                    timed_depots_by_depot[depot_id].add(timed_depot)
                    self.timed_fragments_by_timed_node[timed_depot].update(
                        current_fragments
                    )
                    current_fragments = timed_fragments
                    previous_direction = current_direction

            # Add the last nodes into the model
            timed_depot = TimedDepot(id=depot_id, time=curr_time)
            timed_depots_by_depot[depot_id].add(timed_depot)
            self.timed_fragments_by_timed_node[timed_depot].update(current_fragments)
            self.timed_depots_by_depot = {
                depot_id: sorted(timed_depots)
                for depot_id, timed_depots in timed_depots_by_depot.items()
            }

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
                            for tf in self.timed_fragments_by_timed_node[timed_depot]
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
                            for tf in self.timed_fragments_by_timed_node[timed_depot]
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
                            for tf in self.timed_fragments_by_timed_node[timed_depot]
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

    def set_solution(self, solution):
        for fragment_id in self.fragments_by_id:
            val = fragment_id in solution
            self.model.addConstr(
                self.fragment_vars_by_id[fragment_id] == val, name=f"set_sol_{fragment_id}"
            )

    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("fragment_network.ilp")
            for fragment in self.fragment_set:
                if 0 in fragment.jobs:
                    print(fragment)
        else:
            print(self.model.objval)

    def read_solution(self) -> set[int]:
        """Reads the solution given by the paper"""
        data = pd.read_excel(
            r"data/mdevs_solutions.xlsx", sheet_name="results_regular_CPLEX"
        )
        curr_sol_str = data.query(
            f"ID_instance == {self.data['ID']} and battery_charging == 'constant-time'"
        )["solution"].values[0]
        solution_fragment_ids = set()
        num_depots = len(self.depots_by_id)
        num_jobs = len(self.jobs_by_id)
        print(len(curr_sol_str.split(",")))
        print(curr_sol_str)
        # import re
        # print(sorted([int(match) for match in re.findall(r'(?<=\>)\d+(?=\>)', curr_sol_str)]), len([int(match) - 3 for match in re.findall(r'(?<=\>)\d+(?=\>)', curr_sol_str)]))
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
            solution_fragment_ids.update(self.convert_route_to_fragments(current_route))
        return solution_fragment_ids

    def convert_route_to_fragments(self, route: list):
        """Converts a route into its fragments"""
        fragment_ids = set()
        prev_loc = None
        current_fragment = {
            "jobs": [],
            "start_depot_id": None,
            "end_depot_id": None,
        }

        fragments_in_order = []
        for i, location in enumerate(route):
            match location:
                case Building():
                    # Starting point
                    if not prev_loc:
                        current_fragment["start_depot_id"] = location.id
                    else:
                        # figure out if can recharge before the next task
                        current_fragment["end_depot_id"] = location.id
                        frag_id = self.get_fragment_id(current_fragment)
                        if frag_id:
                            fragments_in_order.append(frag_id)
                            fragment_ids.add(frag_id)
                        else:
                            print(
                                f"Fragment {current_fragment} not found, interpolating..."
                            )
                            interpolated_fragments = self.interpolate_fragments(
                                current_fragment
                            )
                            if interpolated_fragments:
                                fragment_ids.update(interpolated_fragments)
                            # fragment_ids.update(self.interpolate_fragments(current_fragment))
                        current_fragment = {
                            "jobs": [],
                            "start_depot_id": location.id,
                            "end_depot_id": None,
                        }

                case Job():
                    current_fragment["jobs"].append(location)

            prev_loc = location
        return fragment_ids

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

    def interpolate_fragments(self, candidate):
        """Interpolates a series of fragments from sequential jobs in a solution read externally."""
        if len(candidate["jobs"]) == 0:
            print("No jobs in candidate")
            return

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
                min(
                    self.job_to_depot_time_matrix[candidate["jobs"][j].id][depot_id]
                    + self.depot_to_job_time_matrix[depot_id][
                        candidate["jobs"][j + 1].id
                    ]
                    for depot_id in self.depots_by_id
                )
                + RECHARGE_TIME
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
        return interpolated_fragment_ids

    def visualise_solution(self, sol=None):
        """Visualises the solution found by gurobi in the network"""

        # Piece together the routes from the fragments
        if sol is None:
            sol_fragments = {f for f in self.fragment_vars_by_id if self.fragment_vars_by_id[f].x > 0.5} 
        else:
            sol_fragments = sol
    
        vehicle_arcs = []
        for f in sol_fragments:
            used_depots = []
            for depot in self.timed_depots_by_depot:
                for td in self.timed_depots_by_depot[depot]:
                    if f in [t.id for t in self.timed_fragments_by_timed_node[td]]:
                        used_depots.append(td)
            vehicle_arcs.append(sorted(used_depots))
        waiting_arcs = [{"timed_depot": arc, "flow": round(self.waiting_arcs[arc].x)} for arc in self.waiting_arcs]# if self.waiting_arcs[arc].x > 0.5]
        timed_depots = [td for depot in self.timed_depots_by_depot for td in self.timed_depots_by_depot[depot]]

        visualise_timed_network(
            timed_depots,
            vehicle_arcs,
            waiting_arcs,
            instance_label=self.data["label"],
            charge_type="constant-time",
        )

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



def main():
    file = r"data/instances_regular/I-5-5-200-10.json"
    file = r"data/instances_large/I-7-7-1000-01.json"
    file = r"data/instances_regular/I-1-1-50-01.json"
    generator = ConstantFragmentGenerator(file)
    print("generating fragments...")
    fragments = generator.generate_fragments()

    print("generating timed network...")
    generator.generate_timed_network()
    # visualise_timed_network(generator.timed_depots_by_depot, generator.fragment_set, set(
    #     a for d in generator.timed_depots_by_depot for a in zip(generator.timed_depots_by_depot[d][:-1], generator.timed_depots_by_depot[d][1:])
    # ),
    # generator.timed_fragments_by_timed_node)
    print("writing fragments...")
    generator.write_fragments()
    print("building model...")
    generator.build_model()
    # print("incumbent solution")
    prior_solution = generator.read_solution()
    generator.set_solution(prior_solution)
    print("solving...")
    generator.solve()
    print("visualising solution...")
    generator.visualise_solution(sol = prior_solution)
    print("displaying stats...")
    generator.display_stats()
    pass

    # print(any(
    #     list(j.id +3 for j in f.jobs) == [14,35]
    #     for f in fragment.fragment_set
    # ))
    # print(fragments)


if __name__ == "__main__":
    main()
