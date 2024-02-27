import json
import os
from dataclasses import dataclass
from collections import defaultdict

# speed in metres per hour
VEHICLE_MOVE_SPEED_PER_HOUR = 10000
# Length of 1 time unit in minutes
TIME_UNIT_IN_MINUTES = 0.5
VEHICLE_MOVE_SPEED_PER_UNIT = 1000 / 6 * TIME_UNIT_IN_MINUTES
# percentage of max charge used per metre
CHARGE_PER_METRE = 0.05 * 100 / 1000
# percentage of maximum charge per unit of charge
CHARGE_PER_UNIT = 0.5
CHARGE_MAX = 100 / CHARGE_PER_UNIT
# Time cost of swapping batteries over in time units
RECHARGE_TIME = 6

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
    location: tuple = None # Only applies to depots

@dataclass(frozen=True)
class Fragment:
    jobs: tuple[Job] # An ordered tuple of jobs
    start_time: int
    end_time: int
    start_depot_id: int
    end_depot_id: int

@dataclass(frozen=True)
class ContractedFragment:
    jobs: tuple[Job]
    start_depot_ids: list[int]
    end_depot_ids: list[int]

class FragmentGenerator:
    def __init__(self, file: str) -> None:
        # self.read_json(file)
        self.base_dir = os.path.dirname(file)
        self.data = json.load(open(file))
        self.buildings_by_id: dict[int, Building] = self.to_dataclass_by_id(self.data["buildings"], Building)
        self.buildings: list[Building] = list(self.buildings_by_id.values())
        self.depots_by_id: dict[int, Building] = {building.id: building for building in self.buildings if building.type == "depot"} 
        self.depots: list[Building] = list(self.depots_by_id.values())
        self.jobs_by_id: dict[int, Job] = self.to_dataclass_by_id(self.data["jobs"], Job)
        self.jobs: list[Job] = list(self.jobs_by_id.values())
        self.generate_building_distance_matrix()
        self.generate_job_cost_matrix()
        self.generate_job_to_depot_matrices()
        self.generate_depot_to_job_matrices()
        self.fragment_set: set[Fragment] = set()
        self.contracted_fragments: set[ContractedFragment] = set()


    def to_dataclass_by_id(self, json_data: list[dict], dataclass: object, id_field: str="id") -> dict[int, object]:
       return {
           data[id_field] if id_field else i: dataclass(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
            for i, data in enumerate(json_data)
            }
     

    def distance(self, start: list, end: list) -> int:
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    

    def generate_building_distance_matrix(self) -> list[list[int]]:
        self.building_distance = [[0 for _ in range(len(self.buildings))] for _ in range(len(self.buildings))]

        for i, building in enumerate(self.buildings):
            for j, other_building in enumerate(self.buildings):
                if building.id == other_building.id:
                    continue
                distance = self.distance(building.entrance, other_building.entrance)
                self.building_distance[i][j] = distance

        return self.building_distance


    def generate_job_to_depot_matrices(self) -> list[list[int]]:
        """Generates the charge and time cost for going from a job to a depot. [job_id][depot_id]."""
        self.job_to_depot_distance_matrix = [[0 for _ in range(len(self.depots_by_id))] for _ in range(len(self.jobs))]
        self.job_to_depot_charge_matrix = [[0 for _ in range(len(self.depots_by_id))] for _ in range(len(self.jobs))]
        self.job_to_depot_time_matrix = [[0 for _ in range(len(self.depots_by_id))] for _ in range(len(self.jobs))]
        for i, job in enumerate(self.jobs):
            for j, depot in self.depots_by_id.items():
                distance = (
                    self.building_distance[job.building_end_id][depot.id]
                    + self.get_internal_job_distance(job, start=False)
                    + self.distance(depot.entrance, depot.location)
                    )
                self.job_to_depot_distance_matrix[i][j] = distance
                self.job_to_depot_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.job_to_depot_time_matrix[i][j] = self.distance_to_time(distance)

        return self.job_to_depot_distance_matrix, self.job_to_depot_charge_matrix, self.job_to_depot_time_matrix


    def generate_depot_to_job_matrices(self) -> list[list[int]]:
        """Generates the charge and time cost for going from a depot to a job"""
        self.depot_to_job_distance_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.depots_by_id))]
        self.depot_to_job_charge_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.depots_by_id))]
        self.depot_to_job_time_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.depots_by_id))]

        for i, depot in self.depots_by_id.items():
            for j, job in enumerate(self.jobs):
                distance = (
                    self.building_distance[job.building_end_id][depot.id]
                    + self.get_internal_job_distance(job)
                    + self.distance(depot.entrance, depot.location)
                    )
                self.depot_to_job_distance_matrix[i][j] = distance
                self.depot_to_job_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.depot_to_job_time_matrix[i][j] = self.distance_to_time(distance)

        return self.depot_to_job_distance_matrix, self.depot_to_job_charge_matrix, self.depot_to_job_time_matrix


    def get_internal_job_distance(self, job: Job, start=True):
        """Returns the distance of the job to the"""
        if start:
            return self.distance(self.buildings_by_id[job.building_start_id].entrance, job.start_location)
        else:
            return self.distance(job.end_location, self.buildings_by_id[job.building_end_id].entrance)


    def generate_job_cost_matrix(self) -> list[list[int]]:
        """Generates the charge and time cost """
        self.job_distance_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))]
        self.job_charge_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))]
        self.job_time_matrix = [[0 for _ in range(len(self.jobs))] for _ in range(len(self.jobs))]
        for i, job in enumerate(self.jobs):
            for j, other_job in enumerate(self.jobs):
                # NAIVE: if slow, filter to jobs that can feasibly be connected
                if job.id == other_job.id:
                    continue
                if job.building_end_id == other_job.building_start_id:
                    distance = self.distance(job.end_location, other_job.start_location)
                else:
                    # Move between buildings
                    building_distance = self.building_distance[job.building_end_id][other_job.building_start_id]
                    # Move to entrance of buildings
                    travel_distance = (
                        self.get_internal_job_distance(job, start=False)
                        + self.get_internal_job_distance(other_job, start=True)
                    )
                    distance = building_distance + travel_distance

                self.job_distance_matrix[i][j] = distance
                self.job_charge_matrix[i][j] = self.distance_to_charge(distance)
                self.job_time_matrix[i][j] = self.distance_to_time(distance)

        return self.job_distance_matrix


    def distance_to_time(self, distance: int) -> int:
        # round to 
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
                self.job_to_depot_time_matrix[job.id][depot.id] + RECHARGE_TIME + self.depot_to_job_charge_matrix[depot.id][next_job.id] 
                for depot in self.depots 
                if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge 
                )
            # 2.
            if t + recharge_time <= next_job.start_time:
                continue

            charge_cost = (
                self.job_charge_matrix[job.id][next_job.id] 
                + min(self.job_to_depot_charge_matrix[next_job.id][depot.id] for depot in self.depots_by_id.values())
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
        for depot_id, depot in self.depots_by_id.items():
            # starting job
            for job in self.jobs:
                charge = CHARGE_MAX - self.depot_to_job_charge_matrix[depot.id][job.id]
                current_fragment = {
                    "jobs": [job],
                    "start_depot_id": depot.id,
                    "end_depot_id": None,
                    "start_time": job.start_time - self.depot_to_job_time_matrix[depot.id][job.id],
                    "end_time": None
                    }
                self._generate_fragment_starting_at(fragment_set, current_fragment, job, charge)
        self.fragment_set = fragment_set
        self.generate_contracted_fragments()
        return fragment_set


    def generate_contracted_fragments(self):
        """Contracts the fragments by finding all fragments with the same jobs covered and combining the ids."""
        if len(self.fragment_set) == 0:
            self.generate_fragments()
        depots_by_job_list = defaultdict(lambda : defaultdict(set))
        for fragment in self.fragment_set:
            depots_by_job_list[tuple(fragment.jobs)]["start_depot_ids"].add(fragment.start_depot_id)
            depots_by_job_list[tuple(fragment.jobs)]["end_depot_ids"].add(fragment.end_depot_id)
        
        for jobs, depot_info in depots_by_job_list.items():
            self.contracted_fragments.add(
                ContractedFragment(
                    jobs=jobs,
                    start_depot_ids=tuple(depot_info["start_depot_ids"]),
                    end_depot_ids=tuple(depot_info["end_depot_ids"])
                )
            )



    def _generate_fragment_starting_at(self, fragment_set: set[Fragment], current_fragment: dict, job: Job, charge: int) -> set[Fragment]:
        """
        Generates a fragment starting at a given job and depot with a given charge at a given time.
        """
        current_jobs: list[Job] = current_fragment["jobs"]
        # Get all jobs which can be reached from the current job
        # Add this partial part of the journey as a fragment.
        for id, depot in self.depots_by_id.items():
            if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge:
                finish_time = job.end_time + self.job_to_depot_time_matrix[job.id][depot.id] + RECHARGE_TIME
                fragment_set.add(
                    Fragment(
                        jobs=tuple(current_jobs), 
                        start_time=current_fragment["start_time"], 
                        end_time=finish_time,
                        start_depot_id=current_fragment["start_depot_id"],
                        end_depot_id=depot.id
                        )
                    )
                
        reachable_jobs = self.get_jobs_reachable_from(charge, job)
        next_fragment = current_fragment.copy()
        for next_job in reachable_jobs:
            # manage memory
            next_fragment["jobs"] = current_jobs.copy() + [next_job]
            # Otherwise, generate a fragment starting at the next job
            self._generate_fragment_starting_at(fragment_set, next_fragment, next_job, charge - self.job_charge_matrix[job.id][next_job.id])

        return fragment_set

    def write_fragments(self) -> None:
        """Utility method to save the fragments in a json format."""

        with open(f"{self.base_dir}/fragments/f-{self.data['label']}.json", "w") as f:
            json.dump(
                {   
                    "label": self.data["label"],
                    "fragments": [
                        {
                            "jobs": [job.id for job in fragment.jobs],
                            "start_time": fragment.start_time,
                            "end_time": fragment.end_time,
                            "start_depot_id": fragment.start_depot_id,
                            "end_depot_id": fragment.end_depot_id
                        }
                        for fragment in self.fragment_set
                    ],
                    "contracted_fragments": [
                        {
                            "jobs": [job.id for job in fragment.jobs],
                            "start_depot_ids": fragment.start_depot_ids,
                            "end_depot_ids": fragment.end_depot_ids
                        }
                        for fragment in self.contracted_fragments
                    ]
                },
                f
            )




def main():
    file = "data/instances_regular/I-5-5-200-10.json"
    # file = r"data/instances_regular/I-1-1-50-01.json"
    fragment = FragmentGenerator(file)
    fragments = fragment.generate_fragments()
    fragment.write_fragments()
    print(len(fragments))
    print(len(fragment.contracted_fragments))
    # print(fragments)

if __name__ == "__main__":
    main()