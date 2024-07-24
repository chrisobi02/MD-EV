from collections import defaultdict
import math
import time as timer

from formulations.base import *

class NonLinearFragmentGenerator(BaseFragmentGenerator):
    def __init__(self, file: str, config: dict = {}):
        super().__init__(file, params=config)
        self.dilation_factor = self.config.UNDISCRETISED_MAX_CHARGE / 100 * self.config.CHARGE_PER_UNIT 
        self.timed_depots_by_depot: dict[int, list[ChargeDepot]] = defaultdict(list)

    def get_charge(self, charge: int, recharge_time: int):
        """
        Determines the state of charge given a charge level and the amount of time it has to charge.
        """
        if recharge_time <= 0:
            return charge
        return self.get_charge_at_time(self.charge_inverse(charge) + recharge_time)


    def get_charge_at_time(self, t: int) -> int:
        """returns the charge level from 0% when charging for t units of time"""

        if t <= 80 * self.dilation_factor:
            charge = 2 * t
        elif t <= 160 * self.dilation_factor:
            charge = 640/3 - 12800 / (3*t - 160)
        else:
            charge = 200
        return math.floor(charge * self.dilation_factor)
    
    def charge_inverse(self, charge: int):
        """Returns the time to charge to a certain level"""
        if charge <= 160 * self.dilation_factor:
            t = charge / 2
        elif charge <= 200 * self.dilation_factor:
            t = (38400 - 160 * charge) / (640 - 3 * charge)
        else:
            t = 160
        return math.ceil(t / self.dilation_factor)

    def _get_jobs_reachable_from(
            self,
            charge: int,
            job: Job,
            current_sequence: list[Job]=[],
        ) -> set[Fragment]:
        """
        Generates a fragment starting at a given job and depot with a given charge at a given time.
        The three conditions that must be satisfied are:
        1. The candidate job must be reachable before its start time.
        2. The candidate job must be able to reach a depot from the new job.
        3. A detour to a depot will not result in the vehicle reaching the job with a higher charge than it would have directly.
        """
        reachable_jobs = []
        t = job.end_time
        for next_job in self.jobs:
            # Filter out jobs already covered
            if job == next_job:
                continue
            if any(next_job == prev_job for prev_job in current_sequence):
                continue

            # 1.
            arrival_time = t + self.job_time_matrix[job.id][next_job.id]
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
                self.depots, 
                key=(
                    lambda depot: self.job_to_depot_time_matrix[job.id][depot.id]
                    + self.depot_to_job_time_matrix[depot.id][next_job.id]
                )
            )
            recharge_time = (
                next_job.start_time 
                - t
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
                # Can reach job with a higher charge than if it were reached directly
                continue
            reachable_jobs.append(next_job)

        return reachable_jobs

    def generate_timed_network(self) -> None:
        time0 = timer.time()
        self.timed_fragments_by_depot_by_time = defaultdict(lambda: defaultdict(set[TimedFragment]))
        self.timed_fragment_by_id: dict[int, set[TimedFragment]] = {}
        self.timed_depots_by_fragment_id: dict[int, TimedDepotStore] = defaultdict(TimedDepotStore)
        for fragment in self.fragment_set:
            arrival_frag = TimedFragment(id=fragment.id, time=fragment.end_time, direction=Flow.ARRIVAL)
            departure_frag = TimedFragment(id=fragment.id, time=fragment.start_time, direction=Flow.DEPARTURE)
            self.timed_fragments_by_depot_by_time[fragment.end_depot_id][fragment.end_time].add(arrival_frag)
            self.timed_fragments_by_depot_by_time[fragment.start_depot_id][fragment.start_time].add(departure_frag)
            self.timed_fragment_by_id[fragment.id] = [arrival_frag, departure_frag]
        
        self.statistics.update(
            {
                "timed_network_generation": timer.time() - time0,
                "timed_network_size": sum(len(v) for v in self.timed_fragments_by_depot_by_time.values())
            }
        )

        self.timed_depots_by_depot = defaultdict(list)
        for depot_id in self.depots_by_id:
            for time in self.timed_fragments_by_depot_by_time[depot_id]:
                self.timed_depots_by_depot[depot_id].append(
                    ChargeDepot(time=time, id=depot_id, charge=self.config.MAX_CHARGE)
                )

    def validate_route(self, route: list[TimedDepot | Fragment]) -> bool:
        """
        Validates a route to ensure it is feasible.
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time        
        """
        raise NotImplementedError()
        infractions = []
        charge = self.config.MAX_CHARGE
        current_time = prev_time = 0
        # print("validating the following:")
        for i, location in enumerate(route):
            if isinstance(location, ChargeDepot):
                pass
            elif isinstance(location, Fragment):
                charge, prev_time = self.validate_fragment(location, charge, prev_time)
        return True
    
    def get_validated_timed_solution(self, solution: list[set[int]], expected_vehicles: int = None) -> list[list[TimedDepot | Fragment]]:
        pass

    def run(self):
        pass
