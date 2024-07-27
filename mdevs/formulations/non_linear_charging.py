from collections import defaultdict
import math
import time as timer

from mdevs.formulations.base import *
from mdevs.formulations.base import Fragment, TimedDepot

class NonLinearFragmentGenerator(BaseFragmentGenerator):
    def __init__(self, file: str, config: dict = {}):
        super().__init__(file, params=config)
        self.dilation_factor = self.config.UNDISCRETISED_MAX_CHARGE / 100 * self.config.CHARGE_PER_UNIT 
        self.timed_depots_by_depot: dict[int, list[ChargeDepot]] = defaultdict(list)
        self.fragments_by_timed_depot: dict[TimedDepot, set[Fragment]] = defaultdict(set[Fragment])

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
        job_end_time = job.end_time
        for next_job in self.jobs:
            # Filter out jobs already covered
            if job == next_job:
                continue
            if any(next_job == prev_job for prev_job in current_sequence):
                continue

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
                self.depots, 
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
                # Can reach job with a higher charge than if it were reached directly
                continue
            reachable_jobs.append(next_job)

        return reachable_jobs

    def generate_timed_network(self) -> None:
        time0 = timer.time()
        self.fragments_by_timed_depot
        self.timed_fragment_by_id: dict[int, set[TimedFragment]] = {}
        self.timed_depots_by_fragment_id: dict[int, TimedDepotStore] = defaultdict(TimedDepotStore)
        for fragment in self.fragment_set:
            arrival_depot = ChargeDepot(id=fragment.start_depot_id, time=fragment.start_time, charge=self.config.MAX_CHARGE)
            end_depot = ChargeDepot(id=fragment.end_depot_id, time=fragment.end_time, charge=self.config.MAX_CHARGE)
            self.fragments_by_timed_depot[arrival_depot].add(fragment)
            self.fragments_by_timed_depot[end_depot].add(fragment)

            self.timed_depots_by_depot[arrival_depot.id].append(arrival_depot)
            self.timed_depots_by_depot[end_depot.id].append(end_depot)

            self.timed_fragment_by_id[fragment.id] = fragment
        
        self.statistics.update(
            {
                "timed_network_generation": timer.time() - time0,
                "timed_network_size": sum(len(v) for v in self.fragments_by_timed_depot.values())
            }
        )

        self.timed_depots_by_depot = defaultdict(list)

    def _get_flow_coefficient(self, depot: ChargeDepot, fragment: Fragment) -> int:
        """
        Retrieves the direction of flow between a depot and a fragment
        This is based on matching the fragment and the 
        """
        if depot.time == fragment.start_time and depot.id == fragment.start_depot_id:
            return 1
        elif depot.time == fragment.end_time and depot.id == fragment.end_depot_id:
            return -1
        else:
            raise ValueError("Fragment and depot do not have a common time / location")


    def add_flow_balance_constraints(self):
        self.flow_balance = {}
        for depot in self.timed_depots_by_depot:
            for idx, charge_depot in enumerate(self.timed_depots_by_depot[depot]):
                name = f"flow_{str(charge_depot)}"
                if idx == 0:
                    constr = self.model.addConstr(
                        self.starting_counts[depot]
                        == quicksum(
                            self.fragment_vars_by_id[fragment.id]
                            for fragment in self.fragments_by_timed_depot[charge_depot]
                        )
                        + self.waiting_arcs[
                            charge_depot, self.timed_depots_by_depot[depot][1]
                        ],
                        name=name,
                    )

                elif idx == len(self.timed_depots_by_depot[depot]) - 1:
                    constr = self.model.addConstr(
                        self.finishing_counts[depot]
                        == quicksum(
                            self.fragment_vars_by_id[fragment.id]
                            for fragment in self.fragments_by_timed_depot[charge_depot]
                        )
                        + self.waiting_arcs[
                            self.timed_depots_by_depot[depot][-2], charge_depot
                        ],
                        name=name,
                    )
                else:
                    next_timed_depot = self.timed_depots_by_depot[depot][idx + 1]
                    previous_timed_depot = self.timed_depots_by_depot[depot][idx - 1]
                    constr = self.model.addConstr(
                        quicksum(
                            self._get_flow_coefficient(charge_depot, fragment) * self.fragment_vars_by_id[fragment.id]
                            for fragment in self.fragments_by_timed_depot[charge_depot]
                        )
                        + self.waiting_arcs[previous_timed_depot, charge_depot] 
                        - self.waiting_arcs[charge_depot, next_timed_depot]
                        == 0,
                        name=name,
                    )

                self.flow_balance[charge_depot] = constr

    def validate_route(self, route: list[TimedDepot | Fragment]) -> bool:
        """
        Validates a route to ensure it is feasible.
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time
        """
        infractions = []
        current_ = self.config.MAX_CHARGE
        prev_location: TimedDepot | Fragment = None
        # print("validating the following:")
        for i, location in enumerate(route):
            if isinstance(location, ChargeDepot):
                pass
            elif isinstance(location, Fragment):
                current_, prev_time = self.validate_fragment(location, current_, prev_time)
        return True
     
    def get_validated_timed_solution(self, solution: list[set[int]], expected_vehicles: int = None) -> list[list[TimedDepot | Fragment]]:
        raise NotImplementedError()

    def validate_job_sequences(self, routes: list[list[TimedDepot | Fragment]]) -> bool:
        raise NotImplementedError()

    def run(self):
        """Runs an end-to-end solve ."""
        print(f"Solving {self.data['label']}...")
        print("generating fragments...")
        self.generate_fragments()
        print("generating timed network...")
        self.generate_timed_network()
        print("validating timed network...")
        self.validate_timed_network()
        print("building model...")
        self.build_model()
        print("solving...")
        self.solve()
        # print("sequencing routes...")
        # routes = self.create_routes()

        # print("validating solution...")
        # self.validate_solution(routes, self.model.objval, triangle_inequality=False)
        self.write_statistics()
        pass
