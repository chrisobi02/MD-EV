import json
import os
import time
from collections import defaultdict
import heapq
from typing import TypeVar
import glob

from formulations.base import *

T = TypeVar("T")

class ConstantFragmentGenerator(BaseFragmentGenerator):
    def __init__(self, file: str, params: dict={}) -> None:
        super().__init__(file, params=params)
        self.type = "constant-charging"
        self.statistics.update(
            {
                "type": self.type,
                "label": self.data["label"],
            }
        )
        
    def _get_jobs_reachable_from(self, charge: int, job: Job) -> list[Job]:
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
            # 3.
            charge_cost = next_job.charge + self.job_charge_matrix[job.id][next_job.id] + min(
                self.job_to_depot_charge_matrix[next_job.id][depot.id]
                for depot in self.depots_by_id.values()
            )
            if charge < charge_cost:
                # Cannot reach job and recharge.
                continue
            # 1
            arrival_time = t + self.job_time_matrix[job.id][next_job.id]
            if next_job.start_time < arrival_time:
                # Cannot reach job at start time
                continue

            recharge_time = min(
                self.job_to_depot_time_matrix[job.id][depot.id]
                + self.config.RECHARGE_TIME
                + self.depot_to_job_time_matrix[depot.id][next_job.id]
                for depot in self.depots
                if self.job_to_depot_charge_matrix[job.id][depot.id] <= charge
            )
            # 2.
            if t + recharge_time <= next_job.start_time:
                continue


            reachable_jobs.append(next_job)
        return reachable_jobs

    def generate_timed_network(self) -> None:
        """Creates the compressed time network for the current instance."""
        time0 = time.time()
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
                "timed_network_generation": time.time() - time0,
                "timed_network_size": sum(len(v) for v in self.timed_fragments_by_depot_by_time.values())
            }
        )
        
        time0 = time.time()
        self.timed_nodes: set[TimedDepot] = set()
        timed_depots_by_depot = defaultdict(set[TimedDepot])

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
                current_direction = (
                    Flow.ARRIVAL if any(tf.direction == Flow.ARRIVAL for tf in timed_fragments) else Flow.DEPARTURE
                )
                has_both_directions = (
                    any(tf.direction == Flow.ARRIVAL for tf in timed_fragments) 
                    and 
                    any(tf.direction == Flow.DEPARTURE for tf in timed_fragments)
                )

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
                            if tf.direction == Flow.DEPARTURE:
                                self.timed_depots_by_fragment_id[tf.id].start = timed_depot
                            else:
                                self.timed_depots_by_fragment_id[tf.id].end = timed_depot

                        self.timed_fragments_by_timed_depot[timed_depot].update(current_fragments)
                        current_fragments = timed_fragments
                
                previous_direction = current_direction if not has_both_directions else Flow.DEPARTURE

            # Add the last nodes into the model
            timed_depot = TimedDepot(id=depot_id, time=curr_time)
            timed_depots_by_depot[depot_id].add(timed_depot)
            for tf in current_fragments:
                if tf.direction == Flow.DEPARTURE:
                    self.timed_depots_by_fragment_id[tf.id].start = timed_depot
                else:
                    self.timed_depots_by_fragment_id[tf.id].end = timed_depot
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
    
    def get_validated_timed_solution(
            self, solution: list[list[int]], expected_vehicles: int | None=None
        ) -> list[list[TimedDepot | Fragment]]:
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
                # Get start/end 
                s_td, e_td = self.timed_depots_by_fragment_id[f_id].start, self.timed_depots_by_fragment_id[f_id].end
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
                if f_id in [2234, 7869]:
                    pass
                
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
            return current_route
        self.get_fragments_from(next_depot, current_route, fragments_by_timed_depot, flow_by_waiting_arc_start)
        return current_route

    def create_routes(self) -> list[list[TimedDepot | Fragment]]:
        """
        Sequences a set of fragments into complete routes for the job horizon.
        If it is unable to sequence them, it says something
        """
        solution_fragment_ids = {f for f in self.fragment_vars_by_id if self.fragment_vars_by_id[f].x > 0.5}
        # Sequence fragments by their start / end depots
        waiting_arcs = [(start, end, round(self.waiting_arcs[start, end].x)) for start, end in self.waiting_arcs if self.waiting_arcs[start, end].x > 0.5]

        routes = []
        fragments_by_timed_depot: dict[TimedDepot, set[tuple[TimedDepot, int]]] = defaultdict(set[tuple])
        for f_id in solution_fragment_ids:
            # Retrieve the timeddepot it starts and ends at
            start_depot, end_depot = self.timed_depots_by_fragment_id[f_id].start, self.timed_depots_by_fragment_id[f_id].end
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

    def validate_route(self, route: list[TimedDepot | Fragment]) -> bool:
        """
        Validates a route to ensure it is feasible.
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time        
        """
        charge = self.config.MAX_CHARGE
        prev_time = 0
        # print("validating the following:")
        for i, location in enumerate(route):
            if isinstance(location, TimedDepot):
                # ensure timed depots connect to the correct start fragment
                charge = self.config.MAX_CHARGE
                if prev_time > location.time:
                    print(f"depot {location} is after the fragment returns")
                    raise Exception()
                
            elif isinstance(location, Fragment):
                charge, prev_time = self.validate_fragment(location, charge, prev_time)
        return True

    def validate_solution(self, routes: list[list[TimedDepot | TimedFragment]], objective: int, triangle_inequality: bool=True):
        """
        Validates the input solution routes to ensure they are feasible.
        For a solution to be feasible:
        - All routes must be feasible
        - All jobs must be served
        - Must have as many routes as objective value
        """
        # all jobs must be served
        covered_jobs = {job for r in routes for f in r if isinstance(f, Fragment) for job in f.jobs}
        assert (
            covered_jobs == set(self.jobs), 
            f"All jobs must be served in the solution, missing {set(self.jobs).difference(covered_jobs)}"
        )

        if len(routes) != objective:
            print(f"Objective value {objective} does not match number of routes {len(routes)}")
            return False
        for route in routes:
            if not self.validate_route(route):
                return False

        if triangle_inequality:
            self.validate_job_sequences(routes)
    
    def validate_job_sequences(self, routes: list[list[TimedDepot | Fragment]]):
        """
        Validates no job sequenecs A - B - C occur such that A - C is illegal.
        """
        job_sequences = [[j for f in r if isinstance(f, Fragment) for j in f.jobs] for r in routes]
        invalid_jobs = self.validate_triangle_inequality()
        if len(invalid_jobs) == 0:
            print("No invalid Jobs found.")
            return
        job_sequences = self.job_sequences
        for sequence in job_sequences:
            if any(invalid in sequence for invalid in invalid_jobs):
                print("Job sequence contains invalid fragments.")
            
            for i in range(len(sequence)-2):
                job_a = sequence[i]
                job_b = sequence[i+1]
                job_c = sequence[i+2]

                # check if a to c is invalid
                a_to_c_time = job_a.end_time + self.job_time_matrix[job_a.id][job_c.id]
                b_to_c_time = job_b.end_time + self.job_time_matrix[job_b.id][job_c.id]
                if a_to_c_time > b_to_c_time:
                    print(f"Job sequence {job_a.id} - {job_b.id} - {job_c.id} is invalid.")
                    raise Exception()
                    return False
                # check charge
                a_to_c_charge = self.job_charge_matrix[job_a.id][job_c.id]
                b_to_c_charge = (
                    self.job_charge_matrix[job_a.id][job_b.id]
                    + self.job_charge_matrix[job_b.id][job_c.id]
                    + job_b.charge
                )
                if a_to_c_charge > b_to_c_charge:
                    print(f"Job sequence {job_a.id} - {job_b.id} - {job_c.id} is invalid.")
                    raise Exception()
                    return False
        return True

    def validate_triangle_inequality(self) -> list[Job]:
        """
        Ensures the instance data meets the triangle inequality for charge and time costs.
        Note this is checking a lower bound, since the jobs should take longer than this as they are stopping etc.
        """
        # Check travel time from job start to end location is less than the job time
        violations = []
        for job in self.jobs:
            job_time = job.end_time - job.start_time
            if job.building_end_id == job.building_start_id:
                job_distance = self.distance(job.start_location, job.end_location)
            else:
                job_distance = (
                    self.building_distance[job.building_start_id][job.building_end_id] 
                    + self.get_internal_distance(job.start_location) 
                    + self.get_internal_distance(job.end_location)
                )
            if self.distance_to_time(job_distance) > job_time:
                violations.append(
                    {
                    "job": job,
                    "job_time": job_time,
                    "travel_time": self.distance_to_time(job_distance),
                    "distance": job_distance,
                    }
                )
            
            # use time to calculate charge through distance.
            job_charge = self.distance_to_charge(job_distance)
            if job_charge > job.charge:
                violations.append(
                    {
                    "job": job,
                    "job_charge": job.charge,
                    "travel_charge": job_charge,
                    "distance": job_distance,
                    }
                )
                raise Exception()
            
        if len(violations) > 0:
            return [v["job"] for v in violations]
        
        return violations
  
    def forward_label(self) -> list[Route]:
        """
        Sequences the solution routes using a forward labelling procedure.
        """
        label_heap: list[Label] = [] # min heap of Labels
        routes = []
        solution_fragment_ids = {(f, self.fragment_vars_by_id[f].x) for f in self.fragment_vars_by_id if self.fragment_vars_by_id[f].x > 0.5}
        # Sequence fragments by their start / end depots
        waiting_arcs = [
            (start, end, round(self.waiting_arcs[start, end].x)) 
            for start, end in self.waiting_arcs if self.waiting_arcs[start, end].x > 0.5
        ]
        
        # Get the first and last TimedDepot for each Depot 
        start_depots = {min(depot_list) for depot_list in self.timed_depots_by_depot.values()}
        end_depots = {max(depot_list) for depot_list in self.timed_depots_by_depot.values()}

        arcs_by_timed_depot: dict[TimedDepot, list[Arc]] = defaultdict(list[Arc])
        for f_id, f_flow in solution_fragment_ids:
            # Retrieve the timed depot it starts and ends at
            start_depot, end_depot = self.timed_depots_by_fragment_id[f_id].start, self.timed_depots_by_fragment_id[f_id].end
            arcs_by_timed_depot[start_depot].append(Arc(start_depot=start_depot, end_depot=end_depot, f_id=f_id, flow=f_flow))     
            if start_depot not in start_depots:
                continue
            # Decompress the network such that a min heap on time is real. 
            # To do this, we need to map these depots back to their true time.
            fragment = self.fragments_by_id[f_id]
            # start_depot = TimedDepot(time=fragment.start_time, id=fragment.start_depot_id)
            uncompressed_end_depot = TimedDepot(time=fragment.end_time, id=fragment.end_depot_id)
            
            start_label = Label(
                    flow=1, #TODO: THIS WONT WORK IN THE LP RELAXATION
                    prev_label=Label(
                        uncompressed_end_depot=uncompressed_end_depot, end_depot=start_depot, prev_label=None, flow=1, f_id=None
                    ),
                    end_depot=end_depot,
                    f_id=f_id, 
                    uncompressed_end_depot=uncompressed_end_depot
                )
            heapq.heappush(
                label_heap,
                start_label 
            )

        # Add any start labels
        for start_depot, end_depot, count in waiting_arcs:
            arcs_by_timed_depot[start_depot].append(
                Arc(start_depot=start_depot, end_depot=end_depot, f_id=None, flow=count)
            )     
            
            if start_depot not in start_depots:
                continue

            start_label = Label(uncompressed_end_depot=start_depot, flow=count, end_depot=start_depot, prev_label=None, f_id=None)
            heapq.heappush(
                label_heap, 
                Label(uncompressed_end_depot=start_depot, flow=count, prev_label=start_label, end_depot=end_depot, f_id=None)
            )

        while len(label_heap) != 0:
            label = heapq.heappop(label_heap)
            label_flow = label.flow
            arcs = arcs_by_timed_depot[label.end_depot]
            # Check if there are no arcs leaving this depot. If that's the case then this is an end point
            # TODO: confirm this 
            if len(arcs) == 0:
                if label.end_depot not in end_depots:
                    raise Exception("Route ends before the last depot time - very bad:(")
                # assemble the route backwards
                route = []
                while label.prev_label is not None:
                    route.append(label.end_depot)
                    if label.f_id is not None:
                        route.append(self.fragments_by_id[label.f_id])
                    label = label.prev_label
                route.append(label.end_depot)
                routes.append(Route(route_list=route[::-1]))

            for i, arc in enumerate(arcs):
                if arc.flow == 0: # TODO: Tolerance may be needed
                    continue
                delta = min(arc.flow, label_flow)
                uncompressed_end_depot = arc.end_depot

                if arc.f_id is not None:
                    fragment = self.fragments_by_id[arc.f_id]
                    uncompressed_end_depot = TimedDepot(time=fragment.end_time, id=fragment.end_depot_id)
                heapq.heappush(
                    label_heap,
                    Label(
                        uncompressed_end_depot=uncompressed_end_depot,
                        end_depot=arc.end_depot,
                        prev_label=label,
                        flow=delta,
                        f_id=arc.f_id
                    )
                )

                # Decrement flow on the arc and the label
                arc.flow -= delta
                label_flow -= delta
                
                if label_flow == 0:
                    break
        return routes                  

    def run(self) -> None:
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
        print("sequencing routes...")
        routes = self.create_routes()

        print("validating solution...")
        self.validate_solution(routes, self.model.objval, triangle_inequality=False)
        self.write_statistics()
