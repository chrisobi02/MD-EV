from collections import defaultdict
import math
import time as timer
from dataclasses import dataclass, field
import functools
import cProfile

from mdevs.formulations.base import *

@dataclass
class ValidationStore:
    new_fragment_arcs: set[ChargeFragment] = field(default_factory=set)
    depots_to_update: set[ChargeDepot] = field(default_factory=set)
    new_recharge_arcs: set[FrozenChargeDepotStore] = field(default_factory=set)
    jobs_to_update: set[Job] = field(default_factory=set)

    def update(self, other: 'ValidationStore'):
        self.new_fragment_arcs.update(other.new_fragment_arcs)
        self.depots_to_update.update(other.depots_to_update)
        self.new_recharge_arcs.update(other.new_recharge_arcs)
        self.jobs_to_update.update(other.jobs_to_update)

class NonLinearFragmentGenerator(BaseFragmentGenerator):
    def __init__(self, file: str, config: dict = {}):
        super().__init__(file, params=config)
        # % difference from an undiscretised maximum charge of 100%
        self.dilation_factor = 2 * self.config.UNDISCRETISED_MAX_CHARGE / 100 * self.config.PERCENTAGE_CHARGE_PER_UNIT 
        self.charge_fragments_by_charge_depot: dict[ChargeDepot, set[ChargeFragment]] = defaultdict(set[ChargeFragment])
        self.charge_depots_by_depot: dict[int, set[ChargeDepot]] = defaultdict(set)
        self.start_time_by_depot: dict[int, int] = {}
        self.end_times_by_depot: dict[int, int] = {}

    def get_charge(self, charge: int, recharge_time: int):
        """
        Determines the state of charge given a charge level and the amount of time it has to charge.
        """
        if recharge_time <= 0:
            return -1
        
        final_charge = self.get_charge_at_time(self.charge_inverse(charge) + recharge_time)
        return final_charge

    def get_charge_at_time(self, t: int) -> int:
        """returns the charge level from 0% when charging for t units of time as outlined in the paper."""
        if t <= 80:
            charge = 2 * t
        elif t <= 160:
            charge = 640/3 - ((12800 / 9) / (t - 160 / 3))
        else:
            charge = 200

        if charge * self.dilation_factor > self.config.MAX_CHARGE:
            raise ValueError(f"Charge level {charge} exceeds the maximum charge level")
        val = math.floor(charge * self.dilation_factor)
        return val
    
    def charge_inverse(self, charge: int):
        """Returns the time to charge to a certain level"""
        if charge <= 160 * self.dilation_factor:
            t = charge / (2 * self.dilation_factor)
        elif charge < 200 * self.dilation_factor:
            t = 160 * (charge * self.dilation_factor - 240) / (3 * charge - 640* self.dilation_factor)
        else:
            t = 160 / self.dilation_factor
        return math.ceil(t)

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
            if detour_charge > direct_charge:
                # Can reach job with a higher charge than if it were reached directly
                continue
            reachable_jobs.append(next_job)

        return reachable_jobs

    def generate_timed_network(self) -> None:
        time0 = timer.time()
        self.charge_depots_by_charge_fragment: dict[int, ChargeDepotStore] = defaultdict(ChargeDepotStore)
        for fragment in self.fragment_set:
            arrival_depot = ChargeDepot(id=fragment.start_depot_id, time=fragment.start_time, charge=self.config.MAX_CHARGE)
            end_depot = ChargeDepot(id=fragment.end_depot_id, time=fragment.end_time, charge=self.config.MAX_CHARGE)
            charge_fragment = ChargeFragment.from_fragment(start_charge=self.config.MAX_CHARGE, fragment=fragment)
            self.charge_fragments.add(charge_fragment)
            self.charge_fragments_by_charge_depot[arrival_depot].add(charge_fragment)
            self.charge_fragments_by_charge_depot[end_depot].add(charge_fragment)
            self.charge_depots_by_charge_fragment[charge_fragment] = ChargeDepotStore(start=arrival_depot, end=end_depot)
            self.charge_depots_by_depot[arrival_depot.id].add(arrival_depot)
            self.charge_depots_by_depot[end_depot.id].add(end_depot)
        
        # Create the recharge arcs
        for depot in self.charge_depots_by_depot:
            depots_in_time_order = sorted(self.charge_depots_by_depot[depot])
            adjacent_pairs = zip(depots_in_time_order, depots_in_time_order[1:])
            for start, end in adjacent_pairs:
                arc = FrozenChargeDepotStore(start=start, end=end)
                self.recharge_arcs_by_charge_depot[start].add(arc)
                self.recharge_arcs_by_charge_depot[end].add(arc)
    
        self.recharge_arcs = set.union(*[set(arcs) for arcs in self.recharge_arcs_by_charge_depot.values()])
        
        self.all_arcs_by_charge_depot: dict[ChargeDepot, set[FrozenChargeDepotStore | ChargeFragment]] = {
            charge_depot: self.recharge_arcs_by_charge_depot[charge_depot] | self.charge_fragments_by_charge_depot[charge_depot]
            for depot in self.charge_depots_by_depot for charge_depot in self.charge_depots_by_depot[depot]
        }
        self.start_times_by_depot = {
            depot: min(self.charge_depots_by_depot[depot], key = lambda x: x.time).time 
            for depot in self.charge_depots_by_depot 
        }
        self.end_times_by_depot = {
            depot: max(self.charge_depots_by_depot[depot], key = lambda x: x.time).time 
            for depot in self.charge_depots_by_depot 
        }
        
        self.statistics.update(
            {
                "timed_network_generation": timer.time() - time0,
                "timed_network_size": sum(len(v) for v in self.charge_fragments_by_charge_depot.values())
            }
        )

    def _get_valid_charges_for_fragment_id(self, f_id: int) -> range:
        """Returns all charges which can execute a given fragment id"""
        return range(self.fragments_by_id[f_id].charge, self.config.MAX_CHARGE + 1)

    def _get_flow_coefficient(self, depot: ChargeDepot, arc: Fragment | FrozenChargeDepotStore) -> int:
        """
        Retrieves the direction of flow between a depot and an input arc
        This is based on matching the fragment start/end time with the depot time
        """
        match arc:
            case Fragment():
                if depot.time == arc.start_time and depot.id == arc.start_depot_id:
                    return -1
                elif depot.time == arc.end_time and depot.id == arc.end_depot_id:
                    return 1
                
            case FrozenChargeDepotStore():
                if depot == arc.start:
                    return -1
                elif depot == arc.end:
                    return 1
            case _:
                raise ValueError("Fragment and depot do not have a common time / location")

    def _add_flow_balance_constraint(self, charge_depot: ChargeDepot) -> None:
        name = f"flow_{str(charge_depot)}"
        recharge_arcs = self.recharge_arcs_by_charge_depot[charge_depot]
        fragment_flow = quicksum(
            self._get_flow_coefficient(charge_depot, charge_fragment) * self.fragment_vars_by_charge_fragment[charge_fragment]
            for charge_fragment in self.charge_fragments_by_charge_depot[charge_depot]
        )
        recharge_flow = quicksum(
            self._get_flow_coefficient(charge_depot, recharge_arc) * self.recharge_arc_var_by_depot_store[recharge_arc]
            for recharge_arc in recharge_arcs
        )
        
        if charge_depot.time == self.start_times_by_depot[charge_depot.id]:
            constr = self.model.addConstr(
                self.starting_counts[charge_depot.id] + fragment_flow + recharge_flow == 0,
                name=name,
            )
        elif charge_depot.time == self.end_times_by_depot[charge_depot.id]:
            constr = self.model.addConstr(
                fragment_flow + recharge_flow == self.finishing_counts[charge_depot.id],
                name=name,
            )
        else:
            constr = self.model.addConstr(
                fragment_flow + recharge_flow == 0,
                name=name,
            )
        self.flow_balance[charge_depot] = constr

    def add_flow_balance_constraints(self):
        self.flow_balance = {}
        for depot in self.charge_depots_by_depot:
            for charge_depot in self.charge_depots_by_depot[depot]:
                self._add_flow_balance_constraint(charge_depot)

    def validate_route(self, route: list[ChargeDepot | Fragment]) -> tuple[bool, list[ChargeDepot | Fragment]]:
        """
        Validates a route to ensure it is feasible. 
        If it is not feasible, it returns the segment of the route which induced the infeasibility
        Otherwise it 
        For a route to be feasible, it must:
        - Start and end at a depot
        - Never go below 0 charge
        - Be able to feasibly reach each location in time
        """
        infractions = []
        curr_charge = self.config.MAX_CHARGE
        prev_location: ChargeDepot | Fragment = route[0]
        is_valid = True
        prev_time=prev_location.time
        for i, curr_location in enumerate(route[1:]):
            match prev_location, curr_location:
                case ChargeDepot(), ChargeDepot():
                    # recharge arc
                    if prev_location.id != curr_location.id:
                        raise Exception("Holding arcs should only be for the same depot")
                    curr_charge = self.get_charge(curr_charge, curr_location.time - prev_location.time)
                    prev_time = curr_location.time
                case ChargeDepot(), Fragment():
                    curr_charge, prev_time, is_valid = self.validate_fragment(curr_location, curr_charge, prev_time)
                case Fragment(), Fragment():
                    raise Exception("Cannot have two fragments in a row")
            if not is_valid:
                return False, route[:i+1] 
            prev_location = curr_location
        return True, route
     
    def get_validated_timed_solution(self, solution: list[list[int]], expected_vehicles: int = None) -> list[list[ChargeDepot | Fragment]]:
        charge_fragment_routes: list[list[ChargeFragment]] = [
            [ChargeFragment.from_fragment(200, self.fragments_by_id[f_id]) for f_id in route] for route in solution
        ]

        recharge_arcs = defaultdict(int)
        charge_routes= []
        
        for route in charge_fragment_routes:
            cf = route.pop(0)
            charge_route = []
            # if cf.start_time != self.start_times_+by_depot[cf.start_depot_id]:
            for cd1, cd2 in zip(
                sorted(self.charge_depots_by_depot[cf.start_depot_id], key=lambda x: x.time), 
                sorted(self.charge_depots_by_depot[cf.start_depot_id], key=lambda x: x.time)[1:]
            ):
                if cd1.time == cd2.time:
                    continue
                charge_route.append(cd1)
                if cd1.time == cf.start_time:
                    break
            # else:
            #     charge_route.append(cf.start_charge_depot)
            charge_route.append(cf)
            charge_route.append(ChargeDepot(id=cf.end_depot_id, time=cf.end_time, charge=200))

            prev_node = ChargeDepot(id=cf.end_depot_id, time=cf.end_time, charge=200)
            while len(route) != 0:
                cf = route.pop(0)
                if cf.start_depot_id != prev_node.id:
                    prev_node = ChargeDepot(id=cf.end_depot_id, time=cf.end_time, charge=200)
                    continue
                for cd1, cd2 in zip(
                    sorted(self.charge_depots_by_depot[cf.start_depot_id], key=lambda x: x.time), 
                    sorted(self.charge_depots_by_depot[cf.start_depot_id], key=lambda x: x.time)[1:]
                ):
                    if cd1.time <= prev_node.time:
                        continue
                    if cd1.time == cd2.time:
                        continue

                    charge_route.append(cd1)
                    if cd1.time == cf.start_time:
                        break
                charge_route.append(cf)
                charge_route.append(ChargeDepot(id=cf.end_depot_id, time=cf.end_time, charge=200))
                prev_node = ChargeDepot(id=cf.end_depot_id, time=cf.end_time, charge=200)
            
            for cd1 in sorted(self.charge_depots_by_depot[cf.end_depot_id], key=lambda x: x.time):
                if cd1.time <= prev_node.time:
                    continue
                charge_route.append(cd1)
            charge_routes.append(charge_route)

        for route in charge_routes:
            is_valid, _= self.validate_route(route)
            assert is_valid
        return charge_routes        

    def validate_job_sequences(self, routes: list[list[ChargeDepot | Fragment]]) -> bool:
        raise NotImplementedError()

    def inspect_solution(
            self,
            solution_charge_fragments: list[tuple[ChargeFragment, int]],
            waiting_arcs: list[tuple[FrozenChargeDepotStore, int]],
            store: ValidationStore=ValidationStore(),
        ) -> bool:
        routes = self.forward_label(solution_charge_fragments, waiting_arcs)
        has_violations = False
        nodes_to_update = set()
        for route in routes:
            is_valid, segment = self.validate_route(route.route_list)
            if not is_valid:
                has_violations = True
                # print(f"Invalid route: {segment}, repairing...")
                nodes_to_update.update(self.amend_violated_route(segment, store=store))
        
        assert store.depots_to_update == nodes_to_update, (store.depots_to_update.difference(nodes_to_update), nodes_to_update.difference(store.depots_to_update))
        # Update model to reflect new arcs and nodes.
        for node in store.depots_to_update:
            if (constr:=self.flow_balance.pop(node, None)) is not None:
                self.model.remove(constr)
            self._add_flow_balance_constraint(node)

        for job in store.jobs_to_update:
            if (constraint := self.coverage.pop(job, None)) is not None:
                self.model.remove(constraint)
            self._add_coverage_constraint(job)
        self.model.update()
        
        return not has_violations

    def _validate_charge_fragment_flow_balance(self):
        for cf in self.charge_fragments:
            # n_constr = 0
            nodes = self.charge_depots_by_charge_fragment[cf]
            for n in [nodes.start, nodes.end]:
                constr = self.flow_balance[n]
                # print(self.model.getRow(constr))
                for i in range(self.model.getRow(constr).size()):
                    # print(self.model.getRow(constr).getCoeff(i), self.model.getRow(constr).getVar(i).VarName)
                    var = self.model.getRow(constr).getVar(i)
                    if self.fragment_vars_by_charge_fragment[cf].VarName == var.VarName:
                        # n_constr += 1
                        break
                else:
                    raise ValueError(f"Fragment {cf.id} not found in flow balance constraint {constr}")

    def amend_violated_route(self, segment: list[ChargeDepot | Fragment], store=ValidationStore()) -> set[ChargeDepot]:
        """
        Takes a charge-infeasible segment of a route and modifies the network to reflect the true charge cost
        This is done in two steps:
        1. A new ChargeDepot is added to the network at the end of each fragment in the segment
        2. Any fragment with a greater charge cost at the end time is mapped to the new node
        3. The flow balance constraints are updated to reflect the new ChargeDepots
        4.1. If the new charge level is below the minimum charge of the fragments after the node's time
             then we add a recharge arc to reach that point
        4.2. If the new charge level is above that, then we add a new recharge arc which maps to the max level and next fragment.
        #TODO: Determine whether we check if we can recharge to execute something from the next node of fragments
        #      and if not then we repeat until we find the next existing node that can be executed and map it to there instead
        """
        curr_charge = self.config.MAX_CHARGE
        prev_location: ChargeDepot | Fragment = segment[0]
        curr_time = prev_location.time
        nodes_to_update: set[ChargeDepot] = set()
        for i, curr_location in enumerate(segment[1:]):
            match prev_location, curr_location:
                case ChargeDepot(), ChargeDepot():
                    curr_charge = self.get_charge(curr_charge, curr_location.time - prev_location.time)

                case Fragment(), ChargeDepot():
                    curr_charge -= prev_location.charge
            
            if isinstance(curr_location, ChargeDepot) and curr_charge < curr_location.charge:
                nodes_to_update.update(
                    self._add_new_depot(
                        ChargeDepot(id=curr_location.id, time=curr_location.time, charge=curr_charge),
                        store=store
                    )
                )
            prev_location = curr_location
        return nodes_to_update

    def _get_closest_charge_depot(self, charge_depot: ChargeDepot) -> ChargeDepot:
        """
        Gets the same or next highest charge depot at the given time and depot
        """
        charge_depots = self.charge_depots_by_depot[charge_depot.id]
        for level in range(charge_depot.charge, self.config.MAX_CHARGE + 1):
            if (test_depot := ChargeDepot(id=charge_depot.id, time=charge_depot.time, charge=level)) in charge_depots:
                return test_depot
        else:
            raise ValueError(f"No charge depot found for time: {test_depot.time}, charge: {test_depot.charge}")
    
    @functools.cache
    def _get_next_time_from_depot(self, id: int, time: int) -> int:
        """
        Gets the next earliest time an event occurs at the depot after the given charge depot's time
        """
        charge_depots = self.charge_depots_by_depot[id]
        return min((cd.time for cd in charge_depots if cd.time > time), default=None)
    
    def _add_new_depot(
            self, charge_depot: ChargeDepot, store=ValidationStore()
        ) -> set[ChargeDepot]:
        """
        Repairs a given node by adding a new node to the network at the real charge level
        This involves rectifying both fragment and charge arcs from the new node.
        """
        if charge_depot.charge > self.config.MAX_CHARGE:
            raise ValueError("Cannot add a depot with a charge higher than the maximum charge")
        nodes_to_update: set[ChargeDepot] = set()
        updated_fragments = self._update_fragment_arcs(charge_depot, store=store)
        updated_charge = self._update_charge_arcs(charge_depot, use_min_charge_condition=False, store=store)
        
        nodes_to_update.update(updated_fragments)
        nodes_to_update.update(updated_charge)
        store.depots_to_update.add(charge_depot)
        self.charge_depots_by_depot[charge_depot.id].add(charge_depot)
        
        for node in nodes_to_update:
            self.charge_depots_by_depot[node.id].add(node)
        return nodes_to_update

    def _update_fragment_arcs(self, new_depot: ChargeDepot, store=ValidationStore()) -> set[ChargeDepot]:
        """
        Updates the network to include the new depot. 
        Returns a set of nodes which need to have their flow balance constraints updated (or added).
        For inbound arcs (charge-fragments):
        - check their start levels (based on their charge state), see if the end aligns
        - if it is below or at new_depot, we remap their end nodes to this new node.
        - if it is negative, we remove that arc entirely.

        For outbound arcs:
        - Create a new charged copy leaving from new_depot, to the next closest correct state (rounded up)
        - If results in negative charge, don't make new copies
        """
        prev_depot = self._get_closest_charge_depot(new_depot)
        outbound_arcs: list[ChargeFragment] = []
        nodes_to_update: set[ChargeDepot] = {prev_depot, new_depot}
        invalid_inbound_arcs: list[ChargeFragment] = []
        if any(
            cf.id in [452, 448, 450, 446, 451, 444, 447, 445, 449]
            for cf in self.charge_fragments_by_charge_depot[prev_depot]
        ):
            pass
        for charge_fragment in self.charge_fragments_by_charge_depot[prev_depot]:
            # Inbound arc
            if (
                charge_fragment.end_time == new_depot.time 
                and charge_fragment.end_depot_id == new_depot.id 
                and charge_fragment.end_charge <= new_depot.charge
            ):
                invalid_inbound_arcs.append(charge_fragment)
            # Outbound arc
            if (
                charge_fragment.start_time == new_depot.time 
                and charge_fragment.start_depot_id == new_depot.id
                and new_depot.charge - charge_fragment.charge >= 0
            ):
                outbound_arcs.append(charge_fragment)
        
        """
        Dealing with remap arcs:
        - Update the start/end store they're attached to
        - Update the constraint at the OLD end node
        """
        for charge_fragment in invalid_inbound_arcs:
            self.charge_fragments_by_charge_depot[prev_depot].remove(charge_fragment)
            self.charge_fragments_by_charge_depot[new_depot].add(charge_fragment)
            self.charge_depots_by_charge_fragment[charge_fragment].end = new_depot

        """"
        Dealing with outbound arcs
        - create a new ChargeFragment with new_depot's charge level
        - add a new start/end store to this node
        """
        for charge_fragment in outbound_arcs:
            new_charge_fragment = ChargeFragment.from_fragment(
                start_charge=new_depot.charge, fragment=charge_fragment
            )
            end_depot = self._get_closest_charge_depot(
                ChargeDepot(
                    id=charge_fragment.end_depot_id,
                    time=new_charge_fragment.end_time,
                    charge=new_charge_fragment.end_charge
                )
            )
            self.charge_fragments.add(new_charge_fragment)
            self.charge_fragments_by_charge_depot[new_depot].add(new_charge_fragment)
            self.charge_fragments_by_charge_depot[end_depot].add(new_charge_fragment)
            self.charge_depots_by_charge_fragment[new_charge_fragment] = ChargeDepotStore(
                start=new_depot, end=end_depot
            )
            self.fragment_vars_by_charge_fragment[new_charge_fragment] = self.model.addVar(
                vtype=GRB.BINARY, name=f"f_{new_charge_fragment.id}_c_{new_charge_fragment.start_charge}"
            )
            nodes_to_update.add(end_depot)
            store.new_fragment_arcs.add(new_charge_fragment)
            for job in charge_fragment.jobs:
                store.jobs_to_update.add(job)
                self.charge_fragments_by_job[job].add(new_charge_fragment)

        store.depots_to_update.update(nodes_to_update)
        store.new_fragment_arcs.update(invalid_inbound_arcs)

        return nodes_to_update

    def _update_charge_arcs(
            self, new_depot: ChargeDepot, use_min_charge_condition: bool=False, store=ValidationStore()
        ) -> set[ChargeDepot]:
        """
        Updates the network to include charge arcs leaving the new_depot.
        Returns a set of nodes which need to have their flow balance constraints updated (or added).

        If above the minimum charge required to execute ANY fragment leaving this depot at a later time:
            - Map a recharge arc to the next node, rounded up.
        else:
            - Find the first time such that a fragment can feasibly executed with the recharge time required.
            - From that node, add another recharge arc to the next time (rounded up again?)
            e.g. current node is at 10 charge, a fragment needs 20 charge in 10 time units. 
                 If we can recharge to that 20 charge in that time, add the charge arc to that time 
                 (at whatever charge it results in), add a charge copy at that point?
                 otherwise we find the next possible time to do so and repeat until successful
        """
        nodes_to_update: set[ChargeDepot] = set()
        prev_depot = self._get_closest_charge_depot(new_depot)
        inbound_recharge_arcs: list[FrozenChargeDepotStore] = [
            ra for ra in self.recharge_arcs_by_charge_depot[prev_depot] if ra.end == prev_depot
        ]

        for recharge_arc in inbound_recharge_arcs:
            """
            Updating old recharge arcs: when we find a previous recharge arc should actually map to the new node over the old.
            1. Update the node store:
                - Delete the old node store and its associated variable (old_start, old_end)
                - Add a new node store and its associated variable from (old_start, new_end)
            2. Update the flow balance constraint:
                - add ALL nodes that have been messed with (old_start, old_end, new_end) to the set of nodes to update
            """
        # Update inbound recharge arcs that should go here instead of a higher charge node
            if recharge_arc.start.charge > (recharge_val:= self.get_charge(recharge_arc.start.charge, new_depot.time - recharge_arc.start.time)):
                recharge_val = self.get_charge(recharge_arc.start.charge, new_depot.time - recharge_arc.start.time)
            if self.get_charge(recharge_arc.start.charge, new_depot.time - recharge_arc.start.time) <= new_depot.charge:
                # Updating lookups
                if recharge_arc.start.charge > new_depot.charge:
                    pass
                new_recharge_arc = FrozenChargeDepotStore(start=recharge_arc.start, end=new_depot)
                self._remove_recharge_arc_variable(recharge_arc)
                self._add_recharge_arc_variable(new_recharge_arc)
                store.new_recharge_arcs.add(new_recharge_arc)
                nodes_to_update.update({recharge_arc.start, recharge_arc.end, new_depot})
        store.depots_to_update.update(nodes_to_update)
        end_time = self._get_next_time_from_depot(new_depot.id, new_depot.time)
        # This is the last time node. We don't need to add any more charge arcs
        if end_time is None:
            return nodes_to_update
        
        """
        Adding new recharge arcs leaving the new node:
        If using the minimum charge condition:
            - get the end charge for the next time something happens. 
            - If it is below the minimum charge required to execute any fragment at that time, 
                add a new node at that point
        Else:
            - Get the closest node to the true charge level at the next time and add an arc there.
        """
        end_charge = self.get_charge(new_depot.charge, end_time - new_depot.time)
        has_insufficient_charge = False
        if end_charge > self.config.MAX_CHARGE:
            raise ValueError("Cannot add a depot with a charge higher than the maximum charge")
        if use_min_charge_condition:
            # Find the minimum charge required to execute any fragment at this time
            min_charge = self._get_min_fragment_charge_from_depot(new_depot.id, end_time)
            # If we can't execute any fragment at that time, need to add a new node at that point.
            if end_charge <= min_charge:        
                has_insufficient_charge = True
        
        closest_recharge_depot = self._get_closest_charge_depot(
            ChargeDepot(id=new_depot.id, time=end_time, charge=end_charge)
        )
        if new_depot.charge > closest_recharge_depot.charge or end_charge < new_depot.charge:
                pass
    
        recharge_arc = FrozenChargeDepotStore(start=new_depot, end=closest_recharge_depot)
        self._add_recharge_arc_variable(recharge_arc)
        store.new_recharge_arcs.add(recharge_arc)
        nodes_to_update.update({new_depot, closest_recharge_depot})
    
        if has_insufficient_charge:
            end_depot = ChargeDepot(id=new_depot.id, time=end_time, charge=end_charge)
            nodes_to_update.update(self._add_new_depot(end_depot, store=store))
        
        store.depots_to_update.update(nodes_to_update)
        return nodes_to_update

    def _get_min_fragment_charge_from_depot(self, depot_id: int, end_time: int) -> int:
        if self.minimum_charge_by_time_by_depot.get((depot_id, end_time)) is not None:
            return self.minimum_charge_by_time_by_depot[depot_id, end_time]
        min_charge = min(
                (
                    fragment.charge 
                    for fragment in self.charge_fragments_by_charge_depot[
                        ChargeDepot(id=depot_id, time=end_time, charge=self.config.MAX_CHARGE)
                    ]
                ),
                default=-1
            )
        self.minimum_charge_by_time_by_depot[depot_id, end_time] = min_charge
        return min_charge
    
    def _remove_recharge_arc_variable(self, recharge_arc: FrozenChargeDepotStore):
        self.recharge_arcs_by_charge_depot[recharge_arc.start].remove(recharge_arc)
        self.recharge_arcs_by_charge_depot[recharge_arc.end].remove(recharge_arc)
        self.recharge_arcs.remove(recharge_arc)
        self.model.remove(self.recharge_arc_var_by_depot_store[recharge_arc])
        del self.recharge_arc_var_by_depot_store[recharge_arc]

    def _remove_charge_fragment_variable(self, charge_fragment: ChargeFragment):
        self.model.remove(self.fragment_vars_by_charge_fragment[charge_fragment])
        del self.fragment_vars_by_charge_fragment[charge_fragment]
        store = self.charge_depots_by_charge_fragment.pop(charge_fragment)
        for job in charge_fragment.jobs:
            self.charge_fragments_by_job[job].remove(charge_fragment)
        self.charge_fragments_by_charge_depot[store.start].remove(charge_fragment)
        self.charge_fragments_by_charge_depot[store.end].remove(charge_fragment)
        self.charge_fragments.remove(charge_fragment)


    def _add_recharge_arc_variable(self, recharge_arc: FrozenChargeDepotStore) -> None:
        self.recharge_arcs_by_charge_depot[recharge_arc.start].add(recharge_arc)
        self.recharge_arcs_by_charge_depot[recharge_arc.end].add(recharge_arc)
        self.recharge_arcs.add(recharge_arc)
        self.recharge_arc_var_by_depot_store[recharge_arc] = self.model.addVar(
            vtype=GRB.INTEGER, name=f"w_{recharge_arc.start}_{recharge_arc.end}"
        )

    def set_solution(self) -> set[ChargeFragment]:
        paper_solution, _ = self.convert_solution_to_fragments(instance_type="regular", charging_style='non-linear')
        charge_routes = self.get_validated_timed_solution(paper_solution)
        used_fragments = {cf.id for route in charge_routes for cf in route if isinstance(cf, ChargeFragment)}
        removes = set()
        for cf in self.fragment_vars_by_charge_fragment:
            if cf.id not in used_fragments:
                removes.add(cf)
        for cf in removes:
            self._remove_charge_fragment_variable(cf)
        self.model.addConstr(quicksum(self.starting_counts.values()) == len(paper_solution))

        return {cf for cf in self.charge_fragments if cf not in removes}

    def solve(self):
        # self.model.setParam("OutputFlag", 0)
        self.model.update()
        valid_relaxation = False
        var_type = GRB.CONTINUOUS #
        first_time = False
        also_first_time = True
        # sol_fragments = self.set_solution()
        self._validate_waiting_arcs()
        n_iters = 0
        prev_obj = 0
        time_dict: dict[str, float] = {
            "solve_time": 0,
            "inspection_time": 0
        }
        while not valid_relaxation and n_iters < 10:
            n_iters += 1
            print(f"    Starting iteration {n_iters}")
            self.change_variable_domain(var_type)
            self.model.setParam("Seed", 0)
            self.model.optimize()
            time_dict["solve_time"] += self.model.Runtime
            from mdevs.utils.visualiser import visualise_charge_network
            if self.model.status == GRB.Status.OPTIMAL:
                solution_charge_fragments: set[tuple[ChargeFragment, int]] = {
                    (cf, self.fragment_vars_by_charge_fragment[cf].x) 
                    for cf in self.fragment_vars_by_charge_fragment if getattr(self.fragment_vars_by_charge_fragment[cf], 'x', -1) > EPS
                }
                if abs(self.model.objval - prev_obj) > EPS:
                    prev_obj = self.model.objval
                    print(self.model.objval)
                    
                # Sequence fragments by their start / end depots
                waiting_arcs = [
                    (arc.start, arc.end, self.recharge_arc_var_by_depot_store[arc].x) 
                    for arc in self.recharge_arc_var_by_depot_store if self.recharge_arc_var_by_depot_store[arc].x > EPS
                ]
                # waiting_stores = [store for store in self.recharge_arc_var_by_depot_store if self.recharge_arc_var_by_depot_store[store].x > EPS]
                # visualise_charge_network(
                #     [cd for d in self.charge_depots_by_depot for cd in self.charge_depots_by_depot[d]],
                #     [
                #         (
                #           self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
                #         )
                #         for cf, _ in solution_charge_fragments
                #     ], 
                #     [arc for arc in self.recharge_arc_var_by_depot_store if getattr(self.recharge_arc_var_by_depot_store[arc], "x", -1) > EPS],
                #    [],[],[]
                # ).show()
                network_additions = ValidationStore()
                print("     Inspecting solution...")
                time0 = timer.time()
                valid_relaxation = self.inspect_solution(
                    solution_charge_fragments, waiting_arcs, store=network_additions
                )
                inspection_time = timer.time() - time0
                time_dict["inspection_time"] += inspection_time
                print(f"     Inspection took {inspection_time}, Solve took {self.model.Runtime}")
                # new_fragment_stores = [self.charge_depots_by_charge_fragment[cf] for cf in network_additions.new_fragment_arcs]
                self.model.update()
                if valid_relaxation and also_first_time:
                    print("Valid linear relaxation, changing to integer variables...")
                    var_type = GRB.BINARY
                    also_first_time = False
                    valid_relaxation = False
                prev_additions = network_additions
                continue
            else:
            #     visualise_charge_network(
            #         [cd for d in self.charge_depots_by_depot for cd in self.charge_depots_by_depot[d]],
            #         [
            #             (
            #               self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
            #             )
            #             for cf in sol_fragments
            #         ], [],[],[], []
                    # waiting_stores,
                    # prev_additions.new_depots,
                    # [(self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end) for cf in prev_additions.new_fragment_arcs],
                    # prev_additions.new_recharge_arcs.intersection(self.recharge_arcs),
                # ).show()
                visualise_charge_network(
                    [cd for d in self.charge_depots_by_depot for cd in self.charge_depots_by_depot[d]],
                    [
                        (
                          self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
                        )
                        for cf, _ in solution_charge_fragments
                    ], 
                    self.recharge_arcs.difference(prev_additions.new_recharge_arcs),
                    # [arc for arc in self.recharge_arc_var_by_depot_store if getattr(self.recharge_arc_var_by_depot_store[arc], "x", -1) > EPS],
                    prev_additions.depots_to_update,
                    [(self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end) for cf in prev_additions.new_fragment_arcs],
                    prev_additions.new_recharge_arcs.intersection(self.recharge_arcs),
                ).show()#write_html("infeasible_step.html")

                # self.forward_label(solution_charge_fragments, waiting_arcs)
                # if first_time:
                #     self.set_solution()
                #     first_time=False
                # else:
                self.model.computeIIS()
                self.model.write("fragment_network.ilp")
                raise Exception("Relaxation is infeasible")
        # TODO: enumerate the theoretical network size for a-priori approach
        # assert self.model.objval == self._get_paper_objective_value()
        visualise_charge_network(
            [cd for d in self.charge_depots_by_depot for cd in self.charge_depots_by_depot[d]],
            [
                (
                    self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
                )
                for cf, _ in solution_charge_fragments
            ], 
            [arc for arc in self.recharge_arc_var_by_depot_store if getattr(self.recharge_arc_var_by_depot_store[arc], "x", -1) > EPS],
            network_additions.depots_to_update,
            [(self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end) for cf in network_additions.new_fragment_arcs],
            network_additions.new_recharge_arcs.intersection(self.recharge_arcs),
        ).write_html(f"images/{self.data['label']}_solution.html")
        print(f"solved in {n_iters} iterations")
        print(f"{time_dict=}")

    def change_variable_domain(self, var_type: str):
        """Sets all variable vtypes to the given type"""
        for var in self.fragment_vars_by_charge_fragment.values():
            var.vType = var_type

        if var_type == GRB.BINARY:
            var_type = GRB.INTEGER
        for var in self.recharge_arc_var_by_depot_store.values():
            var.vType = var_type
        for var in self.starting_counts.values():
            var.vType = var_type
        for var in self.finishing_counts.values():
            var.vType = var_type

    def validate_timed_network(self):
        """
        Validates the timed network to ensure it is feasible.
        For a timed network to be feasible:
        - No timed depot can have fragments which start after its time
        - Each timed depot cannot have a time earlier than the previous timed depot
        """
        num_recharge_in_per_charge_depot: dict[ChargeDepot] = defaultdict(int)
        num_recharge_out_per_charge_depot: dict[ChargeDepot] = defaultdict(int)
        for depot_id, charge_depots in self.charge_depots_by_depot.items():
            prev_td = None
            for td in sorted(charge_depots):
                if prev_td:
                    assert td.time >= prev_td.time
                    if td.time <= prev_td.time:
                        print(f"Depot {td} has a time earlier than the previous depot.")
                        return False
                fragments = self.charge_fragments_by_charge_depot[td]
                # Check the following:
                # min end time >= max start time
                # no start time later than the current depot time and no earlier than the previous time.
                departure_fragments = [f for f in fragments if ChargeDepot(charge=f.start_charge, time=f.start_time, id=f.start_depot_id) == td]
                arrival_fragments = [f for f in fragments if ChargeDepot(charge=f.start_charge, time=f.start_time, id=f.start_depot_id) != td]
                if len(arrival_fragments) != 0:
                    assert all(prev_td.time <= tf.start_time <= td.time for tf in departure_fragments)
                if len(departure_fragments) != 0:
                    assert all(prev_td.time <= tf.end_time <= td.time for tf in arrival_fragments)
                if len(departure_fragments) != 0 and len(arrival_fragments) != 0:
                   max_arr = max(tf.end_time for tf in arrival_fragments) 
                   min_dep = min(tf.start_time for tf in departure_fragments)
                   assert max_arr <= min_dep
                prev_td=td
        self._validate_waiting_arcs()

    def _validate_waiting_arcs(self) -> bool:
        """Each waiting arc should be connected to one at the next time. They should span from start to end"""
        for depot in self.charge_depots_by_depot:
            charge_depot = min(self.charge_depots_by_depot[depot], key=lambda x: x.time)
            assert charge_depot.time == self.start_times_by_depot[charge_depot.id]
            has_neighbour = True
            while has_neighbour:
                has_neighbour = False
                for arc in self.recharge_arcs_by_charge_depot[charge_depot]:
                    if arc.start != charge_depot:
                        continue
                    if arc.end is None:
                        assert charge_depot.time == self.end_times_by_depot[charge_depot.id]
                        continue
                    assert arc.end.time == self._get_next_time_from_depot(charge_depot.id, charge_depot.time)
                    assert arc.end.charge >= arc.start.charge
                    charge_depot = arc.end
                    has_neighbour = True

    def visualise_solution(
            self,
            sol_routes: list[list[ChargeDepot | Fragment]],
            corrected_routes: list[list[ChargeDepot | Fragment]],
        ) -> None:
        """Visualises a solution to the linaer relaxation."""

    def _get_paper_objective_value(self) -> float:

        data = pd.read_excel(r"mdevs/data/mdevs_solutions.xlsx", sheet_name=None)
        instance_type = "regular" if "000" not in self.data["label"] else 'large' 
        for sheet in data:
            if instance_type in sheet:
                data = data[sheet]
                break

        return data.query(
            f"ID_instance == {self.data['ID']} and battery_charging == 'non-linear'"
        )["objective_value"].values[0]


    def run(self):
        """Runs an end-to-end solve ."""
        print(f"Solving {self.data['label']}...")
        print("generating fragments...")
        self.generate_fragments()
        print("generating timed network...")
        self.generate_timed_network()
        print("validating timed network...")
        self.validate_timed_network()
        # from mdevs.utils.visualiser import visualise_charge_network
        # visualise_charge_network(
        #     [cd for d in self.charge_depots_by_depot for cd in self.charge_depots_by_depot[d]],
        #     [
        #         (
        #             ChargeDepot(id=cf.start_depot_id, time=cf.start_time, charge=cf.start_charge),
        #             ChargeDepot(id=cf.end_depot_id, time=cf.end_time, charge=cf.end_charge)
        #         )
        #         for cf in self.charge_fragments
        #     ], 
        #     self.recharge_arcs, 
        # ).show()
        print("building model...")
        self.build_model()
        print("solving...")
        with cProfile.Profile() as profile:
            self.solve()
            profile.create_stats()
            profile.dump_stats("solve_profile_large_instance.prof")
        # print("sequencing routes...")
        # routes = self.create_routes()

        # print("validating solution...")
        # self.validate_solution(routes, self.model.objval, triangle_inequality=False)
        self.write_statistics()
