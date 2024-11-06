from collections import defaultdict
import math
import time as timer
from dataclasses import dataclass, field
import functools
import numpy as np
from typing import TypedDict, Any
from enum import Enum
from plotly.graph_objects import Figure
import cProfile
import bisect
from mdevs.formulations.base import CalculationConfig
from mdevs.utils.visualiser import visualise_charge_network, visualise_network_transformation, animate_network_transformation, write_network_transformation
from mdevs.formulations.base import *
from mdevs.formulations.charge_functions import PaperChargeFunction
from gurobipy import Constr

class Objective(Enum):
    """
    Enum for the different second objectives that can be used in the DDD process
    """
    MIN_VEHICLES = "min_vehicles"
    MIN_CHARGE = "min_charge"
    MIN_DEADHEADING = "min_deadheading"
    MAX_DEADHEADING = "max_deadheading"

@dataclass
class SolveConfig:
    """Config for different aspects of the DDD process"""
    MAX_ITERATIONS: int = 100000
    TIME_LIMIT: int = 900
    MIP_GAP: float = 0.05
    SECOND_OBJECTIVE: Objective = Objective.MIN_VEHICLES
    SOLVE_SECOND_OBJECTIVE_OPTIMAL: bool = False # Decides whether to use the secondary objective to guide the optimisation process or solve it exactly
    INCLUDE_VISUALISATION: bool = False
    INCLUDE_MODEL_OUTPUT: bool = False

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
    TYPE="non-linear"
    def __init__(
            self, 
            file: str,
            config=CalculationConfig(),
            solve_config=SolveConfig(),
            charge_calculator_class: ChargeCalculator=PaperChargeFunction,
            charge_calculator_kwargs: dict[str, Any]={},
        ):
        """
        Creates a new instance of the NonLinearFragmentGenerator.
        Note: the charge_calculator_class must be the class itself, not an instance of the class.
        This is so the calculation config can be passed in during initialisation of the class instead of passing it in at two points.
        charge_calculator_kwargs are any additional arguments to be passed to the charge_calculator_class specific to an implementation.
        """
        super().__init__(file, charge_calculator_class, config=config, charge_calculator_kwargs=charge_calculator_kwargs)
        self.solve_config = solve_config
        self.charge_fragments_by_charge_depot: dict[ChargeDepot, set[ChargeFragment]] = defaultdict(set[ChargeFragment])
        self.charge_fragments_by_id: dict[int, set[ChargeFragment]] = defaultdict(set[ChargeFragment])
        self.charge_depots_by_depot: dict[int, set[ChargeDepot]] = defaultdict(set)
        self.start_time_by_depot: dict[int, int] = {}
        self.end_times_by_depot: dict[int, int] = {}
        self.next_time_after_depot_time_cache: dict[tuple[int, int], int] = {}
        self.deadheading_time_by_fragment_cache: dict[int, int] = {}
        self.statistics = NonLinearStatistics(**self.statistics)
        self.curr_vtype: str = GRB.BINARY # GRB.CONTINUOUS or GRB.BINARY
        self.vehicle_count: int = None
        self.valid_routes: list[Route] | None = None
        self.mip_solution: list[Route] | None = None
        self.ddd_traces: list = []

    def generate_timed_network(self) -> None:
        """
        Generates the initial network to be solved on.
        This ignores any charge based requirements in the network.
        One exception is that the end node of charge fragments is set to be the first time it reaches the end_depot
        and has enough charge to execute another fragment from that current ChargeDepot
        e.g. if a fragment ends at time 10 with 20 charge, and the smallest charge fragment requires 30 charge at time 10
        then it cannot execute anything from that depot and must recharge.
        Instead, we determine the first time after arriving that it has enough charge to feasibly execute a fragment.
        For the above example this may imply charging for 5 time units until it can execute the new fragment.

        """
        time0 = timer.time()
        self.charge_depots_by_charge_fragment: dict[int, ChargeDepotStore] = defaultdict(ChargeDepotStore)
        # Begin by finding all times something could happen
        # TODO: add a check which will exclude any times where no job can be reached, i.e. when the min-charge condition is required.
        arrival_times_by_depot: dict[int, set[int]] = defaultdict(set)
        departure_times_by_depot: dict[int, set[int]] = defaultdict(set)
        print("getting end/start times")
        time_id_pairs: set[tuple[int, int]] = set()
        for f in self.fragment_set:
            time_id_pairs.add((f.start_depot_id, f.start_time,))
            time_id_pairs.add((f.end_depot_id, f.end_time))
            departure_times_by_depot[f.start_depot_id].add(f.start_time)
            arrival_times_by_depot[f.end_depot_id].add(f.end_time)

        self.departure_times_by_depot = {
            id: sorted(times) for id, times in departure_times_by_depot.items()
        }
        self.arrival_times_by_depot = {
            id: sorted(times) for id, times in arrival_times_by_depot.items()
        }

        for id, time in time_id_pairs:
            self.charge_depots_by_depot[id].add(ChargeDepot(id=id, time=time, charge=self.config.MAX_CHARGE))

        self.start_times_by_depot = {
            depot: min(self.charge_depots_by_depot[depot], key = lambda x: x.time).time 
            for depot in self.charge_depots_by_depot 
        }
        self.end_times_by_depot = {
            depot: max(self.charge_depots_by_depot[depot], key = lambda x: x.time).time 
            for depot in self.charge_depots_by_depot 
        }
        print('creating charge fragments')
        # Get all        
        for fragment in self.fragment_set:
            charge_fragment = ChargeFragment.from_fragment(start_charge=self.config.MAX_CHARGE, fragment=fragment)
            self.charge_fragments_by_id[fragment.id].add(charge_fragment)
            arrival_depot = charge_fragment.start_charge_depot
            # Baking in any required recharge to be useful
            first_depot_which_can_leave = self._get_first_depot_vehicle_can_leave_from(charge_fragment.end_charge_depot)
            # Remaps to the maximum charge
            end_depot = ChargeDepot(
                id=charge_fragment.end_depot_id, time=first_depot_which_can_leave.time, charge=self.config.MAX_CHARGE
            )
            self.charge_fragments.add(charge_fragment)
            self.charge_fragments_by_charge_depot[arrival_depot].add(charge_fragment)
            self.charge_fragments_by_charge_depot[end_depot].add(charge_fragment)
            self.charge_depots_by_charge_fragment[charge_fragment] = ChargeDepotStore(start=arrival_depot, end=end_depot)
        print("creating charge arcs")
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
        
        self.statistics.update(
            {
                "initial_charge_network_generation": timer.time() - time0,
                "initial_charge_network_arcs": len(self.charge_fragments) + len(self.recharge_arcs),
                "initial_charge_network_nodes": len(self.charge_depots)
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
                elif depot.time >= arc.end_time and depot.id == arc.end_depot_id:
                    return 1
                
            case FrozenChargeDepotStore():
                if depot == arc.start:
                    return -1
                elif depot == arc.end:
                    return 1
            case _:
                raise ValueError("Fragment and depot do not have a common time / location")

    def _add_flow_balance_constraint(self, charge_depot: ChargeDepot) -> None:
        if charge_depot.time == self.end_times_by_depot[charge_depot.id] and charge_depot.charge != self.config.MAX_CHARGE:
            raise Exception()
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

    def validate_route(self, route: list[ChargeDepot | Fragment], start_charge: int | None=None) -> tuple[bool, list[ChargeDepot | Fragment]]:
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
        curr_charge = start_charge if start_charge is not None else self.config.MAX_CHARGE
        prev_location: ChargeDepot | Fragment = route[0]
        is_valid = True
        prev_time=prev_location.time
        for i, curr_location in enumerate(route[1:]):
            # If only charge depots remain, then the route is valid
            # if all(isinstance(loc, ChargeDepot) for loc in route[i+1:]):
            #     break
            match prev_location, curr_location:
                case ChargeDepot(), ChargeDepot():
                    # recharge arc
                    if prev_location.id != curr_location.id:
                        raise Exception("Holding arcs should only be for the same depot")
                    curr_charge = self.get_charge(curr_charge, curr_location.time - prev_location.time)
                    prev_time = curr_location.time

                case Fragment(), ChargeDepot():
                    curr_charge, prev_time, is_valid = self.validate_fragment(prev_location, curr_charge, prev_time)
                    # Check if there is any time between the two locations
                    if curr_charge is not None:
                        curr_charge = self.get_charge(curr_charge, curr_location.time - prev_location.end_time)

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
    
    def _check_route_for_invalid_pair(self, route: list[ChargeDepot | Fragment]) -> bool:
        fragments = [loc for loc in route if isinstance(loc, Fragment)]
        if len(fragments) != 2:
            return
        start = fragments[0]
        end = fragments[1]
        start_charge = self.config.MAX_CHARGE - start.charge
        recharge_time = end.start_time - start.end_time
        self.statistics['2_frag_routes'] = self.statistics.get('2_frag_routes', 0) + 1
        if self.get_charge(start_charge, recharge_time) - end.charge < 0:
            self.statistics["inf_2_frags"] = self.statistics.get("inf_2_frags", 0) + 1

    def inspect_solution(
            self,
            solution_charge_fragments: list[tuple[ChargeFragment, int]],
            waiting_arcs: list[tuple[FrozenChargeDepotStore, int]],
            store: ValidationStore=ValidationStore(),
        ) -> bool:
        routes, has_rounding_error = self.forward_label(solution_charge_fragments, waiting_arcs)
        if has_rounding_error:
            print("Rounding error found in labelling solution")
        has_violations = False
        nodes_to_update = set()
        n_added = 0
        for route in routes:
            # self._check_route_for_invalid_pair(route.route_list)
            for i, (curr, next) in enumerate(zip(route.route_list, route.route_list[1:])):
                if not (isinstance(curr, ChargeDepot) and isinstance(next, Fragment)):
                    continue
                route_segment = route.route_list[i:]
                is_valid, segment = self.validate_route(route_segment, start_charge=curr.charge)
                if is_valid:
                    break
                # try:
                nodes_to_update.update(self.amend_violated_route(segment, store=store))       
                # except:
                #     nodes_to_update.update(self.amend_violated_route(segment, store=store))       
                self.statistics["infeasible_route_segments"] += 1
                has_violations = True
                n_added += 1
        if n_added > 0:
            print(f"    violations: {n_added=}")
        
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
        
        return not has_violations or has_rounding_error

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
        prev_location: ChargeDepot | Fragment = segment[0]
        curr_charge = prev_location.charge
        nodes_to_update: set[ChargeDepot] = set()
        prev_charge = curr_charge
        new_depot = None
        for i, curr_location in enumerate(segment[1:]):
            # For cases where a fragment was below the minimum charge at a node, so we need to 
            if (
                new_depot is not None 
                and isinstance(curr_location, ChargeDepot)
                and curr_location.time < new_depot.time
            ):
                prev_location = new_depot
                curr_charge = new_depot.charge
                continue
                
            match prev_location, curr_location:
                case ChargeDepot(), ChargeDepot():
                    curr_charge = self.get_charge(curr_charge, curr_location.time - prev_location.time)

                case Fragment(), ChargeDepot():
                    if prev_location.end_time > curr_location.time:
                        raise ValueError()
                    curr_charge = self.get_charge(
                        curr_charge - prev_location.charge, curr_location.time - prev_location.end_time
                    )
            if curr_charge < 0:
                raise ValueError(f"Charge level {curr_charge} is below 0")
           
            if isinstance(curr_location, ChargeDepot) and curr_charge < curr_location.charge:
                if curr_location.time == self.end_times_by_depot[curr_location.id]:
                    continue
                new_depot = ChargeDepot(id=curr_location.id, time=curr_location.time, charge=curr_charge)
                # After a fragment, if it is below the minimum charge, its destination depot isn't 
                # necessarily the next one in the current route (since it assumed it has a higher charge and can therefore do more)
                # Ensure we add that new depot as well.
                if (
                    curr_charge < self._get_min_fragment_charge_from_depot(curr_location.id, curr_location.time)
                    and isinstance(prev_location, Fragment)
                ):
                    new_depot = self._get_first_depot_vehicle_can_leave_from(new_depot)

                if new_depot in self.charge_depots_by_depot[new_depot.id]:
                    prev_location = curr_location
                    continue

                if new_depot == ChargeDepot(id=1, time=234, charge=39):
                    pass
                nodes_to_update.update(
                    self._add_new_depot(new_depot, store=store)
                )
            prev_charge = curr_charge
            prev_location = curr_location
        return nodes_to_update

    def _get_closest_charge_depot(self, charge_depot: ChargeDepot) -> ChargeDepot:
        """
        Gets the same or next highest charge depot at the given time and depot
        """
        #TODO: keep track of real charge levels by depot same as get_next_departure_time_from_depoth
        charge_depots = self.charge_depots_by_depot[charge_depot.id]
        for level in range(charge_depot.charge, self.config.MAX_CHARGE + 1):
            if (test_depot := ChargeDepot(id=charge_depot.id, time=charge_depot.time, charge=level)) in charge_depots:
                return test_depot
        else:
            raise ValueError(f"No charge depot found for time: {test_depot.time}, charge: {test_depot.charge}")
    
    def _get_first_depot_vehicle_can_leave_from(self, charge_depot: ChargeDepot) -> ChargeDepot:
        """
        Returns the first charge depot with enough charge to execute a fragment
        """
        start_time = new_time = charge_depot.time
        start_charge = charge_depot.charge
        curr_charge = start_charge
        min_charge = self._get_min_fragment_charge_from_depot(charge_depot.id, charge_depot.time)
        max_iters = len(self.departure_times_by_depot[charge_depot.id])
        iters = 0
        while (
            (curr_charge < min_charge or min_charge == -1)
            and iters < max_iters
        ):
            if curr_charge < 0:
                raise ValueError(f"Charge depot {charge_depot} has negative charge")
            new_time = self._get_next_departure_time_from_depot(charge_depot.id, new_time)
            if new_time is None:
                charge_depot = ChargeDepot(
                    id=charge_depot.id, time=self.end_times_by_depot[charge_depot.id], charge=self.config.MAX_CHARGE
                )
                return charge_depot
            curr_charge = self.get_charge(start_charge, new_time - start_time)
            min_charge = self._get_min_fragment_charge_from_depot(charge_depot.id, new_time)
            iters+=1
        if iters == max_iters:
            raise ValueError(f"Could not find a depot with enough charge to execute a fragment from {charge_depot}")
        destination_depot = ChargeDepot(id=charge_depot.id, time=new_time, charge=curr_charge)
        return destination_depot

    def _get_next_departure_time_from_depot(self, id: int, time: int) -> int | None:
        """
        Gets the next earliest time an event occurs at the depot after the given charge depot's time
        Returns None if there is not another time.
        """
        if (id, time) in self.next_time_after_depot_time_cache:
            return self.next_time_after_depot_time_cache[id, time]

        # charge_depots = [cd.time for cd in self.charge_depots_by_depot[id] if cd.time > time]
        # next_time = min(charge_depots, default=None)
        # self.next_time_after_depot_time_cache[id, time] = next_time
        # return next_time

        times = self.departure_times_by_depot[id]
        idx = bisect.bisect_right(times, time)
        
        if idx == len(times):
            next_time = None
        else:
            next_time = times[idx]
       
        self.next_time_after_depot_time_cache[id, time] = next_time
        return next_time
    
    def _add_new_depot(
            self, charge_depot: ChargeDepot, origin_fragment: ChargeFragment | None=None, store=ValidationStore()
        ) -> set[ChargeDepot]:
        """
        Repairs a given node by adding a new node to the network at the real charge level
        This involves rectifying both fragment and charge arcs from the new node.
        """
        nodes_to_update: set[ChargeDepot] = set()
        # if charge_depot in self.charge_depots_by_depot[charge_depot.id]:
        #     return nodes_to_update
        updated_fragments = self._update_fragment_arcs(charge_depot, origin_fragment=origin_fragment, store=store)
        updated_charge = self._update_charge_arcs(charge_depot, store=store)
        
        nodes_to_update.update(updated_fragments)
        nodes_to_update.update(updated_charge)
        store.depots_to_update.add(charge_depot)
        self.charge_depots_by_depot[charge_depot.id].add(charge_depot)
        
        for node in nodes_to_update:
            self.charge_depots_by_depot[node.id].add(node)
        return nodes_to_update

    def _update_fragment_arcs(
            self, new_depot: ChargeDepot, origin_fragment: ChargeFragment | None=None, store=ValidationStore()
        ) -> set[ChargeDepot]:
        """
        Updates the network to include the new depot. 
        Returns a set of nodes which need to have their flow balance constraints updated (or added).
        The parameter origin_fragment allows one to specify the fragment which caused this addition.
            This is needed to change the end node of the fragment to the new depot, 
            since in the case where it is below the minimum charge at the desitnation node, 
            it needs to be rerouted to the first once after.
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

        for charge_fragment in self.charge_fragments_by_charge_depot[prev_depot]:
            # Determine whether this has had recharge time built into it.    

            # Outbound arc
            if (
                charge_fragment.start_time == new_depot.time 
                and charge_fragment.start_depot_id == new_depot.id
                and new_depot.charge - charge_fragment.charge >= 0
            ):
                outbound_arcs.append(charge_fragment)
            elif (
                # Inbound arc with no recharge
                charge_fragment.end_depot_id == new_depot.id 
                # and charge_fragment.end_charge <= new_depot.charge
                and charge_fragment.end_time <= new_depot.time 
                and self.get_charge(
                    charge_fragment.end_charge, new_depot.time - charge_fragment.end_time
                ) <= new_depot.charge
            ):
                invalid_inbound_arcs.append(charge_fragment)

        for charge_fragment in invalid_inbound_arcs:
            self.charge_fragments_by_charge_depot[prev_depot].remove(charge_fragment)
            self.charge_fragments_by_charge_depot[new_depot].add(charge_fragment)
            self.charge_depots_by_charge_fragment[charge_fragment].end = new_depot
        if origin_fragment is not None and origin_fragment not in invalid_inbound_arcs:
            frag_depots = self.charge_depots_by_charge_fragment[origin_fragment]
            store.depots_to_update.add(frag_depots.end)
            self.charge_fragments_by_charge_depot[frag_depots.end].remove(origin_fragment)
            self.charge_fragments_by_charge_depot[new_depot].add(origin_fragment)
            frag_depots.end = new_depot


        for charge_fragment in outbound_arcs:
            new_charge_fragment = ChargeFragment.from_fragment(
                start_charge=new_depot.charge, fragment=charge_fragment
            )
            if new_charge_fragment.id == 143 and new_charge_fragment.start_charge == 51:
                pass
            first_depot_which_can_leave = self._get_first_depot_vehicle_can_leave_from(
                new_charge_fragment.end_charge_depot
            )
            end_depot = self._get_closest_charge_depot(first_depot_which_can_leave)

            self.charge_fragments.add(new_charge_fragment)
            self._remove_charge_fragment_variable(new_charge_fragment)
            self._add_charge_fragment_variable(new_depot, end_depot, new_charge_fragment)
            nodes_to_update.add(end_depot)
            store.new_fragment_arcs.add(new_charge_fragment)
            for job in charge_fragment.jobs:
                store.jobs_to_update.add(job)
                self.charge_fragments_by_job[job].add(new_charge_fragment)

        store.depots_to_update.update(nodes_to_update)
        store.new_fragment_arcs.update(invalid_inbound_arcs)

        return nodes_to_update

    def _add_charge_fragment_variable(
            self, new_depot: ChargeDepot, end_depot: ChargeDepot, new_charge_fragment: ChargeFragment
        ) -> None:
        self.charge_fragments_by_charge_depot[new_depot].add(new_charge_fragment)
        self.charge_fragments_by_charge_depot[end_depot].add(new_charge_fragment)
        self.charge_fragments_by_id[new_charge_fragment.id].add(new_charge_fragment)
        self.charge_depots_by_charge_fragment[new_charge_fragment] = ChargeDepotStore(
                start=new_depot, end=end_depot
        )
        self.fragment_vars_by_charge_fragment[new_charge_fragment] = self.model.addVar(
                vtype=self.curr_vtype, name=f"f_{new_charge_fragment.id}_c_{new_charge_fragment.start_charge}"
        )

    def _update_charge_arcs(
            self, new_depot: ChargeDepot, store=ValidationStore()
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
        end_time = self._get_next_departure_time_from_depot(new_depot.id, new_depot.time)
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
        # if use_min_charge_condition:
        #     # Find the minimum charge required to execute any fragment at this time
        #     min_charge = self._get_min_fragment_charge_from_depot(new_depot.id, end_time)
        #     # If we can't execute any fragment at that time, need to add a new node at that point.
        #     if end_charge <= min_charge:        
        #         has_insufficient_charge = True
        
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

    def _get_min_fragment_charge_from_depot(self, depot_id: int, time: int) -> int:
        """
        Returns the smallest amount of charge required from a given depot location and time 
        needed to execute a fragment that leaves at that point.
        """
        if self.minimum_charge_by_time_by_depot.get((depot_id, time)) is not None:
            return self.minimum_charge_by_time_by_depot[depot_id, time]
        min_charge = min(
                (
                    fragment.charge 
                    for fragment in self.fragment_set
                    if fragment.start_depot_id == depot_id and fragment.start_time == time
                ),
                default=-1
            )
        self.minimum_charge_by_time_by_depot[depot_id, time] = min_charge
        return min_charge
    
    def _remove_recharge_arc_variable(self, recharge_arc: FrozenChargeDepotStore):
        self.recharge_arcs_by_charge_depot[recharge_arc.start].remove(recharge_arc)
        self.recharge_arcs_by_charge_depot[recharge_arc.end].remove(recharge_arc)
        self.recharge_arcs.remove(recharge_arc)
        self.model.remove(self.recharge_arc_var_by_depot_store[recharge_arc])
        del self.recharge_arc_var_by_depot_store[recharge_arc]

    def _remove_charge_fragment_variable(self, charge_fragment: ChargeFragment):
        try:
            self.model.remove(self.fragment_vars_by_charge_fragment[charge_fragment])
            del self.fragment_vars_by_charge_fragment[charge_fragment]
            store = self.charge_depots_by_charge_fragment.pop(charge_fragment)
            for job in charge_fragment.jobs:
                self.charge_fragments_by_job[job].remove(charge_fragment)
            self.charge_fragments_by_charge_depot[store.start].remove(charge_fragment)
            self.charge_fragments_by_charge_depot[store.end].remove(charge_fragment)
            self.charge_fragments.remove(charge_fragment)
        except KeyError:
            pass


    def _add_recharge_arc_variable(self, recharge_arc: FrozenChargeDepotStore) -> None:
        self.recharge_arcs_by_charge_depot[recharge_arc.start].add(recharge_arc)
        self.recharge_arcs_by_charge_depot[recharge_arc.end].add(recharge_arc)
        self.recharge_arcs.add(recharge_arc)
        vtype = GRB.INTEGER if self.curr_vtype == GRB.BINARY else GRB.CONTINUOUS
        self.recharge_arc_var_by_depot_store[recharge_arc] = self.model.addVar(
            vtype=vtype, name=f"w_{recharge_arc.start}_{recharge_arc.end}"
        )

    def set_solution(self) -> set[ChargeFragment]:
        paper_solution, _ = self.convert_solution_to_fragments(instance_type="regular", charging_style='non-linear')
        # charge_routes = self.get_validated_timed_solution(paper_solution)
        # used_fragments = {cf.id for route in charge_routes for cf in route if isinstance(cf, ChargeFragment)}
        # removes = set()
        # for cf in self.fragment_vars_by_charge_fragment:
        #     if cf.id not in used_fragments:
        #         removes.add(cf)
        # for cf in removes:
        #     self._remove_charge_fragment_variable(cf)
        self.model.addConstr(quicksum(self.starting_counts.values()) == len(paper_solution))

        # return {cf for cf in self.charge_fragments if cf not in removes}

    def _add_end_nodes_for_fragments(self) -> None:
        """
        Adds the true end nodes for each fragment generated leaving at full charge.
        This pre-populates the initial solution.
        #TODO: ensure this has propagated the fixes to fuck off copilot 
        """
        depots_to_add: set[ChargeDepot] = set()
        for cf in self.charge_fragments:
            end_depot = self._get_first_depot_vehicle_can_leave_from(cf.end_charge_depot)
            if end_depot.time == self.end_times_by_depot[cf.end_depot_id]:
                continue
            depots_to_add.add(end_depot)
        store = ValidationStore(depots_to_update=depots_to_add)
        for charge_depot in list(depots_to_add):
            self._add_new_depot(charge_depot, use_min_charge_condition=False, store=store)

        for node in store.depots_to_update:
            if (constr:=self.flow_balance.pop(node, None)) is not None:
                self.model.remove(constr)
            self._add_flow_balance_constraint(node)

        for job in store.jobs_to_update:
            if (constraint := self.coverage.pop(job, None)) is not None:
                self.model.remove(constraint)
            self._add_coverage_constraint(job)
        self.model.update()

    def _add_nodes_at_charge_level(self, charge: int, store=ValidationStore()) -> None:
        """
        Adds nodes at each charge level for each depot.
        """    
        depots_to_add: set[ChargeDepot] = set()
        for depot in self.charge_depots_by_depot:
            start_time = self.start_times_by_depot[depot]
            end_time = self.end_times_by_depot[depot]
            times = sorted(set(cd.time for cd in self.charge_depots_by_depot[depot]))
            for time in times:
                if time in [start_time, end_time]:
                    continue
                depots_to_add.add(ChargeDepot(id=depot, time=time, charge=charge))

        for charge_depot in depots_to_add:
            self._add_new_depot(charge_depot, use_min_charge_condition=False, store=store)
        store.depots_to_update.update(depots_to_add)

        for node in store.depots_to_update:
            if (constr:=self.flow_balance.pop(node, None)) is not None:
                self.model.remove(constr)
            self._add_flow_balance_constraint(node)

        for job in store.jobs_to_update:
            if (constraint := self.coverage.pop(job, None)) is not None:
                self.model.remove(constraint)
            self._add_coverage_constraint(job)

        self.model.update()

    def _set_minimum_charge_objective(self) -> None:
        self.model.setObjective(
            quicksum(
                self.fragment_vars_by_charge_fragment[cf] * cf.charge for cf in self.charge_fragments
            ),
            GRB.MINIMIZE
        )

    def _set_dead_heading_objective(self, sense="min"):
        self.model.setObjective(
            quicksum(
                self._get_travel_time_for_fragment(arc.id) * var  for arc, var in self.fragment_vars_by_charge_fragment.items()
            ),
            GRB.MINIMIZE if sense == "min" else GRB.MAXIMIZE
        )
    
    def _set_fleet_size_objective(self) -> None:
        self.model.setObjective(quicksum(self.starting_counts.values()), GRB.MINIMIZE)

    def _set_secondary_objective(self, obj: Objective=None) -> None:
        if obj is None:
            obj = self.solve_config.SECOND_OBJECTIVE
        match obj:
            case Objective.MIN_CHARGE:
                self._set_minimum_charge_objective()
            case Objective.MIN_DEADHEADING:
                self._set_dead_heading_objective(sense="min")
            case Objective.MAX_DEADHEADING:
                self._set_dead_heading_objective(sense="max")
            case Objective.MIN_VEHICLES:
                self._set_fleet_size_objective()

    def _early_exit(self, model, where):
        """
        At MIP nodes, check the solution and if it is feasible, exit early.
        """
        if where == GRB.Callback.MIPSOL:
            charge_fragment_sol = self.model.cbGetSolution(self.fragment_vars_by_charge_fragment)
            waiting_sol = self.model.cbGetSolution(self.recharge_arc_var_by_depot_store)
            solution_charge_fragments: set[tuple[ChargeFragment, int]] = {
                (cf, charge_fragment_sol[cf]) 
                for cf in charge_fragment_sol
                if charge_fragment_sol[cf] > VAR_EPS
            }
            waiting_arcs = [
                (arc.start, arc.end, waiting_sol[arc]) 
                for arc in waiting_sol 
                if waiting_sol[arc] > VAR_EPS
            ]
            routes, _ = self.forward_label(solution_charge_fragments, waiting_arcs)
            is_valid = True
            for route in routes:
                is_valid, segment = self.validate_route(route.route_list)
                if not is_valid:
                    break

            if is_valid:
                print("\n\nSolution found in branch and bound!\n\n")
                # self.mip_solution = routes
                model.terminate()
                return
        # if where == GRB.Callback.MIP:
        #     lower = self.model.cbGet(GRB.Callback.MIP_OBJBST)
        #     upper = self.model.cbGet(GRB.Callback.MIP_OBJBND)
        #     gap = abs((upper - lower)/lower)
        #     if gap < 0.05:
        #         print("\n\n Gap isless than 5%, proceeding to DDD...\n\n")
        #         # self.mip_solution = routes
        #         model.terminate()
        #         return


    def solve(self):
        print("setting solve...")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("Seed", 0)
        self.model.update()
        use_visualisation = self.solve_config.INCLUDE_VISUALISATION
        valid_solution = False
        has_valid_lp_relaxation = False
        model_solve_time_key = "lp_runtime"
        model_iter_key = "num_lp_iters"
        n_iters = 0
        prev_obj = 0
        if use_visualisation:
            fig = Figure()
        self.statistics.update(
            {
                "runtime": 0,
                "inspection_time": 0,
                "num_lp_iters": 0,
                "num_mip_iters": 0,
                "lp_runtime": 0,
                "mip_runtime": 0,
                "initial_bound": 0,
                "final_bound": 0,
                "infeasible_route_segments": 0,
            }
        )

        # self._add_initial_network_cuts()
        self._set_variable_domain(GRB.CONTINUOUS)
        self.model.setParam("TimeLimit", self.solve_config.TIME_LIMIT)
        self.model.optimize()
        remaining_time = self.solve_config.TIME_LIMIT - self.model.Runtime
        self.statistics["runtime"] += self.model.Runtime
        self.statistics["initial_bound"] = self.model.objval
        self.statistics[model_solve_time_key] += self.model.Runtime
        # self.set_solution()
        self.vehicle_count = self.model.objval
        print(f"Initial solve took {self.model.Runtime}, {self.vehicle_count=}")#, paper: {self._get_paper_objective_value()}")
        self.obj_constr = self.model.addConstr(quicksum(self.starting_counts.values()) == self.vehicle_count, name="objective_limit")
        callback = None
        while (
            not valid_solution 
            and n_iters < self.solve_config.MAX_ITERATIONS 
            and 0 < remaining_time
        ):
            self._set_secondary_objective()
            n_iters += 1
            self.statistics[model_iter_key] += 1
            print(f"Starting iteration {n_iters}")            
            self.model.update()
            self.model.setParam("Seed", 0)
            self.model.setParam("TimeLimit", remaining_time)
            self.model.optimize(callback)
            self.statistics["runtime"] += self.model.Runtime
            self.statistics[model_solve_time_key] += self.model.Runtime
            remaining_time -= self.model.Runtime

            if self.model.status == GRB.Status.OPTIMAL or self.model.status == GRB.Status.INTERRUPTED:
                solution_charge_fragments: set[tuple[ChargeFragment, int]] = {
                    (cf, self.fragment_vars_by_charge_fragment[cf].x) 
                    for cf in self.fragment_vars_by_charge_fragment 
                    if getattr(self.fragment_vars_by_charge_fragment[cf], 'x', -1) > VAR_EPS
                }
                if abs(self.model.objval - prev_obj) > VAR_EPS:
                    prev_obj = self.model.objval
                    print("    ", self.model.objval)
                    
                waiting_arcs = [
                    (arc.start, arc.end, self.recharge_arc_var_by_depot_store[arc].x) 
                    for arc in self.recharge_arc_var_by_depot_store 
                    if self.recharge_arc_var_by_depot_store[arc].x > VAR_EPS
                ]
                if use_visualisation:
                    fig = self.add_solution_trace(itern=n_iters)
                network_additions = ValidationStore()
                print("    Inspecting solution...")
                time0 = timer.time()
                valid_solution = self.inspect_solution(
                    solution_charge_fragments, waiting_arcs, store=network_additions
                )
                inspection_time = timer.time() - time0
                if use_visualisation:
                    self.add_inspection_trace(n_iters, network_additions)
                self.statistics["inspection_time"] += inspection_time
                remaining_time -= inspection_time
                if valid_solution and not has_valid_lp_relaxation:
                    valid_solution = False
                    print("    Valid linear relaxation, changing to integer variables...")
                    self._set_variable_domain(GRB.BINARY)
                    callback = self._early_exit
                    # self.model.setParam("LazyConstraints", 1)
                    self.model.setParam("outputFlag", 1)
                    model_solve_time_key = "mip_runtime"
                    model_iter_key = "num_mip_iters"
                    has_valid_lp_relaxation = True
            # elif self.model.status == GRB.Status.INTERRUPTED:
            #     print("Interrupted")
            #     break
            else:
                # Hit a lower bound issue
                self.handle_infeasible_model()

        hit_runtime_limit = (
            remaining_time <= 0
            or n_iters >= self.solve_config.MAX_ITERATIONS
        )
        # Default no routes
        routes = None
        if not hit_runtime_limit:
            print(f"solved in {n_iters} iterations")
            if self.model.Status == GRB.Status.OPTIMAL or self.model.Status == GRB.Status.INTERRUPTED:
                solution_charge_fragments: set[tuple[ChargeFragment, int]] = {
                    (cf, self.fragment_vars_by_charge_fragment[cf].x) 
                    for cf in self.fragment_vars_by_charge_fragment 
                    if getattr(self.fragment_vars_by_charge_fragment[cf], 'x', -1) > VAR_EPS
                }   
                waiting_arcs = [
                    (arc.start, arc.end, self.recharge_arc_var_by_depot_store[arc].x) 
                    for arc in self.recharge_arc_var_by_depot_store 
                    if self.recharge_arc_var_by_depot_store[arc].x > VAR_EPS
                ]
                routes, _ = self.forward_label(solution_charge_fragments, waiting_arcs)
            # elif self.model.Status == GRB.Status.INTERRUPTED:
            #     routes = self.mip_solution

            for route in routes:
                is_valid, _ = self.validate_route(route.route_list)
                if not is_valid:
                    is_valid, _ = self.validate_route(route.route_list)
                assert is_valid
        else:
            print("Hit runtime limit")

        self.statistics.update(
            {
                "num_iters": n_iters,
                "objective": len(routes) if not hit_runtime_limit else -1,
                'solved': not hit_runtime_limit,
                'solve_type': self.solve_config.SECOND_OBJECTIVE.value,
                "final_bound": self.vehicle_count,
                "total_time": self.statistics["runtime"] + self.statistics['inspection_time'],
            }
        )
        if use_visualisation:
            # visualise_network_transformation(fig)
            # fig = animate_network_transformation(
            #     self.ddd_traces, [0, int(self.config.MAX_CHARGE * 1.2)], self.statistics['num_lp_iters']
            # )
            path = f"../images/{self.data['label']}_transformation_{self.solve_config.SECOND_OBJECTIVE.value}/"
            if not os.path.exists(path):
                os.makedirs(path)
            write_network_transformation(
                self.ddd_traces, [0, int(self.config.MAX_CHARGE * 1.2)], self.statistics['num_lp_iters'], path
            )
            # fig.write_html(
            #     f"../images/{self.data['label']}_transformation_{self.solve_config.SECOND_OBJECTIVE.value}.html"
            # )
        print(f"{self.statistics=}")
        return routes

    def handle_infeasible_model(self):
        """
        Increases the constraint on the number of vehicles leaving the start by 1.
        If the vehicle count is above the number of jobs, then the relaxation is infeasible and an IIS is computed.
        """
        if (
            self.vehicle_count is None
            or self.obj_constr is None
            or self.vehicle_count > len(self.jobs)
        ):
            self.model.computeIIS()
            self.model.write("fragment_network.ilp")
            raise Exception("Relaxation is infeasible")
        self.vehicle_count += 1            
        self.obj_constr.RHS = self.vehicle_count

    def add_solution_trace(self, itern: int):
        # if fig is None:
            # fig = Figure()
        solution_charge_fragments = [
            cf for cf in self.fragment_vars_by_charge_fragment if getattr(self.fragment_vars_by_charge_fragment[cf], 'x', -1) > VAR_EPS
        ]
        frame = visualise_charge_network(
                    self.charge_depots,
                    [
                        (
                          self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
                        )
                        for cf in solution_charge_fragments
                    ], 
                    [arc for arc in self.recharge_arc_var_by_depot_store if getattr(self.recharge_arc_var_by_depot_store[arc], "x", -1) > VAR_EPS],
                   iter=itern,
                   graph_type="Solution",
                   max_charge=self.config.MAX_CHARGE,
                   showlegend=False,
                   iter_code=f"Iteration {itern} - {'LP' if self.statistics['num_mip_iters'] == 0 else 'MIP'} - |A| = {len(self.charge_fragments) + len(self.recharge_arcs)} - |N| = {len(self.charge_depots)}",
                )
        self.ddd_traces.append(frame)

    def add_inspection_trace(self, itern: int, network_additions: ValidationStore):
        if (
            len(network_additions.new_fragment_arcs) == 0
            and len(network_additions.new_recharge_arcs) == 0
            and len(network_additions.depots_to_update) == 0
        ):
            foo_depot = ChargeDepot(id=0, time=0, charge=2000)
            new_charge_fragments = [(foo_depot, foo_depot)]
            new_depots = {foo_depot}
            new_recharge_arcs = {FrozenChargeDepotStore(start=foo_depot, end=foo_depot)}
        else:
            new_charge_fragments = [
                (
                    self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
                )
                for cf in network_additions.new_fragment_arcs
            ]
            new_recharge_arcs = network_additions.new_recharge_arcs.intersection(self.recharge_arcs)
            new_depots = network_additions.depots_to_update
    
        frame = visualise_charge_network(
            new_depots,
            new_charge_fragments, 
            new_recharge_arcs,
            iter=itern,
            graph_type="Repair",
            max_charge=self.config.MAX_CHARGE,
            showlegend=False
        )
        self.ddd_traces[-1].data += frame.data
        # trace = visualise_charge_network(
        #             network_additions.depots_to_update,
        #             [
        #                 (
        #                   self.charge_depots_by_charge_fragment[cf].start, self.charge_depots_by_charge_fragment[cf].end
        #                 )
        #                 for cf in network_additions.new_fragment_arcs
        #             ], 
        #             network_additions.new_recharge_arcs.intersection(self.recharge_arcs),
        #             fig=fig,
        #             graph_type="Repair"
        #         )
        # self.ddd_traces.append(frame)

    def _add_initial_network_cuts(self):
        """Adds additional nodes prior to optimisation."""
        # self._add_nodes_at_charge_level(int(np.percentile([cf.charge for cf in self.charge_fragments], 50)))
        self._add_end_nodes_for_fragments()

    def _set_variable_domain(self, var_type: str) -> None:
        """
        Sets all variable vtypes to the given type. This is either GRB.BINARY or GRB.CONTINUOUS
        In the binary case, integer variables are set instead where appropriate. 
        i.e. sets it to the linear relaxation or integer version of the problem.
        """
        self.curr_vtype = var_type
        for var in self.fragment_vars_by_charge_fragment.values():
            var.VType = var_type
        
        # Can't have binary variables for the counts or recharge arcs
        if self.curr_vtype == GRB.BINARY:
            var_type = GRB.INTEGER

        for var in self.recharge_arc_var_by_depot_store.values():
            var.VType = var_type
        for var in self.starting_counts.values():
            var.VType = var_type
        for var in self.finishing_counts.values():
            var.VType = var_type

    def _get_travel_time_for_fragment(self, id: int) -> int:
        """
        Gets the travel time for a fragment from the data
        """
        if self.deadheading_time_by_fragment_cache.get(id) is None:
            fragment = self.fragments_by_id[id]
            travel_distance = (
                self.depot_to_job_charge_matrix[fragment.start_depot_id][fragment.jobs[0].id]
                + self.job_to_depot_charge_matrix[fragment.jobs[-1].id][fragment.end_depot_id]
            )
            self.deadheading_time_by_fragment_cache[id] = travel_distance
        return self.deadheading_time_by_fragment_cache[id]

    def validate_timed_network(self):
        """
        Validates the timed network to ensure it is feasible.
        For a timed network to be feasible:
        - No timed depot can have fragments which start after its time
        - Each timed depot cannot have a time earlier than the previous timed depot
        """
        for depot_id, charge_depots in self.charge_depots_by_depot.items():
            prev_td = ChargeDepot(id=depot_id, time=0, charge=0)
            for td in sorted(charge_depots):
                if prev_td:
                    assert td.time >= prev_td.time
                    if td.time < prev_td.time:
                        print(f"Depot {td} has a time earlier than the previous depot.")
                        return False
                fragments = self.charge_fragments_by_charge_depot[td]
                # Check the following:
                # min end time >= max start time
                # no start time later than the current depot time and no earlier than the previous time.
                departure_fragments = [cf for cf in fragments if cf.start_charge_depot == td]
                arrival_fragments = [cf for cf in fragments if cf.start_charge_depot != td]
                if len(departure_fragments) != 0:
                    assert all(prev_td.time <= tf.start_time <= td.time for tf in departure_fragments)
                if len(arrival_fragments) != 0:
                    for tf in arrival_fragments:
                        end_depot = self._get_first_depot_vehicle_can_leave_from(tf.end_charge_depot)
                        assert prev_td.time <= end_depot.time <= td.time
                    # assert all(prev_td.time <= self._get_first_depot_vehicle_can_leave_from(tf.end_charge_depot).time <= td.time for tf in arrival_fragments)
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
                    dep_idx = bisect.bisect_right(self.departure_times_by_depot[charge_depot.id], charge_depot.time)
                    arr_idx = bisect.bisect_right(self.arrival_times_by_depot[charge_depot.id], charge_depot.time)
                    # No time greater
                    if dep_idx == len(self.departure_times_by_depot[charge_depot.id]):
                        dep_time = 1e10
                    else:
                        dep_time = self.departure_times_by_depot[charge_depot.id][dep_idx]
                    if arr_idx == len(self.arrival_times_by_depot[charge_depot.id]):
                        arr_time = 1e10
                    else:
                        arr_time = self.arrival_times_by_depot[charge_depot.id][arr_idx]
                    next_time = min(
                        arr_time, dep_time
                    )
                    assert arc.end.time == next_time, f"{arc.end.time=}, {self._get_next_departure_time_from_depot(charge_depot.id, charge_depot.time)=}"
                    assert arc.end.charge >= arc.start.charge
                    charge_depot = arc.end
                    has_neighbour = True

    def _get_paper_objective_value(self, path = r"data/mdevs_solutions.xlsx") -> float:
        data = pd.read_excel(path, sheet_name=None)
        instance_type = "regular" if "000" not in self.data["label"] else 'large' 
        for sheet in data:
            if instance_type in sheet:
                data = data[sheet]
                break

        return data.query(
            f"ID_instance == {self.data['ID']} and battery_charging == 'non-linear'"
        )["objective_value"].values[0]

    def _calculate_upper_bound_on_network_size(self) -> None:
        """
        Calculates the number of nodes in the network required of an a-priori solve.  
        To determine this number, we need to consider the following:
        From the initial network: 
        - iterate through the existing charge fragments
        - add each end node, keep track of everything new that was added with ValidationStore.
        - repeat with new charge_fragments from last iteration until no new arcs are added.
        - use add_repair_trace with each iteration.       
        """
        store = ValidationStore()
        fig = Figure()
        for cf in list(self.charge_fragments):
            self._add_new_depot(cf.end_charge_depot, store=store)
        while len(store.new_fragment_arcs) > 0:
            new_store = ValidationStore()
            for cf in store.new_fragment_arcs:
                self._add_new_depot(cf.end_charge_depot, store=new_store)
            self.add_inspection_trace(fig=fig, network_additions=new_store)
            self.add_inspection_trace(fig=fig, network_additions=new_store)
            store = new_store             
        visualise_network_transformation(fig).write_html(f"../images/a_priori_generation_{self.data['label']}.html")
        self.statistics.update(
            {
                "a_priori_charge_network_arcs": len(self.charge_fragments) + len(self.recharge_arcs),
                "a_priori_charge_network_nodes": len(self.charge_depots)
            }
        )

    def run(self, output_override: str | None = None):
        """Runs an end-to-end solve ."""
        print(f"Solving {self.data['label']}...")
        print("generating fragments...")
        self.generate_fragments()
        print("generating timed network...")
        self.generate_timed_network()
        # with cProfile.Profile() as profile:
        #     profile.create_stats()
        #     profile.dump_stats("network_gen_profile_new_next_time.prof")
        print("validating timed network...")
        self.validate_timed_network()
        print("building model...")
        self.build_model()
        print("solving...")
        # with cProfile.Profile() as profile:
        routes = self.solve()
            # profile.create_stats()
            # profile.dump_stats("solve_profile_no_all_depot_check.prof")
        # Final network size
        self.statistics.update(
            {
                "final_charge_network_arcs": len(self.charge_fragments) + len(self.recharge_arcs),
                "final_charge_network_nodes": len(self.charge_depots)
            }
        )
        # routes = []
        # self._calculate_upper_bound_on_network_size()
        self.write_solution(routes, output_override=output_override)
        # if output_path is not None:
        #     self.write_statistics(file=output_path)
