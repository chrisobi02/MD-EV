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
from itertools import product
from visualiser import visualise_timed_network, visualise_routes
from fragment_generation import ConstantFragmentGenerator
from non_linear_fragment_generation import NonLinearFragmentGenerator
from naive_ip import NaiveIP

from constants import *

def solve_ip(file: str):
    """Solve the IP model."""
    ip = NaiveIP(file)
    ip.run()

def main():
    # file = r"data/instances_regular/I-5-5-200-10.json"
    # file = r"data/instances_large/I-7-7-1000-01.json"
    file = r"data/instances_regular/I-1-1-50-04.json"
    solve_ip(file)
# 
    # Specify the directory you want to search
    directory = "data/instances_large/"
    directory = "data/instances_regular/"

    # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
    EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
    # Iterate over the list of filepaths & open each file
    for json_file in json_files:     
        if "fragments" in str(json_file):
            continue
        # if any(ex in str(json_file) for ex in EXCLUDED_INSTANCES):
        # if "I-3-3-100-08.json" not in str(json_file):
        # if "50" not in str(json_file):
        # if "I-5-5-200-07.json" not in str(json_file):
        # if "I-3-3-100-05" not in str(json_file):
            # continue
        # if "50" not in str(json_file):
        #     continue
        print(f"Solving {json_file}...")
        print("generating fragments...")
        # remove the last /, append fragments and then the part on the other side f the slice

        frag_file = json_file.split("/")
        # prev_runs = pd.read_csv("large_results.csv")
        # if frag_file[-1].split(".")[0] in prev_runs[prev_runs["method"] == "fragments"]["label"].values:
        #     continue
        generator = ConstantFragmentGenerator(json_file)
        # generator = NonLinearFragmentGenerator(json_file)
        # generator = ConstantFragmentGenerator(json_file)
        str_frag_file = "/".join(frag_file[:-1]) + "/fragments/" + "f-" + frag_file[-1]
        # generator.generate_fragments()#file=str_frag_file)
        generator.run()#file=str_frag_file)
        
        # print("generating timed network...")
        # print(len(generator.fragment_set))
        for fragment in generator.fragment_set:
            generator.validate_fragment(fragment, CHARGE_MAX, fragment.start_time)
        # generator.generate_timed_network()
        # generator.validate_timed_network()
        # # visualise_timed_network(generator.timed_depots_by_depot, generator.fragment_set, set(
        # #     a for d in generator.timed_depots_by_depot for a in zip(generator.timed_depots_by_depot[d][:-1], generator.timed_depots_by_depot[d][1:])
        # # ),
        # # generator.timed_fragments_by_timed_node)
        # # print("writing fragments...")
        # # generator.write_fragments()
        # print("building model...")
        # generator.build_model()
        # print("incumbent solution")
        # # prior_solution, solution_routes = generator.read_solution(instance_type=frag_file[-2].split("instances_")[-1])
        
        # # generator.visualise_routes(generator.get_validated_timed_solution(prior_solution))
        # # generator.set_solution(prior_solution, n_vehicles=len(prior_solution))
        # # all_prior_fragments = set(f for s in prior_solution for f in s)
        # # get fragments associated with a timed depot
        
        # print("solving...")
        # generator.solve()
        # # print(f"Prior Solution: {len(solution_routes)}")
        # print("sequencing routes...")
        # # routes = generator.create_routes()
        # # print(f"Fragment routes: {len(routes)}")
        # # generator.validate_solution(routes, generator.model.objval)
        # # paper_results = pd.read_csv('large_results.csv', index_col=0)
        # paper_results = pd.read_csv('fixed_charge_cost.csv', index_col=0)
        # # paper_results = pd.read_excel('data/mdevs_solutions.xlsx', sheet_name="results_large_BCH")
        # # paper_results = paper_results[paper_results["battery"] == "constant-time"]
        # # # check the incumbent solution
        # assert generator.model.objval == paper_results[paper_results["label"] == generator.data["label"]]["objective"].values[0]
        # # assert the number of fragments is the same
        # assert len(generator.fragment_set) == paper_results[paper_results["label"] == generator.data["label"]]["num_fragments"].values[0], paper_results[paper_results["label"] == generator.data["label"]]["num_fragments"].values[0]
        # generator.statistics["method"] = "fragments"
        # generator.write_statistics(file="large_results.csv")
        # defaultdict(default_factory=)

if __name__ == "__main__":
    main()