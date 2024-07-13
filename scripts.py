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

def compare_sequencing_procedures():
    """Compare recursive and forward labelling algorithms for route construction."""
    directories= ["data/instances_large/", "data/instances_regular/"] 
    sequence_data = []
    fails = []
    for directory in directories:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        # EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if "fragments" in str(json_file):
                continue
            # if any(ex in str(json_file) for ex in EXCLUDED_INSTANCES):
            # if "I-3-3-100-08.json" not in str(json_file):
            # if "50" not in str(json_file):
            # if "I-5-5-200-07.json" not in str(json_file):
            # if "I-3-3-100-08" not in str(json_file):
            # if "I-3-3-100-08" not in str(json_file):
            #     continue
            # # if "50" not in str(json_file):
            #     continue
            print(f"Solving {json_file}...")
            print("generating fragments...")
            # remove the last /, append fragments and then the part on the other side f the slice

            frag_file = json_file.split("/")
            # prev_runs = pd.read_csv("large_results.csv")
            # if frag_file[-1].split(".")[0] in prev_runs[prev_runs["method"] == "fragments"]["label"].values:
            #     continue
            # json_file = r"data/instances_regular/I-1-1-50-04.json"
            str_frag_file = "/".join(frag_file[:-1]) + "/fragments/" + "f-" + frag_file[-1]
            # generator.generate_fragments()#file=str_frag_file)
            
            params = {"UNDISCRETISED_MAX_CHARGE": 100}
            try:
                generator = ConstantFragmentGenerator(json_file)
                # ip = NaiveIP(json_file, params=params)
                generator.run()#file=str_frag_file)
                result = {"Jobs": len(generator.jobs), "Depots": len(generator.depots)}
                time0 = time.time()
                routes = generator.create_routes()
                result["Recursion"] = time.time() - time0
                generator.validate_solution(routes, generator.model.objval, triangle_inequality=False)
                            
                time0 = time.time()
                routes = generator.forward_label()
                result["Forward Labelling"] = time.time() - time0
                generator.validate_solution([r.route_list for r in routes], generator.model.objval, triangle_inequality=False)
                sequence_data.append(result)
            except Exception as e:
                fails.append((json_file, e))
    pd.DataFrame(sequence_data).to_csv("data/results/sequencing_algorithm_results.csv")
    print("Failed instances:", fails)
    for f in fails:
        print(f[0])
        print(f[1])

def main():
    # file = r"data/instances_regular/I-5-5-200-10.json"
    # file = r"data/instances_large/I-7-7-1000-01.json"
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
        # if "I-3-3-100-08" not in str(json_file):
        if "I-3-3-100-08" not in str(json_file):
            continue
        # if "50" not in str(json_file):
        #     continue
        print(f"Solving {json_file}...")
        print("generating fragments...")
        # remove the last /, append fragments and then the part on the other side f the slice

        frag_file = json_file.split("/")
        # prev_runs = pd.read_csv("large_results.csv")
        # if frag_file[-1].split(".")[0] in prev_runs[prev_runs["method"] == "fragments"]["label"].values:
        #     continue
        # json_file = r"data/instances_regular/I-1-1-50-04.json"
        str_frag_file = "/".join(frag_file[:-1]) + "/fragments/" + "f-" + frag_file[-1]
        # generator.generate_fragments()#file=str_frag_file)
        params = {"UNDISCRETISED_MAX_CHARGE": 70}
        generator = ConstantFragmentGenerator(json_file, params=params)
        # ip = NaiveIP(json_file, params=params)
        generator.run()#file=str_frag_file)
        routes = generator.create_routes()
        generator.validate_solution(routes, generator.model.objval, triangle_inequality=False)
        route_ids = []
        paper_format = []
        for route in routes:
            curr = [route[0].id]
            paper_curr = []
            for i, loc in enumerate(route):
                if not isinstance(loc, TimedDepot):
                    curr.extend(j.offset_id for j in loc.jobs)
                    paper_curr.extend(f"{j.offset_id+ 1}" for j in loc.jobs)
                else:
                    if i ==0:
                        paper_curr.append(f"D{loc.id+1}")
                    else:
                        paper_curr.append(f"S{loc.id +len(generator.depots) + 1 + len(generator.jobs)}")
                    # paper_curr.append(loc.id+1)

            curr.append(loc.id)
            paper_curr.append(f"D{loc.id+1}")
            paper_format.append("["+str.join(">", paper_curr)+"]")
            route_ids.append(curr)

        # ip.run(sol=route_ids)
        paper_ids = []
        print(paper_format)
        # paper_format = []
        for route in route_ids:
            curr = []
            id_curr = []
            for i, loc in enumerate(route):
                if loc < len(generator.depots):
                    if i == 0 or i == len(route)-1:
                        # curr.append(f"D{loc+1}")
                        id_curr.append(loc+1)
                else:
                    id_curr.append(loc+1)
            # paper_format.append(str.join(">", curr))
            paper_ids.append(id_curr)
        print(paper_ids)
# solution
# [D2>4>S105>16>S104>27>S104>40>S105>61>S104>77>S105>95>S104>D1],[D1>5>13>S104>38>S105>79>S104>D1],[D1>6>S106>17>S105>31>S104>53>S104>80>S104>D1],[D1>7>S105>25>S105>42>S104>57>S104>69>S104>85>96>S104>D1],[D1>8>S104>23>S105>36>S105>52>S104>70>S104>91>S104>D1],[D1>9>S104>21>S104>35>S104>58>S104>72>S106>93>S104>D1],[D1>10>S104>24>S104>39>S105>54>S106>71>S106>87>S105>100>S104>D1],[D1>11>S104>41>S104>60>S104>82>S104>97>S104>D1],[D1>12>S105>55>S105>84>S104>D1],[D1>14>S105>34>S104>47>S104>66>S105>89>S104>D1],[D1>15>S105>26>S105>37>S105>56>S105>67>S106>90>S105>101>S104>D1],
# [D1>18>S106>29>S105>45>S105>63>S105>86>S104>103>S104>D1],[D1>19>S105>32>S104>51>S105>75>S106>94>S104>D1],[D1>20>S105>33>S105>50>S104>68>S104>81>S106>99>S104>D1],[D1>22>S106>30>S104>49>S106>74>S104>D1],[D1>28>S104>44>S104>62>S105>76>S105>98>S104>D1],[D1>43>59>S106>73>88>S104>D1],[D2>46>64>S104>78>S104>92>S104>102>S104>D1],[D1>48>S105>65>S106>83>S104>D1]
        # Check paper format works
        # ConstantFragmentGenerator(json_file, params=params).read

        # ip.run()
        # assert ip.model.Status == GRB.OPTIMAL
        # routes = ip.sequence_routes()
        # assert len([j for r in routes for j in r if isinstance(j, Job)]) == 50
        # fragment_routes = []
        # for current_route in routes:
        #     route_fragment_ids = generator.convert_route_to_fragments(current_route)
        #     fragment_routes.append(route_fragment_ids)
        # ip.run(sol=route_ids)
        # generator.set_solution(fragment_routes, n_vehicles=ip.model.objVal)
        print()
        # print("incumbent solution")
        # prior_solution, solution_routes = generator.read_solution(instance_type=frag_file[-2].split("instances_")[-1])
        # generator.set_solution(prior_solution, n_vehicles=len(prior_solution))
        
        # print("solving...")
        # generator.solve()
        # # print(f"Prior Solution: {len(solution_routes)}")
        # print("sequencing routes...")
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
    # main()
    compare_sequencing_procedures()