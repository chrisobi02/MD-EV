import json
import os
import time
import pandas as pd
import glob
import click
from mdevs.formulations import *

@click.group()
def cli():
    pass

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
            if "fragments" in str(json_file) or "4000" not in json_file:
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

def constant_time_debug():
    # Specify the directory you want to search
    directory = "data/instances_regular/"
    directory = "data/instances_large/"

    # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
    EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
    # Iterate over the list of filepaths & open each file
    for json_file in json_files:     
        if "fragments" in str(json_file):
            continue
        # if any(ex in str(json_file) for ex in EXCLUDED_INSTANCES):
        # if "I-3-3-100-08.json" not in str(json_file):
        # if "I-5-5-200-07.json" not in str(json_file):
        # if "I-7-7-4000-02.json" not in str(json_file):
        # if "I-3-3-100-05" not in str(json_file):
        # if "50" not in str(json_file):
            # continue
        # print(f"Solving {json_file}...")
        # print("generating fragments...")
        # remove the last /, append fragments and then the part on the other side f the slice

        frag_file = json_file.split("/")
        # prev_runs = pd.read_csv("large_results.csv")
        # if frag_file[-1].split(".")[0] in prev_runs[prev_runs["method"] == "fragments"]["label"].values:
        #     continue
        generator = ConstantFragmentGenerator(json_file)
        violations = generator.validate_triangle_inequality()
        
        str_frag_file = "/".join(frag_file[:-1]) + "/fragments/" + "f-" + frag_file[-1]
        generator.generate_fragments()#file=str_frag_file)
        
        # print("generating timed network...")
        generator.generate_timed_network()
        generator.validate_timed_network()
        # # visualise_timed_network(generator.timed_depots_by_depot, generator.fragment_set, set(
        # #     a for d in generator.timed_depots_by_depot for a in zip(generator.timed_depots_by_depot[d][:-1], generator.timed_depots_by_depot[d][1:])
        # # ),
        # # generator.timed_fragments_by_timed_node)
        # print("writing fragments...")
        # # generator.write_fragments()
        print("building model...")
        generator.model.setParam("OutputFlag", 0)
        generator.build_model()
        # # print("incumbent solution")
        # # prior_solution, solution_routes = generator.get_solution_fragments(instance_type=frag_file[-2].split("instances_")[-1], sheet_name="results_regular_BCH")
        
        # # generator.set_solution(prior_solution, n_vehicles=len(prior_solution))
        # # all_prior_fragments = set(f for s in prior_solution for f in s)
        # # get fragments associated with a timed depot
        print("solving...")
        generator.solve()
        # print(f"Prior Solution: {len(solution_routes)}")
        print("sequencing routes...")
        generator.routes = routes = generator.create_routes()
        new_routes = generator.forward_label()
        # generator.visualise_routes([r.route_list for r in new_routes])
        print(f"Fragment routes: {len(routes)}, {len(new_routes)}")
        print(sum(len(r.jobs) for r in new_routes))
        routess = [r.route_list for r in new_routes]
        generator.validate_solution([r.route_list for r in new_routes], generator.model.objval)

@cli.command()
def non_linear_debug():
    click.echo("Running non-linear debug")
      # Specify the directory you want to search
    directory = "mdevs/data/instances_large/"
    directory = "mdevs/data/instances_regular/"

    # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
    EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
    click.echo(json_files)
    # Iterate over the list of filepaths & open each file
    for json_file in json_files:     
        if "fragments" in str(json_file):
            continue
        print(f"Solving {json_file}...")
        model = NonLinearFragmentGenerator(json_file, config={"RECHARGE_TIME": 0.5})
        model.run()
        model.create_routes()
        continue

cli.add_command(non_linear_debug)

if __name__ == "__main__":
    # cli()
    non_linear_debug()
    # constant_time_debug()
    # compare_sequencing_procedures()
    # non_linear_debug()