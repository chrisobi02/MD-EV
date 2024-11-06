import json
import os
import time
import pandas as pd
import glob
import click
from mdevs.formulations import *
from mdevs.formulations.charge_functions import ConstantChargeFunction, NoChargeFunction
import cProfile

NON_LINEAR_BASIC_CONFIG = CalculationConfig(
    UNDISCRETISED_MAX_CHARGE=100,
    RECHARGE_DELAY_IN_MINUTES=0.5,        
)

CONSTANT_TIME_BASIC_CONFIG = CalculationConfig(
    UNDISCRETISED_MAX_CHARGE=100,
)

DIRECTORIES = [
    # "data/instances_regular/",
    # "data/instances_constrained/",
    "data/instances_num_of_depots_experim",
    # "data/instances_large/",
]

@click.group()
def cli():
    pass


def constant_time_debug():
    # Specify the directory you want to search
    directory = "data/instances_large/"
    directory = "mdevs/data/instances_regular/"

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
        generator = ConstantTimeFragmentGenerator(json_file)
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
        # generator.routes = routes = generator.create_routes()
        new_routes = generator.forward_label()
        # generator.visualise_routes([r.route_list for r in new_routes])
        print(f"Fragment routes: {len(new_routes)}")
        print(sum(len(r.jobs) for r in new_routes))
        routess = [r.route_list for r in new_routes]
        generator.validate_solution([r.route_list for r in new_routes], generator.model.objval)

def interpolation_debug():   
    click.echo("Running intepolr debug")
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
        # if "50-01" not in str(json_file):
        # if "100" not in str(json_file):
            # continue
        print(f"Solving {json_file}...")
        model = InterpolationIP(
            json_file, 
            charge_calculator_class=PaperChargeFunction,
            charge_calculator_kwargs={"discretise": False},
            config=NON_LINEAR_BASIC_CONFIG,
        )
        routes = model.run()
        model = NonLinearFragmentGenerator(
            json_file, 
            config=NON_LINEAR_BASIC_CONFIG,
            solve_config=SolveConfig(
                SECOND_OBJECTIVE=Objective.MIN_DEADHEADING,
                INCLUDE_VISUALISATION=False,
            )
        )
        model.generate_fragments()
        for route in routes:
            frags = model.convert_route_to_fragment_route(route)
            frag_route = []
            for f in frags:
                cf = ChargeFragment.from_fragment(start_charge=model.config.MAX_CHARGE, fragment=model.fragments_by_id[f])
                frag_route.extend(
                    [
                        cf.start_charge_depot,
                        cf,
                        cf.end_charge_depot,
                    ]
                )
            is_valid, segment = model.validate_route(frag_route)
            if not is_valid:
                print(f"Invalid route: {segment}")
                raise ValueError("Invalid route")
    
@cli.command()
def non_linear_debug():   
    click.echo("Running non-linear debug")
      # Specify the directory you want to search
    directory = "mdevs/data/instances_large/"
    directory = "mdevs/data/instances_regular/"
    directory = "data/instances_num_of_depots_experim/"

    # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True) 
    EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
    click.echo(json_files)
    # Iterate over the list of filepaths & open each file
    for json_file in json_files:     
        if any([s in str(json_file) for s in ["fragments", 'solutions']]):
            continue
        # if "200-01" not in str(json_file):
        if "1-1-200-01" not in str(json_file):
            continue
        print(f"Solving {json_file}...")
        # model = ConstantTimeFragmentGenerator(
        #     json_file, 
        #     config=CONSTANT_TIME_BASIC_CONFIG,
        # )
        # model.generate_fragments()
        for obj in [
            Objective.MIN_CHARGE,
            Objective.MIN_VEHICLES,
            # Objective.MIN_DEADHEADING,
        ]:
            model = NonLinearFragmentGenerator(
                json_file, 
                config=NON_LINEAR_BASIC_CONFIG,
                solve_config=SolveConfig(
                    SECOND_OBJECTIVE=obj,
                    INCLUDE_VISUALISATION=True,
                )
            )
            # # all_prior_fragments = set(f for s in prior_solution for f in s)
            # # get fragments associated with a timed depot
            routes = model.run()
        # with cProfile.Profile() as profile:
        #     profile.create_stats()
        #     profile.dump_stats("constant_run_profile.prof")
        # data = pd.read_excel("mdevs/data/mdevs_solutions.xlsx", sheet_name=None)
        # instance_type = "large_BCH"# if "000" not in model.data["label"] else 'large_BCH' 

@cli.command()
# @click.argument("output_path", type=str)
@click.option("--size", type=str, default="50") 
@click.option("--exclude", type=str, default=None)
def run_instances(size: str, exclude: str):
    # for directory in ["data/instances_large/"]:
    for directory in DIRECTORIES:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True) 
        click.echo(json_files)
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if any([s in str(json_file) for s in ["fragments", 'solutions', 'BENCHMARK']]):
                continue        
            if exclude is not None and exclude in str(json_file):
                continue
            if size not in str(json_file):
                continue
            # if '00-04' not in str(json_file):
            #     continue
            print(f"Solving {json_file}...")
            model = NonLinearFragmentGenerator(
                json_file,
                config=NON_LINEAR_BASIC_CONFIG,
                solve_config=SolveConfig(
                    SECOND_OBJECTIVE=Objective.MIN_DEADHEADING,
                    TIME_LIMIT=7200,
                    # TIME_LIMIT=7200 * 12,
                    # MAX_ITERATIONS=5
                )
            )
            sol_file_str =  (
                f"{model.base_dir}/solutions/{model.TYPE}/c-{model.config.UNDISCRETISED_MAX_CHARGE}-{model.data['label']}.json"
            )
            # if os.path.exists(sol_file_str):
            #     print(f"Skipping {model.data['label']}")
            #     continue
            model.run()

@cli.command()
@click.option("--exclude", type=str, default=None)
def run_charge_constrained_instances(exclude: str):
    for directory in ["data/instances_regular/"]:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        # click.echo(json_files)
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if any([s in str(json_file) for s in ["fragments", 'solutions']]):
                continue        
            if exclude is not None and exclude in str(json_file):
                continue
            if "100" not in str(json_file):
                continue
            print(f"Solving {json_file}...")
            for charge_max in range(70, 131, 10):
                # model = NonLinearFragmentGenerator(
                #     json_file,
                #     config=CalculationConfig(
                #         UNDISCRETISED_MAX_CHARGE=charge_max,
                #         RECHARGE_DELAY_IN_MINUTES=0.5,        
                #     ),
                #     solve_config=SolveConfig(
                #         SECOND_OBJECTIVE=Objective.MIN_DEADHEADING,
                #         TIME_LIMIT=3600,
                #         # MAX_ITERATIONS=5
                #     )
                # )
                # model.run()

                # Constant time test for interpolation
                model = InterpolationIP(
                    json_file,
                    charge_calculator_class=ConstantChargeFunction,
                    charge_calculator_kwargs={"discretise": False},
                    config=CONSTANT_TIME_BASIC_CONFIG,
                )
                model.run()

@cli.command()
def run_objective_test():
    # for directory in ["data/instances_regular/"]:
    for directory in ["data/instances_regular/"]:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        # click.echo(json_files)
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if any([s in str(json_file) for s in ["fragments", 'solutions']]):
                continue        
            # if any([s in str(json_file) for s in ["fragments", 'solutions', "05", '09', '10', '06', '07', '03']]):
            #     continue        
            if "200-" not in str(json_file):
                continue
            print(f"Solving {json_file}...")
            for obj in [
                # Objective.MIN_DEADHEADING,
                # Objective.MIN_CHARGE,
                Objective.MIN_VEHICLES,
            ]:
                model = NonLinearFragmentGenerator(
                    json_file,
                    config=NON_LINEAR_BASIC_CONFIG,
                    solve_config=SolveConfig(
                        SECOND_OBJECTIVE=obj,
                        TIME_LIMIT=3600,
                        # MAX_ITERATIONS=5
                    )
                )
                file_str = f"data/instances_regular/solutions/objective_test/{obj.value}/c-{model.config.UNDISCRETISED_MAX_CHARGE}-{model.data['label']}.json"
                # if os.path.exists(file_str):
                #     print(f"Skipping {model.data['label']}")
                #     continue
                model.run(output_override=file_str)

            # data = pd.read_csv(output_path)
            # if model.data["label"] in data["label"].values:
            #     print(f"Skipping {model.data['label']}")
            #     continue
            # data = pd.read_excel("data/mdevs_solutions.xlsx", sheet_name=None)
            # instance_type = "regular" if "000" not in model.data["label"] else 'large' 
            # for sheet in data:
            #     if instance_type in sheet:
            #         data = data[sheet]
            #         break
            # obj = data.query(
            #     f"ID_instance == {model.data['ID']} and battery_charging == 'constant-time'"
            # )["objective_value"].values[0]
            # profiler = cProfile.Profile()
            # assert model.model.objval == obj

@cli.command()
@click.option("--size", type=str, default="50")
def constant_time_single(size: str):
    directory = "mdevs/data/instances_large/"
    directory = "mdevs/data/instances_regular/"
    for directory in ["mdevs/data/instances_regular/", "data/instances_large/"]:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if "fragments" in str(json_file):
                continue        
            if size not in str(json_file):
                continue
            generator = ConstantTimeFragmentGenerator(json_file)
            generator.generate_fragments()#file=str_frag_file)
            generator.generate_timed_network()
            generator.validate_timed_network()
            print("building model...")
            generator.model.setParam("OutputFlag", 0)
            generator.build_model()
            print("solving...")
            profiler = cProfile.Profile()
            with cProfile.Profile() as profile: 
                generator.solve()
            profiler.create_stats()
            profiler.dump_stats(open("solve_profile.prof", "w"))
            profiler.print_stats()
            print("sequencing routes...")
            routes = generator.forward_label()
            print(f"Fragment routes: {len(routes)}")
            generator.validate_solution([r.route_list for r in routes], generator.model.objval)
            with open(f"mdevs/data/results/constant_run/{generator.data['label']}.json", "r") as f:
                assert generator.model.objval == json.load(f)["objective"]
                print(f"instance {generator.data['label']} passed")
                
            # result_json = {
            #     "label": json_file.split("/")[-1].split(".")[0],
            #     "method": "constant_time",
            # } | generator.statistics
            # json.dump(result_json, open(f"data/results/constant_run/{generator.data['label']}.json", "w"))
            pass

@cli.command()
def constant_time_run():
    directory = "mdevs/data/instances_large/"
    directory = "mdevs/data/instances_regular/"
    # for directory in ["data/instances_regular/", "data/instances_large/"]:
    for directory in DIRECTORIES:
    # [
    #     "data/instances_constrained/",
    #     "data/instances_num_of_depots_experim",
    #     ]:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if any([s in str(json_file) for s in ["fragments", 'solutions']]):
                continue        
            print(json_file)
            generator = ConstantTimeFragmentGenerator(json_file)
            routes = generator.run()
            # print(f"Fragment routes: {len(routes)}")
            # generator.validate_solution([r.route_list for r in routes], generator.model.objval)
@cli.command()
def constant_time_battery_constrained_run():
    directory = "mdevs/data/instances_large/"
    directory = "mdevs/data/instances_regular/"
    # for directory in ["data/instances_regular/", "data/instances_large/"]:
    for directory in [
        # "data/instances_constrained/",
        'data/instances_regular/',
        ]:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if any([s in str(json_file) for s in ["fragments", 'solutions']]):
                continue        
            for json_file in json_files:     
                if any([s in str(json_file) for s in ["fragments", 'solutions']]):
                    continue        
                if "100" not in str(json_file):
                    continue
                print(f"Solving {json_file}...")
                for charge_max in range(70, 131, 10):
                    model = ConstantTimeFragmentGenerator(
                        json_file,
                        config=CalculationConfig(
                            UNDISCRETISED_MAX_CHARGE=charge_max,
                            RECHARGE_DELAY_IN_MINUTES=0.5,        
                        )
                    )
                    model.run()
            # print(f"Fragment routes: {len(routes)}")
            # generator.validate_solution([r.route_list for r in routes], generator.model.objval)

@cli.command()
def constant_time_interpolation_run():
    directory = "mdevs/data/instances_large/"
    directory = "mdevs/data/instances_regular/"
    # for directory in [
    # "data/instances_regular/", "data/instances_large/"]:
    for directory in [
        # "data/instances_regular/",
        # "data/instances_constrained/",
        # "data/instances_num_of_depots_experim",
        "data/instances_large/",
        ]:
        # Use glob to match the pattern '**/*.json', which will find .json files in the specified directory and all its subdirectories
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        EXCLUDED_INSTANCES = ["instances_regular/I-5-5-200-06.json", "I-5-5-200-10.json"]
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:     
            if any([s in str(json_file) for s in [
                "fragments", 'solutions',
                '3000', '4000'
                ]]):
                continue        
            print(json_file)
            sol_file_str =  (
                f"{directory}solutions/interpolation-constant/c-{CONSTANT_TIME_BASIC_CONFIG.UNDISCRETISED_MAX_CHARGE}-{json_file.split('/')[-1]}"
            )
            print(sol_file_str)
            if os.path.exists(sol_file_str):
                print(f"Skipping")
                continue
            model = InterpolationIP(
                json_file,
                charge_calculator_class=ConstantChargeFunction,
                charge_calculator_kwargs={"discretise": False},
                config=CONSTANT_TIME_BASIC_CONFIG
            )
            routes = model.run()
            # print(f"Fragment routes: {len(routes)}")
            # generator.validate_solution([r.route_list for r in routes], generator.model.objval)

@cli.command()
def solve_models_without_charge():
    """Solves each model without considering charge constraints"""
    for directory in DIRECTORIES:
        json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        # Iterate over the list of filepaths & open each file
        for json_file in json_files:
            if any([s in str(json_file) for s in ["fragments", 'solutions']]):
                continue        
            print(json_file)
            if "2000-04" not in json_file:
                continue
            generator = NoChargeIP(
                json_file,
                charge_calculator_class=NoChargeFunction,
                charge_calculator_kwargs={"discretise": False},
                config=NON_LINEAR_BASIC_CONFIG
            )
            routes = generator.run()
                # print(f"Fragment routes: {len(routes)}")
                # generator.validate_solution([r.route_list for r in routes], generator.model.objval)

            # print(f"Fragment routes: {len(routes)}")
            # generator.validate_solution([r.route_list for r in routes], generator.model.objval)


cli.add_command(constant_time_single)
cli.add_command(constant_time_run)
cli.add_command(non_linear_debug)
cli.add_command(run_charge_constrained_instances)
# cli.add_command(constant_time_naive_ip_run)
cli.add_command(run_objective_test)
cli.add_command(non_linear_debug)
cli.add_command(solve_models_without_charge)


if __name__ == "__main__":
    # constant_time_single("50")
    # non_linear_debug()
    # interpolation_debug()
    # run_objective_test()
    # run_charge_constrained_instances()
    cli()
    # run_non_linear_regular_instances("data/results/non_linear_run/s.csv", "1000", None)
    # constant_time_debug()
    # constant_time_naive_ip_run()
    # compare_sequencing_procedures()
    # non_linear_debug()