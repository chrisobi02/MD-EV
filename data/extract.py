import pandas as pd
import os
import json
import ast
INSTANCE_SHEETS = [
    "instances_constrained",
    "instances_large",
    "instances_num_of_depots_experim",
    "instances_regular",
]
data: dict[str, pd.DataFrame] = pd.read_excel("data/mdevs_data.xlsx", sheet_name=INSTANCE_SHEETS)

for instance_type in data:
    path = f"data/{instance_type}"
    print(path)
    if not os.path.exists(path):
        print("deone")
        os.mkdir(path)

    for i, instance in data[instance_type].iterrows():
        json_data = {
            "ID": instance["ID"],
            "label": instance["label"],
            "num_jobs": instance["n"],
            "buildings": {},
            "jobs": None
        }
        try:
            maximum_per_depot = ast.literal_eval(instance["m_d"])             
            job_start_time = ast.literal_eval(instance["s_j"])
            job_end_time = ast.literal_eval(instance["e_j"])
            job_charge = ast.literal_eval(instance["c_j"])
            job_start_locations = [ast.literal_eval(start) for start in instance["tripStartLocations"].split(";")]
            job_end_locations = [ast.literal_eval(end) for end in instance["tripEndLocations"].split(";")]
            depot_location = [ast.literal_eval(depot) for depot in instance["depotLocations"].split(";")]
            station_location = [ast.literal_eval(station) for station in instance["stationLocations"].split(";")]
        except:
            print(f"{instance['label']} failed due to truncated data")
        
        if not len(job_start_time) == len(job_end_time) == len(job_charge) == len(job_start_locations) == len(job_end_locations):
            print(f"{instance['label']} failed due to mismatched data dimensions")
            print(len(job_start_time), len(job_end_time), len(job_charge), len(job_start_locations), len(job_end_locations))
            continue

        print(instance["label"], instance["n"], len(job_start_time), len(job_end_time), len(job_charge), len(job_start_locations), len(job_end_locations))
        buildings = [] # Abstract buildings into a list so coordiantes aren't stored on the job level
        depots = []
        for building in depot_location:
            if not any(build == building[:2] for build in buildings):
                buildings.append(building[:2])
                depots.append(building)
        
        stations = []
        for building in station_location:
            if not any(build == building[:2] for build in buildings):
                buildings.append(building[:2])
                stations.append(building[:2])

        # Extract jobs into a JSON-applicable format
        jobs = []
        for i in range(len(job_start_time)):
            # print(len(job_start_locations[i]), len(job_end_locations[i]))
            job_start_entrance, job_start_location = tuple(job_start_locations[i][:2]), tuple(job_start_locations[i][2:])
            job_end_entrance, job_end_location = tuple(job_end_locations[i][:2]), tuple(job_end_locations[i][2:])

            for entrance in [job_start_entrance, job_end_entrance]:
                if not any(build == entrance for build in buildings):
                    buildings.append(entrance)
            job_start_building_id = buildings.index(job_start_entrance)
            job_end_building_id = buildings.index(job_end_entrance)

            jobs.append(
                {   
                "id": i,
                "start_time": job_start_time[i],
                "end_time": job_end_time[i],
                "charge": job_charge[i],
                "building_start_id": job_start_building_id,
                "building_end_id": job_end_building_id,
                "start_location": job_start_location,
                "end_location": job_end_location
                }
                )
        building_json = []
        
        depots_just_entrance = [depot[:2] for depot in depots]
        for i, building in enumerate(buildings):
            building_kwargs = {
                "entrance": building,
                "id": i,
                "capacity": None,
                "location": None,
                }
            if building in depots_just_entrance:
                building_kwargs["type"] = "depot"
                building_kwargs["capacity"] = maximum_per_depot[depots_just_entrance.index(building)]
                building_kwargs["location"] = depots[depots_just_entrance.index(building)][2:]
            elif building in stations:
                building_kwargs["type"] = "station"
            else:
                building_kwargs["type"] = "workstation"

            building_json.append(building_kwargs)

        json_data["buildings"] = building_json
        json_data["jobs"] = jobs
        with open(f"{path}/{instance['label']}.json", "w") as f:
            json.dump(json_data, f)