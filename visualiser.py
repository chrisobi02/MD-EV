import plotly.express as px
import json
import pandas as pd

def visualise(data: dict):
    print((COLUMNS:= data["jobs"]["0"]))
    print(data["jobs"])
    df = pd.DataFrame(data["jobs"].values(), columns = COLUMNS)
    print(data["buildings"])
    location_by_id = {
        int(i): {
            "x": building["entrance"][0],
            "y": building["entrance"][1],
            "type": building["type"]
        } 
        for i, building in data["buildings"].items()}
    df["start_location_x"] = df.apply(lambda x: x["start_location"][0] + location_by_id[x["building_start_id"]]["x"], axis=1)
    df["start_location_y"] = df.apply(lambda x: x["start_location"][1] + location_by_id[x["building_start_id"]]["y"], axis=1)
    df["end_location_x"] = df.apply(lambda x: x["end_location"][0] + location_by_id[x["building_end_id"]]["x"], axis=1)
    df["end_location_y"] = df.apply(lambda x: x["end_location"][1]  + location_by_id[x["building_end_id"]]["y"], axis=1)
    loc_x = df["start_location_x"].to_list() + df["end_location_x"].to_list()
    loc_y = df["start_location_y"].to_list() + df["end_location_y"].to_list()
    print(loc_x, loc_y)
    px.scatter(x=loc_x, y=loc_y, title="Job start and end locations").show()
    # px.scatter(df, x="start_location_x", y="start_location_y", color="building_start_id", title="Job start locations and building ID").show()
    # px.scatter(df, x="end_location_x", y="end_location_y", color="building_end_id", title="Job end locations and building ID").show()
if __name__ == "__main__":
    data = json.load(open("data/instances_regular/I-1-1-50-01.json"))
    visualise(data)