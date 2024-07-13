import plotly.express as px
import json
import pandas as pd
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import TimedDepot
from plotly import graph_objects as go, express as px

def visualise_map(data: dict):
    print((COLUMNS := data["jobs"]["0"]))
    print(data["jobs"])
    df = pd.DataFrame(data["jobs"].values(), columns=COLUMNS)
    print(data["buildings"])
    location_by_id = {
        int(i): {
            "x": building["entrance"][0],
            "y": building["entrance"][1],
            "type": building["type"],
        }
        for i, building in data["buildings"].items()
    }
    df["start_location_x"] = df.apply(
        lambda x: x["start_location"][0] + location_by_id[x["building_start_id"]]["x"],
        axis=1,
    )
    df["start_location_y"] = df.apply(
        lambda x: x["start_location"][1] + location_by_id[x["building_start_id"]]["y"],
        axis=1,
    )
    df["end_location_x"] = df.apply(
        lambda x: x["end_location"][0] + location_by_id[x["building_end_id"]]["x"],
        axis=1,
    )
    df["end_location_y"] = df.apply(
        lambda x: x["end_location"][1] + location_by_id[x["building_end_id"]]["y"],
        axis=1,
    )
    loc_x = df["start_location_x"].to_list() + df["end_location_x"].to_list()
    loc_y = df["start_location_y"].to_list() + df["end_location_y"].to_list()
    print(loc_x, loc_y)
    px.scatter(x=loc_x, y=loc_y, title="Job start and end locations").show()
    # px.scatter(df, x="start_location_x", y="start_location_y", color="building_start_id", title="Job start locations and building ID").show()
    # px.scatter(df, x="end_location_x", y="end_location_y", color="building_end_id", title="Job end locations and building ID").show()


def visualise_timed_network(
    timed_depots: list[TimedDepot],
    vehicle_arcs: list[tuple[TimedDepot, TimedDepot]],
    waiting_arcs: list[dict[str, int | tuple[TimedDepot, TimedDepot]]],
    **kwargs
) -> go.Figure:
    fig = go.Figure()
    graph_arc_x = []
    graph_arc_y = []
    for start, end in vehicle_arcs:
        # fig.add_annotation(
        #     x=end.time,
        #     y=end.id,
        #     ax=start.time,
        #     ay=start.id,
        #     xref='x',
        #     yref='y',
        #     axref='x',
        #     ayref='y',
        #     showarrow=True,
        #     arrowhead=2,
        #     arrowsize=0.5,
        #     arrowwidth=2,
        #     arrowcolor='#636363'
        # )
        if start.id == end.id:
            arc_x = (start.time, (start.time+end.time)/2, end.time)
            arc_y = (start.id, start.id + 0.5, end.id)
        else:
            arc_x = (start.time, end.time)
            arc_y = (start.id, end.id)
        graph_arc_x.extend([*arc_x,  None])
        graph_arc_y.extend([*arc_y, None])

    fig.add_trace(
        go.Scatter(
            x=graph_arc_x,
            y=graph_arc_y,
            mode="lines",
            marker=dict(size=10, color="blue"),
            line=dict(width=0.5),#, color='#888'),
            name="Vehicle Arcs",
        )
    )



    waiting_arc_x = []
    waiting_arc_y = []

    for arc in waiting_arcs:
        start, end = arc["timed_depot"]
        waiting_arc_x.extend([start.time, end.time, None])
        waiting_arc_y.extend([start.id, end.id, None])
        
        # # annotate the flow
        # fig.add_annotation(
        #     x=(end.time + start.time) / 2,
        #     y=(end.id + start.id) / 2,
        #     # text=arc.get("flow", None),
        #     showarrow=True,
        #     arrowhead=1,
        #     arrowcolor="black",
        # )

    fig.add_trace(
        go.Scatter(
            x=waiting_arc_x,
            y=waiting_arc_y,
            mode="lines",
            # line=dict(width=0.5, color='#888'),
            # marker=dict(size=10, color="red"),
            name="Waiting Arcs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[t.time for t in timed_depots],
            y=[t.id for t in timed_depots],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="Timed Depots",
        )
    )
    # Add a title
    fig.update_layout(title_text=f"Timed Network for {kwargs.get('instance_label', '##')}, type: {kwargs.get('charge_type', 'N/A')}")
    # fig.show()

    return fig
    pass

def visualise_routes(
        routes: list[TimedDepot | tuple[TimedDepot, TimedDepot]], 
        timed_depots: list[TimedDepot],
    ) -> go.Figure:
    fig = go.Figure()
    # Choose len(routes) colors from the color palette
    colors = px.colors.qualitative.Alphabet

    for i, route in enumerate(routes):
        route = [td for td in route if isinstance(td, TimedDepot)]
        # timed_depots = [td for td in routes if isinstance(td, TimedDepot)]
        # vehicle_arcs = [td for td in routes if isinstance(td, tuple)]
        route_path_x = [td.time for td in route]
        route_path_y = [td.id for td in route]
        fig.add_trace(
            go.Scatter(
                x=route_path_x,
                y=route_path_y,
                mode="lines+markers",
                marker=dict(size=10, color="blue"),
                line=dict(width=0.5, color=colors[i]),
                name=f"Route {i}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[t.time for t in timed_depots],
            y=[t.id for t in timed_depots],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="Timed Depots",
        )
    )
    
    return fig

if __name__ == "__main__":
    data = json.load(open("data/instances_regular/I-1-1-50-01.json"))
    # visualise_map(data)
