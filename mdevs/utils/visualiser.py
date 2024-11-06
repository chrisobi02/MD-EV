import plotly.express as px
import json
import pandas as pd
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly import graph_objects as go, express as px

from mdevs.formulations.base import *

def visualise_map(data: dict):
    print((COLUMNS := data["jobs"]))
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
    timed_depots: list[ChargeDepot],
    vehicle_arcs: list[tuple[ChargeDepot, ChargeDepot]],
    waiting_arcs: list[dict[str, int | tuple[ChargeDepot, ChargeDepot]]],
    fig=go.Figure(),
    **kwargs
) -> go.Figure:
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
        routes: list[ChargeDepot | tuple[ChargeDepot, ChargeDepot]], 
        timed_depots: list[ChargeDepot],
    ) -> go.Figure:
    fig = go.Figure()
    # Choose len(routes) colors from the color palette
    colors = px.colors.qualitative.Alphabet

    for i, route in enumerate(routes):
        route = [td for td in route if isinstance(td, ChargeDepot)]
        # timed_depots = [td for td in routes if isinstance(td, ChargeDepot)]
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

def visualise_charge_network(
    charge_depots: list[ChargeDepot],
    vehicle_arcs: list[tuple[ChargeDepot, ChargeDepot]],
    waiting_arcs: set[ChargeDepotStore],
    iter: int,
    max_charge: int,
    graph_type="",
    iter_code="",
    **kwargs
) -> go.Figure:
    fig_data = []
    graph_labels = dict(
        xaxis=dict(title='Time'),
        yaxis=dict(title='Charge'),
    )
    node_kwarg_by_type = {
        "Solution": dict(size=2, color="blue"),
        "Repair": dict(size=2, color="red"),
    }
    # For non-gif visualisation
    # node_kwarg_by_type = {
    #     "Solution": dict(size=10, color="blue"),
    #     "Repair": dict(size=10, color="red"),
    # }
    arc_kwarg_by_type = {
        "Solution": dict(width=0.5, color='blue'),
        "Repair": dict(width=0.5, color='red')
    }
    recharge_kwarg_by_type = arc_kwarg_by_type
    # {
    #     "Solution": dict(width=0.5, color="red"),
    #     "Repair": dict(width=0.5, color='orange')
    # }
    graph_arc_x = []
    graph_arc_y = []

    for start, end in vehicle_arcs:
        if start.id == end.id and start.charge == end.charge:
            arc_x = (start.time, (start.time + end.time) / 2, end.time)
            arc_y = (start.charge, min((3 * start.charge + end.charge) / 2, int(max_charge*1.15)), end.charge)
        else:
            arc_x = (start.time, end.time)
            arc_y = (start.charge, end.charge)
        graph_arc_x.extend([*arc_x, None])
        graph_arc_y.extend([*arc_y, None])

    fig_data.append(
        go.Scatter(
            x=graph_arc_x,
            y=graph_arc_y,
            mode="lines",
            marker=node_kwarg_by_type[graph_type],
            line=arc_kwarg_by_type[graph_type],
            name=f"{graph_type} Vehicle Arcs",
            **kwargs
        )
    )

    waiting_arc_x = []
    waiting_arc_y = []

    for arc in waiting_arcs:
        start, end = arc.start, arc.end
        waiting_arc_x.extend([start.time, end.time, None])
        waiting_arc_y.extend([start.charge, end.charge, None])

    fig_data.append(
        go.Scatter(
            x=waiting_arc_x,
            y=waiting_arc_y,
            mode="lines",
            line=recharge_kwarg_by_type[graph_type],
            marker=node_kwarg_by_type[graph_type],
            name=f"{graph_type} Waiting Arcs",
            **kwargs
        )
    )
    kwargs['showlegend'] = True
    fig_data.append(
        go.Scatter(
            x=[t.time for t in charge_depots],
            y=[t.charge for t in charge_depots],
            mode="markers",
            marker=node_kwarg_by_type[graph_type],
            # name=f"{graph_type} Timed Depots",
            name=f"{'Added' if graph_type == 'Repair' else 'Solution/Existing'}",
            **kwargs,
        )
    )
    
    # for start, end in vehicle_arcs:
    #     if start.id == end.id and start.charge == end.charge:
    #         arc_x = (start.time, (start.time + end.time) / 2, end.time)
    #         arc_y = (start.charge, min((3 * start.charge + end.charge) / 2, int(max_charge*1.25)), end.charge)
    #     else:
    #         arc_x = (start.time, end.time)
    #         arc_y = (start.charge, end.charge)
    #     graph_arc_x.extend([*arc_x, None])
    #     graph_arc_y.extend([*arc_y, None])

    # fig_data.append({
    #     "x": graph_arc_x,
    #     "y": graph_arc_y,
    #     "mode": "lines",
    #     "marker": node_kwarg_by_type[graph_type],
    #     "line": arc_kwarg_by_type[graph_type],
    #     "name": f"{graph_type} Vehicle Arcs",
    #     **kwargs
    # })

    # waiting_arc_x = []
    # waiting_arc_y = []

    # for arc in waiting_arcs:
    #     start, end = arc.start, arc.end
    #     waiting_arc_x.extend([start.time, end.time, None])
    #     waiting_arc_y.extend([start.charge, end.charge, None])

    # fig_data.append({
    #     "x": waiting_arc_x,
    #     "y": waiting_arc_y,
    #     "mode": "lines",
    #     "line": recharge_kwarg_by_type[graph_type],
    #     "marker": node_kwarg_by_type[graph_type],
    #     "name": f"{graph_type} Waiting Arcs",
    # })

    # fig_data.append({
    #     "x": [t.time for t in charge_depots],
    #     "y": [t.charge for t in charge_depots],
    #     "mode": "markers",
    #     "marker": node_kwarg_by_type[graph_type],
    #     "name": f"{graph_type} Timed Depots",
    #     **kwargs,
    # })

    # Add a title
    # layout = go.Layout(
    #     xaxis=dict(title='Time'),
    #     yaxis=dict(title='Charge'),
    #     title_text=f"Charge Network for {kwargs.get('instance_label', '##')}, type: {kwargs.get('charge_type', 'N/A')}",
    # )

    frame = go.Frame(data=fig_data, name=f"Iteration {iter}", layout=dict(title_text=iter_code))
    return frame

def visualise_network_transformation(fig: go.Figure) -> go.Figure:
    """
    Generates a visualisation of the network over multiple iterations.
    Each iteration is based on the visualise_charge_network function.
    Each iteration is a frame in the slider.
    """
    TRACES_PER_ITER = 6 # waiting arcs, fragment arcs, depots *2 for the ones added.
    increments = []
    for i in range(0, len(fig.data), TRACES_PER_ITER):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Iteration {i//TRACES_PER_ITER}",
        )
        for j in range(TRACES_PER_ITER):
            step["args"][0]["visible"][i + j] = True
        increments.append(step)

    sliders = [dict(
        active=10,
        # currentvalue={"prefix": "Trials: "},
        pad={"t": 50},
        steps=increments
    )]

    fig.update_layout(
        sliders=sliders
    )
    return fig

# def visualise_charge_network(
#     charge_depots: list[ChargeDepot],
#     vehicle_arcs: list[tuple[ChargeDepot, ChargeDepot]],
#     waiting_arcs: set[ChargeDepotStore],
#     fig_data=go.Figure(),
#     graph_type="",
#     **kwargs
# ) -> go.Figure:
#     fig_data = []

#     graph_labels=dict(
#         xaxis=dict(title='Time'),
#         yaxis=dict(title='ID'),
#         zaxis=dict(title='Charge'),
#     )
#     node_kwarg_by_type = {
#         "Solution": dict(size=10, color="blue"),
#         "Repair": dict(size=10, color="green"),
#     }
#     arc_kwarg_by_type = {
#         "Solution": dict(width=0.5, color='#888'),
#         "Repair": dict(width=0.5, color='black')
#     }
#     recharge_kwarg_by_type = {
#         "Solution": dict(width=0.5, color="red"),
#         "Repair": dict(width=0.5, color='orange')
#     }
#     graph_arc_x = []
#     graph_arc_y = []
#     graph_arc_z = []

#     for start, end in vehicle_arcs:
#         if start.id == end.id and start.charge == end.charge:
#             arc_x = (start.time, (start.time+end.time)/2, end.time)
#             arc_y = (start.id, start.id + 0.5, end.id)
#             arc_z = (start.charge, (3*start.charge+end.charge)/2, end.charge)
#         else:
#             arc_x = (start.time, end.time)
#             arc_y = (start.id, end.id)
#             arc_z = (start.charge, end.charge)
#         graph_arc_x.extend([*arc_x,  None])
#         graph_arc_y.extend([*arc_y, None])
#         graph_arc_z.extend([*arc_z, None])

#     fig_data.append(
#         go.Scatter3d(
#             x=graph_arc_x,
#             y=graph_arc_y,
#             z=graph_arc_z,
#             mode="lines",
#             marker=node_kwarg_by_type[graph_type],
#             line=arc_kwarg_by_type[graph_type],
#             name=f"{graph_type} Vehicle Arcs",
#             **kwargs
#         )
#     )

#     waiting_arc_x = []
#     waiting_arc_y = []
#     waiting_arc_z = []

#     for arc in waiting_arcs:
#         start, end = arc.start, arc.end
#         waiting_arc_x.extend([start.time, end.time, None])
#         waiting_arc_y.extend([start.id, end.id, None])
#         waiting_arc_z.extend([start.charge, end.charge, None])

#     fig_data.append(
#         go.Scatter3d(
#             x=waiting_arc_x,
#             y=waiting_arc_y,
#             z=waiting_arc_z,
#             mode="lines",
#             line=recharge_kwarg_by_type[graph_type],
#             marker=node_kwarg_by_type[graph_type],
#             name=f"{graph_type} Waiting Arcs",
         
#         )
#     )
#     fig_data.append(
#         go.Scatter3d(
#             x=[t.time for t in charge_depots],
#             y=[t.id for t in charge_depots],
#             z=[t.charge for t in charge_depots],
#             mode="markers",
#             marker=node_kwarg_by_type[graph_type],
#             name=f"{graph_type} Timed Depots",
#             **kwargs,
#         )
#     )

    # # New vehicle arcs
    # new_graph_arc_x = []
    # new_graph_arc_y = []
    # new_graph_arc_z = []

    # for start, end in new_vehicle_arcs:
    #     if start.id == end.id and start.charge == end.charge:
    #         arc_x = (start.time, (start.time + end.time) / 2, end.time)
    #         arc_y = (start.id, start.id + 0.5, end.id)
    #         arc_z = (start.charge, (3 * start.charge + end.charge) / 2, end.charge)
    #     else:
    #         arc_x = (start.time, end.time)
    #         arc_y = (start.id, end.id)
    #         arc_z = (start.charge, end.charge)
    #     new_graph_arc_x.extend([*arc_x, None])
    #     new_graph_arc_y.extend([*arc_y, None])
    #     new_graph_arc_z.extend([*arc_z, None])

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=new_graph_arc_x,
    #         y=new_graph_arc_y,
    #         z=new_graph_arc_z,
    #         mode="lines",
    #         marker=dict(size=10, color="purple"),
    #         line=dict(width=0.5),
    #         name="New Vehicle Arcs",
    #         **kwargs
    #     )
    # )

    # # New waiting arcs
    # new_waiting_arc_x = []
    # new_waiting_arc_y = []
    # new_waiting_arc_z = []

    # for arc in new_waiting_arcs:
    #     start, end = arc.start, arc.end
    #     new_waiting_arc_x.extend([start.time, end.time, None])
    #     new_waiting_arc_y.extend([start.id, end.id, None])
    #     new_waiting_arc_z.extend([start.charge, end.charge, None])

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=new_waiting_arc_x,
    #         y=new_waiting_arc_y,
    #         z=new_waiting_arc_z,
    #         mode="lines",
    #         name="New Waiting Arcs",
    #     )
    # )

    # # New charge depots
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=[t.time for t in new_charge_depots],
    #         y=[t.id for t in new_charge_depots],
    #         z=[t.charge for t in new_charge_depots],
    #         mode="markers",
    #         marker=dict(size=10, color="orange"),
    #         name="New Timed Depots",
    #         **kwargs,
    #     )
    # )

#     # Add a title
#     fig_data.update_layout(
#         go.Layout(
#             scene=dict(
#                 xaxis=dict(title='Time'),
#                 yaxis=dict(title='ID'),
#                 zaxis=dict(title='Charge'),
#             ),
#             title_text=f"Charge Network for {kwargs.get('instance_label', '##')}, type: {kwargs.get('charge_type', 'N/A')}", 
#         )
#     )
#     # fig.show()
#     return fig_data

# def visualise_network_transformation(fig: go.Figure) -> go.Figure:
#     """
#     Generates a visualisation of the network over multiple iterations.
#     Each iteration is based on the visualise_charge_network function.
#     Each iteration is a frame in the slider.
#     """
#     TRACES_PER_ITER = 6 # waiting arcs, fragment arcs, depots *2 for the ones added.
#     increments = []
#     for i in range(0, len(fig.data), TRACES_PER_ITER):
#         step = dict(
#             method="update",
#             args=[{"visible": [False] * len(fig.data)}],
#             label=f"Iteration {i//TRACES_PER_ITER}",
#         )
#         for j in range(TRACES_PER_ITER):
#             step["args"][0]["visible"][i + j] = True
#         increments.append(step)

#     sliders = [dict(
#         active=10,
#         # currentvalue={"prefix": "Trials: "},
#         pad={"t": 50},
#         steps=increments
#     )]

#     fig.update_layout(
#         sliders=sliders
#     )
#     return fig

def animate_network_transformation(frames: list[go.Frame], yaxis_range: list[int], lp_iter_cuttoff: int, animation_seconds=10) -> go.Figure:
    """
    Generates an animation of the network over multiple iterations.
    Each iteration is based on the visualise_charge_network function.
    Each iteration is a frame in the slider.
    """
    # fig.add_trace(frames[0].data[0])
    # for i, frame in enumerate(fig.frames):
    #     step = dict(
    #         method="animate",
    #         args=[
    #             {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "quadratic-in-out"}},
    #             # {"visible": [False] * len(fig.data)}
    #         ],
    #         label=f"Iteration {i}",
    #     )
    #     # for j in range(TRACES_PER_ITER):
    #     #     step["args"][0]["visible"][i + j] = True
    #     increments.append(step)
    fig = go.Figure(frames=frames)
    for trace in frames[0].data:
        fig.add_trace(trace)
    duration = 1000
    frame_duration = animation_seconds * 1000 / len(frames)
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    sliders = [
        {
            "args": [
                [f.name],
                {"frame": {"duration": duration, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": duration}},
            ],
            "label": f"Iter {k+1} - {'LP' if k+1 <= lp_iter_cuttoff else 'MIP'}",
            "method": "animate",
        }
        for k, f in enumerate(fig.frames)
    ]
    sliders_dict["steps"] = sliders
    fig.update_layout(
        sliders=[sliders_dict],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": False},
                                        "fromcurrent": False, "transition": {"duration": 0,
                                                                            "easing": "linear"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": duration, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": duration, "easing": "linear"}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": True,
                # 'prefix': "Iteration: ",
                "type": "buttons",
                "x": 0.05,
                "xanchor": "right",
                "y": 0.1,
                "yanchor": "top"
            }
        ],
        yaxis=dict(range=yaxis_range),
        showlegend=False,   
    )
    fig.show()
    return fig

def write_network_transformation(frames: list[go.Frame], yaxis_range: list[int], lp_iter_cuttoff: int, output_folder: str) -> go.Figure:
    """
    Generates an animation of the network over multiple iterations.
    Each iteration is based on the visualise_charge_network function.
    Each iteration is a frame in the slider.
    """
    for i, frame in enumerate(frames):
        fig = go.Figure(data = frame.data, layout=frame.layout)
        fig.update_layout(
            # title_text=f"Iteration {i+1} - {'LP' if i+1 <= lp_iter_cuttoff else 'MIP'}", 
            # showlegend=False,
            xaxis=dict(title='Time'),
            yaxis=dict(title='Charge', range=yaxis_range),
        )
        fig.write_image(f"{output_folder}/iteration_{i+1}.png")


if __name__ == "__main__":
    data = json.load(open("data/instances_regular/I-1-1-50-01.json"))
    # visualise_map(data)
