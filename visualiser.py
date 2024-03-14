import plotly.express as px
import json
import pandas as pd
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import Fragment, TimedDepot, TimedFragment, Flow, Route
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
) -> None:
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
        
        # annotate the flow
        fig.add_annotation(
            x=(end.time + start.time) / 2,
            y=(end.id + start.id) / 2,
            text=arc["flow"],
            showarrow=True,
            arrowhead=1,
            arrowcolor="black",
        )

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
    fig.show()


    pass
# def visualise_timed_network(
#     timed_nodes_by_depot: dict[int, list[TimedDepot]],
#     fragments: set[Fragment],
#     waiting_arcs: set[tuple[TimedDepot, TimedDepot]],
#     fragments_by_timed_node: tuple[TimedDepot, TimedDepot],
# ) -> None:
#     nodes = set()
#     node_id_lookup = {}
#     idx = 0
#     for t in timed_nodes_by_depot:
#         for n in timed_nodes_by_depot[t]:
#             nodes.add((idx, (n.time, n.id)))
#             node_id_lookup[n] = idx
#             idx += 1

#     arcs = set()
#     for f in fragments:
#         start_fragment = TimedFragment(
#             time=f.start_time, id=f.id, direction=Flow.DEPARTURE
#         )
#         end_fragment = TimedFragment(time=f.end_time, id=f.id, direction=Flow.ARRIVAL)
#         for td in timed_nodes_by_depot[f.start_depot_id]:
#             if start_fragment in fragments_by_timed_node[td]:
#                 start_node = td
#                 break
#         else:
#             raise ValueError(
#                 f"fragment {f.start_time, f.start_depot_id} not found in start timed nodes"
#             )

#         for td in timed_nodes_by_depot[f.end_depot_id]:
#             if end_fragment in fragments_by_timed_node[td]:
#                 end_node = td
#                 break
#         else:
#             raise ValueError(
#                 f"fragment {f.start_time, f.start_depot_id} not found in end timed nodes"
#             )

#         arcs.add((node_id_lookup[start_node], node_id_lookup[end_node]))

#     vehicle_arc_ids = [(a[0], a[1]) for a in arcs]
#     waiting_arc_with_weight_ids = [
#         (node_id_lookup[a[0]], node_id_lookup[a[1]], a[2]) for a in waiting_arcs
#     ]
#     waiting_arc_ids = [
#         (node_id_lookup[a[0]], node_id_lookup[a[1]]) for a in waiting_arcs
#     ]
#     g = nx.DiGraph()
#     g.add_nodes_from((id, {"pos": pos}) for id, pos in nodes)

#     # vehicle_nodes = set(i for arc in vehicle_arc_ids for i in arc)
#     # drone_nodes = set(i for arc in waiting_arcs for i in arc)
#     # unvisited = set(g.nodes) - vehicle_nodes - drone_nodes

#     mpl.rcParams["savefig.bbox"] = "tight"
#     pos = nx.get_node_attributes(g, "pos")
#     fig = plt.figure()
#     cmap = mpl.colormaps["tab20c"]
#     tc = 1
#     dc = 9
#     uc = 17
#     ns = 15
#     ax = plt.subplot(111)
#     # nodes_noid = set(i for arc in nodes for i in arc)
#     # Drone, Truck, and unvisited nodes
#     g.add_edges_from(
#         (
#             (
#                 i,
#                 j,
#             )
#             for i, j in arcs
#         )
#     )
#     for i, j, weight in waiting_arc_with_weight_ids:
#         g.add_edge(i, j, weight=weight)

#     nx.draw_networkx_nodes(
#         g,
#         pos,
#         g.nodes,
#         ax=ax,
#         node_color=[cmap(dc)],
#         edgecolors=[cmap(dc - 1)],
#         node_size=ns,
#         label="Timed Depots",
#     )

#     edge_labels = nx.get_edge_attributes(g, "weight")
#     nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

#     nx.draw_networkx_edges(
#         g,
#         pos,
#         vehicle_arc_ids,
#         ax=ax,
#         edge_color=[cmap(tc - 1)],
#         label="Vehicle fragments",
#         width=0.5,
#         connectionstyle="arc3, rad = 0.1",
#     )
#     nx.draw_networkx_edges(
#         g,
#         pos,
#         waiting_arc_ids,
#         ax=ax,
#         edge_color=[cmap(dc - 1)],
#         label="Waiting arcs",
#         width=0.5,
#     )
#     # ax.axis("off")
#     nx.draw(
#         g,
#         pos,
#         with_labels=True,
#     )

#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
#     ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
#     plt.show()


if __name__ == "__main__":
    data = json.load(open("data/instances_regular/I-1-1-50-01.json"))
    # visualise_map(data)
