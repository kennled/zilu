import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000
node_df = pd.read_csv('output.csv')

# Create a subset of nodes with max_EP_lu values 'RESIDENTIAL' and 'COMMERCIAL'
source_nodes = node_df[node_df['max_EP_lu'] == 'RESIDENTIAL']['LinkID'].tolist()
target_nodes = node_df[(node_df['max_EP_lu'] == 'COMMERCIAL')]['LinkID'].tolist()

# Create a new empty graph
G = nx.DiGraph()

# Add all nodes to the graph with their attributes
for _, row in node_df.iterrows():
    G.add_node(row['LinkID'], EP_R=row['EP_RESIDENTIAL'], EP_C=row['EP_COMMERCIAL'], latitude=row['latitude'], longitude=row['longitude'], EP=row['max_EP'])

# Iterate over the rows in the node_df data frame
for _, row in node_df.iterrows():
    # Check if the end point of a link matches the start point of another link
    next_links = node_df[node_df['StartPoint'] == row['EndPoint']]
    for _, next_row in next_links.iterrows():
        # Check if the start point of the next link equals the end point of the current link and vice versa
        if not ((row['StartPoint'] == next_row['EndPoint']) and (row['EndPoint'] == next_row['StartPoint'])):
            # Calculate the distance between the two nodes using the haversine function
            distance = haversine(row['longitude'], row['latitude'], next_row['longitude'], next_row['latitude'])
            # Add an edge between the two nodes with the weight set to the distance between them
            G.add_edge(next_row['LinkID'], row['LinkID'], weight=distance)
            
for node in G.nodes():
    G.nodes[node]['my_betweenness'] = 0
SUM=0
no_path_count=0
for source in G.nodes():
    for target in G.nodes():
        if source == target:
            continue
        try:
            path = nx.shortest_path(G, source=source, target=target, weight='weight')
            for node in path:
                G.nodes[node]['my_betweenness'] += G.nodes[source]['EP_R'] * G.nodes[target]['EP_C']
                SUM+=G.nodes[source]['EP_R'] * G.nodes[target]['EP_C']
        except nx.NetworkXNoPath:
            no_path_count += 1
            continue
G.nodes[node]['my_betweenness'] = G.nodes[node]['my_betweenness']/SUM
        

# # Generate a colormap with as many colors as the number of unique groups
# cmap = plt.cm.get_cmap('tab20', len(node_df['max_EP_lu'].unique()))

# # Create a dictionary mapping each group to a color
# color_dict = {group: cmap(i/len(node_df['max_EP_lu'].unique())) for i, group in enumerate(node_df['max_EP_lu'].unique())}

# # Create a figure with a single subplot
# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot the nodes with their respective positions and color them by group
# pos = {node_id: (node_df.loc[node_df['LinkID'] == node_id, 'longitude'].iloc[0], node_df.loc[node_df['LinkID'] == node_id, 'latitude'].iloc[0]) for node_id in G.nodes()}

# # Get the 20 nodes with the highest betweenness centrality
# top_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['my_betweenness'], reverse=True)[:20]

# # Update the node colors to black for the top nodes
# node_colors = [color_dict[G.nodes[node_id]['max_EP_lu']] if node_id not in top_nodes else 'red' for node_id in G.nodes()]

# # Update the node size to 10 for the top nodes
# node_size = [2 if node_id not in top_nodes else 100 for node_id in G.nodes()]

# # Plot the nodes with their respective positions and color them by group
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=node_size)

# nx.draw_networkx_edges(G, pos, arrows=False)

# # Create a legend mapping each group to its color
# legend_handles = [mpatches.Patch(color=color_dict[group], label=group) for group in node_df['max_EP_lu'].unique()]
# ax.legend(handles=legend_handles, loc='upper left')

# # Show the plot
# plt.show()

            
            

