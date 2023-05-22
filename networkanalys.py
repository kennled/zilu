import pandas as pd
import networkx as nx
import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000

#Read the CSV file
df_road = pd.read_csv('../2019.02.19.csv',nrows=58780)

# Create a dictionary mapping link ids to their respective start and end points
locations = df_road["Location"].str.split().apply(lambda x: tuple(map(float, x))).tolist()
sp = [tuple(map(float, x.split()))[:2] for x in df_road["Location"]]
ep = [tuple(map(float, x.split()))[2:] for x in df_road["Location"]]
location_dict = dict(zip(df_road["LinkID"], zip(sp, ep)))

# Create an empty list to store the node data
node_data = []

# Iterate over the location dictionary and only keep the links within the boundary
for link_id, (start_point, end_point) in location_dict.items():
    # Append the link_id, start_point, and end_point as a tuple
    node_data.append((link_id, start_point, end_point))

# Create a Pandas DataFrame from the link data
node_df = pd.DataFrame(node_data, columns=["LinkID", "StartPoint", "EndPoint"])

# Calculate the center point for each link
node_df["CenterPoint"] = node_df[["StartPoint", "EndPoint"]].apply(lambda row: ((row["StartPoint"][0] + row["EndPoint"][0]) / 2, (row["StartPoint"][1] + row["EndPoint"][1]) / 2), axis=1)

# Create separate "Latitude" and "Longitude" columns from "CenterPoint"
node_df[["latitude", "longitude"]] = node_df["CenterPoint"].apply(lambda point: pd.Series([point[0], point[1]]))


# Create a new empty graph
G = nx.DiGraph()

# Add all nodes to the graph with their attributes
for _, row in node_df.iterrows():
    G.add_node(row['LinkID'], latitude=row['latitude'], longitude=row['longitude'])

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
            G.add_edge(row['LinkID'], next_row['LinkID'], weight=distance)
#%%
# Calculate node degrees
degrees = [G.degree(node) for node in G.nodes()]

# Plot degree distribution
plt.hist(degrees, bins='auto', alpha=0.7)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
plt.show()
# Calculate neighbor degrees
neighbor_degrees = []
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if neighbors:
        neighbor_degree = sum(G.degree(neighbor) for neighbor in neighbors) / len(neighbors)
        neighbor_degrees.append(neighbor_degree)

# Plot neighbor degree distribution
plt.hist(neighbor_degrees, bins='auto', alpha=0.7)
plt.xlabel('Neighbor Degree')
plt.ylabel('Frequency')
plt.title('Neighbor Degree Distribution')
plt.show()
# Calculate local clustering coefficient
clustering_coefficients = nx.clustering(G)

cc_values = list(clustering_coefficients.values())

plt.hist(cc_values, bins=20, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')
plt.title('Log-Log Clustering Coefficient Distribution')
plt.show()
#%%

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(30, 20))

# Plot the nodes with their respective positions and color them by group
pos = {node_id: (node_df.loc[node_df['LinkID'] == node_id, 'longitude'].iloc[0], node_df.loc[node_df['LinkID'] == node_id, 'latitude'].iloc[0]) for node_id in G.nodes()}

# Get the top 10 nodes with highest clustering coefficients
top_nodes = sorted(clustering_coefficients, key=clustering_coefficients.get, reverse=True)[:200]

# Update the node size to 10 for the top nodes
node_size = [1 if node_id not in top_nodes else 100 for node_id in G.nodes()]

# Update the node colors to red for the top nodes
node_color = ['red' if node_id in top_nodes else 'blue' for node_id in G.nodes()]

# Plot the nodes with their respective positions and color them by group
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color)

nx.draw_networkx_edges(G, pos, arrows=False)

# Show the plot
plt.show()
