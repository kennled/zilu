import matplotlib.dates as mdates
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from datetime import datetime
# node_df = pd.read_csv('all_node.csv')
# # Read the CSV file
# df_road = pd.read_csv('../speeddata.csv')

# # Filter df_road to include only rows where LinkID is in node_df
# df_road = df_road[df_road['LinkID'].isin(node_df['LinkID'])]
#%%
# mean_speedband = df_road.groupby('LinkID')['SpeedBand'].mean()
# # Reset the index of mean_speedband to align it with df_road
# mean_speedband = mean_speedband.reset_index()

# # Merge mean_speedband with df_road based on LinkID
# df_road = df_road.merge(mean_speedband, on='LinkID', suffixes=['', '_mean'])

# # Create a new column indicating if the SpeedBand is below mean_speedband * 0.5
# df_road['IsBelowThreshold'] = df_road['SpeedBand'] < (df_road['SpeedBand_mean'] * 0.5)
#%%
# #筛选 Isbelowthreshold
# # Create a new column indicating if the True values persist for three or more connected rows
# df_road['IsPersisting'] = df_road['IsBelowThreshold'].rolling(3).sum() >= 3

# unique_linkids = df_road['LinkID'].unique()

# for idx, row in df_road.iterrows():
#     if row['IsPersisting']:
#         # Set the above 3 rows (including itself) of 'IsBelowThreshold' to True
#         df_road.loc[max(0, idx - 3):idx, 'IsBelowThreshold'] = True
#     else:
#         # Set other rows of 'IsBelowThreshold' to False
#         df_road.loc[idx, 'IsBelowThreshold'] = False
#     # Display progress
#     print(f"Progress: {idx+1}/{len(df_road)}", end="\r")
    
# # Convert the 'RequestTimestamp' column to datetime format
# df_road['RequestTimestamp'] = pd.to_datetime(df_road['RequestTimestamp'])
# # Iterate over each LinkID and check the condition
# for linkid in unique_linkids:
#     condition = (df_road['LinkID'] == linkid) & (df_road['RequestTimestamp'].dt.time < pd.Timestamp('06:00:00').time())
#     if df_road.loc[condition, 'IsBelowThreshold'].all():
#         df_road.loc[df_road['LinkID'] == linkid, 'IsBelowThreshold'] = False
#     # Display progress
#     print(f"Progress: {linkid}/{unique_linkids[-1]}", end="\r")

#%%
#看某一个路的情况
# link_id =103064442

# filtered_data = df_road[df_road['LinkID'] == link_id]
# # Sort the filtered data by RequestTimestamp
# sorted_data = filtered_data.sort_values('RequestTimestamp')
# # Plot SpeedBand vs RequestTimestamp as a step curve
# plt.figure(figsize=(10, 6))
# plt.step(sorted_data['RequestTimestamp'], sorted_data['SpeedBand'], where='post')
# plt.xlabel('Request Timestamp')
# plt.ylabel('SpeedBand')
# plt.title(f'SpeedBand vs RequestTimestamp for LinkID {link_id}')
# plt.show()
#%%等分成几个组
# # Sort the node_df based on my_betweenness values
# node_df_sorted = node_df.sort_values('my_betweenness')
# g=4
# # Calculate the number of LinkIDs in each group
# group_size = len(node_df_sorted) // g

# # Create the groups
# groups = []
# for i in range(g):
#     start_idx = i * group_size
#     end_idx = (i + 1) * group_size
#     if i == g-1:
#         end_idx = len(node_df_sorted)
#     group = node_df_sorted.iloc[start_idx:end_idx]['LinkID']
#     groups.append(group)

# # Print the groups
# for i, group in enumerate(groups):
#     print(f"Group {i+1}:")
#     print(group)
#     print()
# group_means = []
# for group in groups:
#     group_mean = df_road[df_road['LinkID'].isin(group)].groupby('RequestTimestamp')['SpeedBand'].mean()
#     group_means.append(group_mean)
# # Iterate over group_means and plot each group
# for i, group_mean in enumerate(group_means):
#     group_label = f'Group {i+1}'
#     plt.plot(group_mean.index, group_mean.values, label=group_label)

# # Customize the plot
# plt.xlabel('Request Timestamp')
# plt.ylabel('SpeedBand')
# plt.title('SpeedBand vs RequestTimestamp')
# plt.legend()

# # Display the plot
# plt.show()
#%%
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000
node_df = pd.read_csv('output_land_find_road_303.csv')

# Create a new empty graph
G = nx.DiGraph()

# Add all nodes to the graph with their attributes
for _, row in node_df.iterrows():
    G.add_node(row['LinkID'], max_EP_lu=row['max_EP_lu'], group=row['group'], latitude=row['latitude'], longitude=row['longitude'], EP=row['max_EP'])

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
betweenness=pd.read_csv('all_node.csv')
for _, row in betweenness.iterrows():
    link_id = row['LinkID']
    betweenness_value = row['my_betweenness']
    G.nodes[link_id]['betweenness'] = betweenness_value
#%%
# df_road = pd.read_csv('Congested.csv')
# congested_data = df_road[df_road['IsBelowThreshold'] == True]
# # Function to find predecessors up to a given level
# def find_predecessors_up_to_level(G, linkid, level):
#     predecessors = list(G.predecessors(linkid))
    
#     for _ in range(level - 1):
#         next_level_predecessors = []
        
#         for pred in predecessors:
#             next_level_predecessors.extend(list(G.predecessors(pred)))
        
#         predecessors = next_level_predecessors.copy()
    
#     return predecessors

# # Group congested_data by RequestTimestamp
# groups = congested_data.groupby('RequestTimestamp')

# # Create an empty set to store the filtered congested_data
# filtered_data = set()

# level_number = 8  # Set the desired level number

# # Iterate over each group
# for _, group_df in groups:
#     # Iterate over each row in the current group
#     for _, row in group_df.iterrows():
#         # Get the current LinkID
#         linkid = row['LinkID']
        
#         # Dictionary to store predecessors at different levels
#         predecessors_by_level = {}

#         # Find the predecessors up to each level and store them in the dictionary
#         for i in range(1, level_number+1):
#             predecessors_up_to_level = find_predecessors_up_to_level(G, linkid, i)
#             predecessors_by_level[i] = predecessors_up_to_level

#         # Dictionary to store predecessors in the current group at different levels
#         predecessors_of_level_in_group = {}

#         # Flag to track if all levels' predecessors are in the current group
#         all_levels_present = True

#         # Iterate over the levels and check if each level's predecessors are in the current group
#         for level in range(1, level_number+1):
#             predecessors_of_level = predecessors_by_level[level]
#             predecessors_of_level_in_group[level] = group_df[group_df['LinkID'].isin(predecessors_of_level)]
#             if predecessors_of_level_in_group[level].empty:
#                 all_levels_present = False
#                 break

#  # If all levels' predecessors are in the current group, add the rows to the filtered_data set
#         if all_levels_present:
#             for level in range(1, level_number+1):
#                 filtered_data.add(tuple(predecessors_of_level_in_group[level].values.tolist()[0]))
        
#             # Add the row itself to the filtered_data set
#             filtered_data.add(tuple(row.values.tolist()))

# # Convert the filtered_data set to a list
# filtered_data = list(filtered_data)

# # Create a new DataFrame from the filtered_data list
# filtered_congested_data = pd.DataFrame(filtered_data, columns=congested_data.columns)

#%%

df_road = pd.read_csv('Congested.csv')
congested_data = df_road[df_road['IsBelowThreshold'] == True]
# Group congested_data by RequestTimestamp
groups = congested_data.groupby('RequestTimestamp')
# Lists to store the RequestTimestamp, largest connected component sizes, and LinkIDs
timestamps = []
component_sizes = []
component_linkids = []

# Iterate over each group
for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Create a subgraph containing only the nodes present in the current group_df
    subgraph = G.subgraph(group_df['LinkID'])
    
    # Find all connected components in the subgraph
    connected_components = nx.weakly_connected_components(subgraph)
    
    # Find the largest connected component
    largest_component = max(connected_components, key=len)
    
    # Store the RequestTimestamp, largest connected component size, and LinkIDs
    timestamps.append(timestamp)
    component_sizes.append(len(largest_component))
    component_linkids.append(largest_component)

# Find the index of the biggest largest connected component
biggest_component_index = component_sizes.index(max(component_sizes))
biggest_component_size = component_sizes[biggest_component_index]
biggest_component_timestamp = timestamps[biggest_component_index]
biggest_component_linkids = component_linkids[biggest_component_index]

# Plotting the data
fig, ax = plt.subplots()
ax.plot(timestamps, component_sizes)
# Format the x-tick labels to show only the hour and minute
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('RequestTimestamp(hour:minute)')
plt.ylabel('Largest connected component size')
plt.title('Largest Connected Component Size over Time')
plt.tight_layout()
plt.show()

print("Biggest Largest Connected Component:")
print("Timestamp:", biggest_component_timestamp)
print("Size:", biggest_component_size)
print("LinkIDs:", biggest_component_linkids)

mean_speed = df_road[df_road['LinkID'].isin(biggest_component_linkids)].groupby('RequestTimestamp')['SpeedBand'].mean()
mean_speed.index = pd.to_datetime(mean_speed.index)  # Convert index to datetime format

# Plot the mean speed for the top nodes
fig, ax = plt.subplots()
ax.plot(mean_speed.index, mean_speed.values)
# Format the x-tick labels to show only the hour and minute
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('RequestTimestamp(hour:minute)')
plt.ylabel('Mean SpeedBand')
plt.title('Mean SpeedBand')

plt.tight_layout()
plt.show()
#%%
df_road = pd.read_csv('Congested.csv')
congested_data = df_road[df_road['IsBelowThreshold'] == True]
# Group congested_data by RequestTimestamp
groups = congested_data.groupby('RequestTimestamp')
# Initialize counters
count_total = 0  # Total count of LinkIDs in largest components with size > 8
count_linkid = {}  # Dictionary to store counts for each LinkID

# Iterate over each group
for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Create a subgraph containing only the nodes present in the current group_df
    subgraph = G.subgraph(group_df['LinkID'])
    
    # Find all connected components in the subgraph
    connected_components = nx.weakly_connected_components(subgraph)
    
    # Count the LinkIDs in connected components with size > 8
    for component in connected_components:
        if len(component) > 9:
            count_total += len(component)  # Increment the total count by the size of the component

            # Count the occurrences of LinkIDs in the current component
            for linkid in component:
                count_linkid[linkid] = count_linkid.get(linkid, 0) + 1

# Print the total count and counts for each LinkID
print("Total count of LinkIDs in largest components with size > 3:", count_total)
print("Counts for each LinkID:")
for linkid, count in count_linkid.items():
    print("LinkID:", linkid, "Count:", count)
# Create a list to store the LinkID and count information
linkid_counts = []

# Iterate over the count_linkid dictionary
for linkid, count in count_linkid.items():
    linkid_counts.append({'LinkID': linkid, 'Count': count})

# Create a DataFrame from the linkid_counts list
linkid_counts_df = pd.DataFrame(linkid_counts)
#%%
node_df = pd.read_csv('all_node.csv')


merged_df = pd.merge(linkid_counts_df, node_df, on='LinkID')

merged_df=merged_df[merged_df['LinkID'].isin(linkid_counts_df['LinkID'])]

plt.scatter(merged_df['my_betweenness'], merged_df['Count'])
plt.xlabel('my_betweenness')
plt.ylabel('Congested level')
plt.xscale('log')
# plt.yscale('log')
plt.show()
#%%

df_road = pd.read_csv('Congested.csv')
# Convert the 'RequestTimestamp' column to a pandas datetime object
df_road['RequestTimestamp'] = pd.to_datetime(df_road['RequestTimestamp'])
# Group the data by RequestTimestamp
grouped_data = df_road.groupby('RequestTimestamp')

# Calculate the proportion of IsBelowThreshold being True for each group
proportions = grouped_data['IsBelowThreshold'].mean()
# Plot the proportions
fig, ax = plt.subplots()
ax.plot(proportions.index, proportions.values)
# Format the x-tick labels to show only the hour and minute
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('RequestTimestamp(hour:minute)')
plt.ylabel('proportion of congested road ')
plt.title('zonewide congeestion ratio')

plt.tight_layout()
plt.show()
#%%

# Calculate the excess degree for each node
excess_degrees = [degree - 1 for node, degree in G.degree()]

# Calculate the mean excess degree
mean_excess_degree = sum(excess_degrees) / G.number_of_nodes()
#%%
# Calculate the average neighbor's degree
avg_neighbor_degree = nx.average_neighbor_degree(G)

avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
avg_neighbor_degree_values = list(avg_neighbor_degree.values())
avg_neighbor_degree_avg = sum(avg_neighbor_degree_values) / len(avg_neighbor_degree_values)


avg_degree-avg_neighbor_degree_avg
#%%
df_road = pd.read_csv('Congested.csv')
congested_data = df_road[df_road['IsBelowThreshold'] == True]

# Create lists to store the cluster sizes and timestamps
cluster_sizes = []
timestamps = []
betweenness_sums = []
# Group congested_data by RequestTimestamp
groups = congested_data.groupby('RequestTimestamp')

# Iterate over each group
for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Create a subgraph containing only the nodes present in the current group_df
    subgraph = G.subgraph(group_df['LinkID'])
    
    # Find all connected components in the subgraph
    connected_components = list(nx.weakly_connected_components(subgraph))
    
   # Calculate the size of each connected component with a minimum size of 3 and store in the cluster_sizes list
    cluster_sizes.append(sum(len(component) for component in connected_components if len(component) >= 3))
    # Calculate the size and sum of betweenness for each connected component
    betweenness_sums.append(sum(sum(G.nodes[node]['betweenness'] for node in component)for component in connected_components if len(component) >= 3))
    
    # Store the corresponding timestamp in the timestamps list
    timestamps.append(timestamp)

# Convert timestamps to numeric format
numeric_timestamps = mdates.date2num(timestamps)
import matplotlib.tri as mtri
# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter points
ax.scatter(numeric_timestamps, betweenness_sums, cluster_sizes, c=cluster_sizes, cmap='viridis')

# Set x-axis labels as timestamps
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Set labels for each axis
ax.set_xlabel('Timestamp')
ax.set_ylabel('Betweenness Sums')
ax.set_zlabel('Cluster Sizes')

# Set the colorbar to indicate the cluster sizes
cbar = plt.colorbar(ax.scatter([], [], [], c=[], cmap='viridis'))
cbar.set_label('Cluster Sizes')
plt.subplots_adjust(right=2)
# Show the plot
plt.show()

#%%
df_road = pd.read_csv('Congested.csv')
congested_data = df_road[df_road['IsBelowThreshold'] == True]
# Group congested_data by RequestTimestamp
groups = congested_data.groupby('RequestTimestamp')
component_sizes = []  # List to store component sizes

# Iterate over each group
for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Create a subgraph containing only the nodes present in the current group_df
    subgraph = G.subgraph(group_df['LinkID'])
    
    # Find all connected components in the subgraph
    connected_components = nx.weakly_connected_components(subgraph)
    
    # Iterate over each connected component
    for component in connected_components:
        if len(component) >= 3:
            component_sizes.append(len(component))
# Calculate the frequency of each component size
component_freq = [component_sizes.count(size) for size in set(component_sizes)]

# Convert the set of component sizes to a list
component_sizes = list(set(component_sizes))

# Sort the component_sizes and component_freq arrays in ascending order of component_sizes
component_sizes, component_freq = zip(*sorted(zip(component_sizes, component_freq)))

# Plot the curve
plt.plot(component_sizes, component_freq, marker='o')
plt.xlabel('Cluster Size')
plt.ylabel('Frequency')
plt.title('Cluster Size vs Frequency')
plt.yscale('log')

plt.show()
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
import matplotlib.patches as mpatches
# Generate a colormap with as many colors as the number of unique groups
cmap = plt.cm.get_cmap('tab20', len(node_df['max_EP_lu'].unique()))

# Create a dictionary mapping each group to a color
color_dict = {group: cmap(i/len(node_df['max_EP_lu'].unique())) for i, group in enumerate(node_df['max_EP_lu'].unique())}

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the nodes with their respective positions and color them by group
pos = {node_id: (node_df.loc[node_df['LinkID'] == node_id, 'longitude'].iloc[0], node_df.loc[node_df['LinkID'] == node_id, 'latitude'].iloc[0]) for node_id in G.nodes()}

# Get the top 10 nodes with highest clustering coefficients
top_nodes = sorted(clustering_coefficients, key=clustering_coefficients.get, reverse=True)[:20]

# Update the node colors to black for the top nodes
node_colors = [color_dict[G.nodes[node_id]['max_EP_lu']] if node_id not in top_nodes else 'red' for node_id in G.nodes()]

# Update the node size to 10 for the top nodes
node_size = [2 if node_id not in top_nodes else 100 for node_id in G.nodes()]

# Plot the nodes with their respective positions and color them by group
nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=node_size)

nx.draw_networkx_edges(G, pos, arrows=False)

# Create a legend mapping each group to its color
legend_handles = [mpatches.Patch(color=color_dict[group], label=group) for group in node_df['max_EP_lu'].unique()]
ax.legend(handles=legend_handles, loc='upper left')

# Show the plot
plt.show()
#%%

# Perform modularity optimization
partition = nx.community.greedy_modularity_communities(G)

# Convert partition to dictionary format
node_to_comm = {}
for comm_id, comm in enumerate(partition):
    for node in comm:
        node_to_comm[node] = comm_id

# Generate a colormap with as many colors as the number of communities
cmap = plt.cm.get_cmap('tab20', len(partition))

# Plot the nodes with their respective positions and color them by group
pos = {node_id: (node_df.loc[node_df['LinkID'] == node_id, 'longitude'].iloc[0], node_df.loc[node_df['LinkID'] == node_id, 'latitude'].iloc[0]) for node_id in G.nodes()}

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the nodes with their respective positions and color them by community
node_colors = [cmap(node_to_comm[node]) for node in G.nodes()]
nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, ax=ax,node_size=5)

nx.draw_networkx_edges(G, pos, arrows=False)
# Create a legend mapping each community to its color
legend_handles = [mpatches.Patch(color=cmap(comm_id), label=f'Community {comm_id}') for comm_id in set(node_to_comm.values())]
ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.05, 0.95), bbox_transform=plt.gcf().transFigure)
# Show the plot
plt.show()
        
           














