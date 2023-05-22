
df_road = pd.read_csv('Congested.csv')
congested_data = df_road[df_road['IsBelowThreshold'] == True]
# Group congested_data by RequestTimestamp
groups = congested_data.groupby('RequestTimestamp')
# Plot the nodes with their respective positions and color them by group
pos = {node_id: (node_df.loc[node_df['LinkID'] == node_id, 'longitude'].iloc[0], node_df.loc[node_df['LinkID'] == node_id, 'latitude'].iloc[0]) for node_id in G.nodes()}



plt.axis('off')
plt.show()
for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

    # Check if the timestamp is within the desired time range
    if datetime.time(timestamp) >= datetime.time(datetime.strptime('19:05:00', '%H:%M:%S')) and datetime.time(timestamp) <= datetime.time(datetime.strptime('19:35:00', '%H:%M:%S')):

        # Create a subgraph containing only the nodes present in the current group_df
        subgraph = G.subgraph(group_df['LinkID'])

        # Find all connected components in the subgraph
        connected_components = nx.weakly_connected_components(subgraph)

        # Find the largest connected component
        largest_component = max(connected_components, key=len)

        if largest_subgraph is None:
            # If it's the first largest component found, initialize the largest_subgraph
            largest_subgraph = subgraph.subgraph(largest_component)
        else:
            # Otherwise, update the largest_subgraph if a larger component is found
            current_subgraph = subgraph.subgraph(largest_component)
            if len(current_subgraph) > len(largest_subgraph):
                largest_subgraph = current_subgraph


for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

    # Check if the timestamp is within the desired time range
    if datetime.time(timestamp) == datetime.time(datetime.strptime('19:05:00', '%H:%M:%S')):
        
        # Create a subgraph containing only the nodes present in the current group_df
        subgraph = G.subgraph(group_df['LinkID'])

        # Find all connected components in the subgraph
        connected_components = nx.weakly_connected_components(subgraph)

        # Find the largest connected component
        largest_component1 = max(connected_components, key=len)


        
for timestamp, group_df in groups:
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    if datetime.time(timestamp) >= datetime.time(datetime.strptime('18:50:00', '%H:%M:%S')) and datetime.time(timestamp) <= datetime.time(datetime.strptime('19:00:00', '%H:%M:%S')):
        subgraph = G.subgraph(group_df['LinkID'])
        connected_components = nx.weakly_connected_components(subgraph)
        
        # Iterate over connected components
        for component in connected_components:
            # Check if any node from the largest_component is present in the current component
            if any(node in component for node in largest_component1):
                # Create a subgraph for the current component
                subgraph_component = subgraph.subgraph(component)
                # Plotting the largest connected component with arrows and LinkID labels
                plt.figure(figsize=(10, 10), dpi=300)
                plt.title(f"Largest component at {timestamp}")
                                
                nx.draw_networkx_nodes(subgraph_component, pos, node_color='blue', node_size=1)
                nx.draw_networkx_edges(subgraph_component, pos, edge_color='gray', arrows=True)
                nx.draw_networkx_labels(subgraph_component, pos, labels={node: node for node in largest_subgraph.nodes()}, font_size=1, font_color='black')
                plt.show()
for timestamp, group_df in groups:
    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

    # Check if the timestamp is within the desired time range
    if datetime.time(timestamp) >= datetime.time(datetime.strptime('19:05:00', '%H:%M:%S')) and datetime.time(timestamp) <= datetime.time(datetime.strptime('19:35:00', '%H:%M:%S')):
        
        # Create a subgraph containing only the nodes present in the current group_df
        subgraph = G.subgraph(group_df['LinkID'])

        # Find all connected components in the subgraph
        connected_components = nx.weakly_connected_components(subgraph)

        # Find the largest connected component
        largest_component1 = max(connected_components, key=len)

        # Create a subgraph for the largest connected component
        largest_subgraph1 = subgraph.subgraph(largest_component1)
        
        # Plotting the largest connected component with arrows and LinkID labels
        plt.figure(figsize=(10, 10), dpi=300)
        plt.title(f"Largest component at {timestamp}")
        
        # Draw nodes with positions and color them by group
        nx.draw_networkx_nodes(largest_subgraph1, pos, node_color='blue', node_size=10)
        nx.draw_networkx_edges(largest_subgraph1, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(largest_subgraph1, pos, labels={node: node for node in largest_subgraph.nodes()}, font_size=8, font_color='black')
        
        plt.axis('off')
        plt.show()


#                 # Plotting the largest connected component with arrows and LinkID labels
#         plt.figure(figsize=(10, 10), dpi=300)
#         plt.title(f"Largest component at {timestamp}")
        
#         # Draw nodes with positions and color them by group
#         nx.draw_networkx_nodes(largest_subgraph, pos, node_color='blue', node_size=10)
#         nx.draw_networkx_edges(largest_subgraph, pos, edge_color='gray', arrows=True)
#         nx.draw_networkx_labels(largest_subgraph, pos, labels={node: node for node in largest_subgraph.nodes()}, font_size=8, font_color='black')
        
#         plt.axis('off')
#         plt.show()
#%%

# Calculate the in-degree and out-degree of node 103103190
betweenness=pd.read_csv('betweenness_land_find_road_303')
in_degree = G.in_degree(103012819)
out_degree = G.out_degree(103012819)
# Calculate the cluster coefficient of node 103103190
cluster_coefficient = nx.clustering(G, 103012819)

# Print the cluster coefficient
print(f"Cluster coefficient of node 103103190: {cluster_coefficient}")
# Print the in-degree and out-degree
print(f"In-degree of node 103103190: {in_degree}")
print(f"Out-degree of node 103103190: {out_degree}")
#%%

# Read the betweenness data into a DataFrame
betweenness = pd.read_csv('betweenness_land_find_road_303_r^-1.csv')

# Add a new column called "rank_number" to the betweenness DataFrame
betweenness['rank_number'] = betweenness['my_betweenness'].rank(ascending=False)

# Print the betweenness and rank number of nodes in the largest_subgraph
for node in largest_subgraph.nodes():
    link_id = node
    node_betweenness = betweenness.loc[betweenness['LinkID'] == link_id, 'my_betweenness']
    node_rank = betweenness.loc[betweenness['LinkID'] == link_id, 'rank_number']
    if not node_betweenness.empty:
        print(f"Node {node} - LinkID {link_id}: Betweenness {node_betweenness.values[0]}, Rank Number {node_rank.values[0]}")
# Calculate the mean betweenness and rank number of nodes in the subgraph
subgraph_betweenness = betweenness.loc[betweenness['LinkID'].isin(largest_subgraph.nodes()), 'my_betweenness']
subgraph_rank = betweenness.loc[betweenness['LinkID'].isin(largest_subgraph.nodes()), 'rank_number']
mean_betweenness = subgraph_betweenness.mean()
mean_rank = subgraph_rank.mean()

# Print the mean betweenness and rank number of nodes in the subgraph
print(f"Mean Betweenness in Subgraph: {mean_betweenness}")
print(f"Mean Rank Number in Subgraph: {mean_rank}")
#%%
plt.figure(figsize=(10, 6), dpi=300)
# Draw nodes with positions and color them by group
nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=1)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=False)
# Draw nodes with positions and color them by group
nx.draw_networkx_nodes(largest_subgraph, pos, node_color='red', node_size=10)
nx.draw_networkx_edges(largest_subgraph, pos, edge_color='gray', arrows=False)


plt.title('Graph with Largest Subgraph Highlighted')
plt.axis('off')
plt.show()




# prev_connected_components = set()

# for timestamp, group_df in groups:
#     # Convert the timestamp to datetime format
#     timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

#     # Create a subgraph containing only the nodes present in the current group_df
#     subgraph = G.subgraph(group_df['LinkID'])

#     # Find all connected components in the subgraph
#     connected_components = nx.weakly_connected_components(subgraph)
#     # Calculate the number of connected components
#     num_components = len(list(connected_components))

#     # Print the number of connected components
#     print(f"Number of connected components: {num_components}")
    # # Check if any connected component from the previous timestamp is growing
    # growing_components = []

    # for component in prev_connected_components:
    #     for new_component in connected_components:
    #         if component.issubset(new_component):
    #             growing_components.append(new_component)

    # # Print the growing connected components for this timestamp
    # if growing_components:
    #     print(f"Growing components at {timestamp}:")
    #     for component in growing_components:
    #         print(component)

    # # Update the previous connected components for the next iteration
    # prev_connected_components = connected_components