import pandas as pd
import geopandas as gpd
import shapely.geometry as geom
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches


# Read the JSON file into a DataFrame
df = pd.read_json('../SG_landuse_wGPR.json')

# Define a function to round the coordinates
def round_coordinates(coords):
    for ring in coords['rings']:
        for coord in ring:
            coord[0] = round(coord[0], 13)
            coord[1] = round(coord[1], 13)
    return coords

# Apply the function to the 'rings' list in the 'geometry' column
df['geometry'] = df['geometry'].apply(round_coordinates)

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df)

#Change the geometry format 
gdf['geometry'] = gdf['geometry'].apply(lambda x: geom.Polygon(x['rings'][0]))

# Get the center point of each polygon and add it as a new column 'center'
gdf['center'] = gdf['geometry'].centroid

# Get the area from SHAPE_Area and add it as a new column 'area'
gdf['area'] = gdf['attributes'].apply(lambda x: x['SHAPE_Area'])

#Get Lu_DESC
gdf['LU_DESC'] = gdf['attributes'].apply(lambda x: x['LU_DESC'])

#Get 'GPR_NUM'
gdf['GPR_NUM'] = gdf['attributes'].apply(lambda x: x['GPR_NUM'])

#Replace attributes with 'OBJECTID'
gdf['attributes'] = gdf['attributes'].apply(lambda x: x['OBJECTID'])

#select the rows where the 'GPR_NUM' column has NaN values
nan_rows = df[df['GPR_NUM'].isna()]

#group the 'nan_rows' DataFrame by the 'LU_DESC' column
nan_rows_count_by_lu = nan_rows.groupby('LU_DESC').size()

#find the mean value of the 'GPR_NUM' column for their respective 'LU_DESC'
mean_gpr_by_lu = df.groupby('LU_DESC')['GPR_NUM'].mean()

#set nan and 0 to the mean value of the 'GPR_NUM' column for their respective 'LU_DESC'
df['GPR_NUM'] = df.groupby('LU_DESC')['GPR_NUM'].apply(lambda x: x.replace(0, np.nan).fillna(x.mean()))

#calculate the expected population
df['EP'] = df['area'] * df['GPR_NUM']

#set white EP to be 0
df.loc[df['LU_DESC'] == 'WHITE', 'EP'] = 0

# Group the DataFrame by 'LU_DESC'
grouped = df.groupby('LU_DESC')
#%%
# #画EP的分布 对每一种LU
# # Loop over each group and create a histogram of the 'EP' column
# for name, group in grouped:
#     group_size = len(group)
#     num_bins = max(group_size//100, 5) # set bins to size of group/100, minimum of 5
#     plt.hist(group['EP'], bins=num_bins)
#     plt.xlabel('EP')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of EP for LU_DESC = {}'.format(name))
#     plt.show()
#%%
# #画landuse的边
# # Filter the DataFrame for attributes = 1244
# poly_df = gpd.GeoDataFrame(df[df['attributes'] == 25854], geometry='geometry')

# # Plot the polygon
# fig, ax = plt.subplots(figsize=(10,10))
# poly_df.plot(ax=ax, alpha=0.5)
# plt.axis('off')
# plt.show()
#%%
#准备工作 有对 #画那些百分比以下速度时间最长的点 的准备
# Define the area boundaries
min_lat, max_lat, min_lon, max_lon = [1.27523 , 1.2986 , 103.8100, 103.8550]

# Filter the DataFrame
filtered_df = df[(df['center'].apply(lambda p: p.x) >= min_lon) & 
                 (df['center'].apply(lambda p: p.x) <= max_lon) &
                 (df['center'].apply(lambda p: p.y) >= min_lat) & 
                 (df['center'].apply(lambda p: p.y) <= max_lat)]

# sorted_df = filtered_df.sort_values('EP', ascending=False)
# sorted_df = filtered_df.sort_values('GPR_NUM', ascending=False)

unique_lu_desc = filtered_df['LU_DESC'].unique()
# filtered1_df = filtered_df[filtered_df['LU_DESC'] == 'BUSINESS 1 - WHITE']

# with open('network.pkl', 'rb') as f:
#     network = pickle.load(f)

# Extract latitude and longitude from "center" column
filtered_df['latitude'] = filtered_df["center"].apply(lambda x: x.y)
filtered_df['longitude'] = filtered_df["center"].apply(lambda x: x.x)

palette = sns.color_palette("tab20", 15)

# Create a scatter plot with different colors based on the LU_DESC column
sns.scatterplot(x="longitude", y="latitude", hue="LU_DESC", palette=palette[:len(unique_lu_desc)], data=filtered_df)

# Set the title and axis labels
plt.title("Filtered Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Adjust legend position
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

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

# Define the area boundaries
min_lat, max_lat, min_lon, max_lon = [min_lat-0.0018, max_lat-0.0018, min_lon-0.0018, max_lon-0.0018]

# Iterate over the location dictionary and only keep the links within the boundary
for link_id, (start_point, end_point) in location_dict.items():
    # Check if the start and end points are within the area boundaries
    if min_lat <= start_point[0] <= max_lat and min_lat <= end_point[0] <= max_lat and min_lon <= start_point[1] <= max_lon and min_lon <= end_point[1] <= max_lon:
        # Append the link_id, start_point, and end_point as a tuple
        node_data.append((link_id, start_point, end_point))

# Create a Pandas DataFrame from the link data
node_df = pd.DataFrame(node_data, columns=["LinkID", "StartPoint", "EndPoint"])

# Calculate the center point for each link
node_df["CenterPoint"] = node_df[["StartPoint", "EndPoint"]].apply(lambda row: ((row["StartPoint"][0] + row["EndPoint"][0]) / 2, (row["StartPoint"][1] + row["EndPoint"][1]) / 2), axis=1)

# Create separate "Latitude" and "Longitude" columns from "CenterPoint"
node_df[["latitude", "longitude"]] = node_df["CenterPoint"].apply(lambda point: pd.Series([point[0], point[1]]))

# Read the CSV file
df_road = pd.read_csv('../2019.02.19.csv')

# Convert the 'RequestTimestamp' column to a pandas datetime object
df_road['RequestTimestamp'] = pd.to_datetime(df_road['RequestTimestamp'], format='%Y.%m.%d.%H.%M.%S')

# # Filter the DataFrame to include only rows where RequestTimestamp is between '2019.02.19.08.30.00' and '2019.02.19.23.00.00'
# df_road = df_road[(df_road['RequestTimestamp'] >= pd.to_datetime('2019.02.19.08.30.00', format='%Y.%m.%d.%H.%M.%S')) & (df_road['RequestTimestamp'] <= pd.to_datetime('2019.02.19.23.00.00', format='%Y.%m.%d.%H.%M.%S'))]

# Create a new column called 'max_speed' that stores the maximum value of 'SpeedBand' for each 'LinkID'
df_road['max_speed'] = df_road.groupby('LinkID')['SpeedBand'].transform('max')

# Create a new column indicating whether each row's SpeedBand value is below the max_speed value for its corresponding LinkID
df_road['speed_below_max'] = df_road['SpeedBand'] < df_road['max_speed']*0.2

# Group the DataFrame by LinkID and count the number of True values in the speed_below_mean column for each group
count_df = df_road.groupby('LinkID')['speed_below_max'].sum().reset_index()

# Rename the speed_below_mean column to num_speed_below_mean
count_df = count_df.rename(columns={'speed_below_max': 'num_speed_below_max'})

# Merge the node_df and count_df DataFrames on 'LinkID'
node_df = pd.merge(node_df, count_df[['LinkID', 'num_speed_below_max']], on='LinkID', how='left')
#%%画道路的长度分布
# # Loop through each row in node_df
# for index, row in node_df.iterrows():
#     # Calculate length of road segment using haversine formula
#     length = haversine(row['StartPoint'][1], row['StartPoint'][0], row['EndPoint'][1], row['EndPoint'][0])
#     # Update node_df with length value
#     node_df.at[index, 'Length'] = length
    
# # create a histogram of the length column
# plt.hist(node_df['Length'], bins=20)

# # set the plot title and axis labels
# plt.title('Distribution of Length')
# plt.xlabel('Length (m)')
# plt.ylabel('Frequency')

# # display the plot
# plt.show()
#%%合并两个数据集用road找land
# node_df['max_EP'] = 0
# # Loop through each row in node_df
# # Loop through each row in node_df
# for index, row in node_df.iterrows():
#     # Initialize variables to keep track of max EP and corresponding LU_DESC
#     max_ep = -float('inf')
#     max_lu = ''
#     # Loop through each unique LU_DESC value in filtered_df
#     for lu in filtered_df['LU_DESC'].unique():
#         # Subset filtered_df to only rows with matching LU_DESC value
#         filtered_lu = filtered_df[filtered_df['LU_DESC'] == lu]
#         # Calculate sum of EP values for matching rows within 200 meters
#         ep_sum = 0
#         for findex, frow in filtered_lu.iterrows():
#             dist = haversine(row['longitude'], row['latitude'], frow['longitude'], frow['latitude'])
#             if dist <= 350:
#                 ep_sum += frow['EP']
#         # Update max EP and corresponding LU_DESC if current sum is bigger
#         if ep_sum > max_ep:
#             max_ep = ep_sum
#             max_lu = lu
#             if max_ep==0:
#                 max_lu='None'
#     # Update node_df with max EP and corresponding LU_DESC for current row
#     node_df.at[index, 'max_EP'] = max_ep
#     node_df.at[index, 'max_EP_lu'] = max_lu       

# unique_max_EP_lu = node_df['max_EP_lu'].unique()
# # Create a scatter plot with different colors based on the LU_DESC column
# sns.scatterplot(x="longitude", y="latitude", hue='max_EP_lu', palette=palette[:len(unique_max_EP_lu)], data=node_df)

# # Set the title and axis labels
# plt.title("Road segement")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# # Adjust legend position
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# # display the plot
# plt.show()
#%%合并两个数据集用land来找road
filtered_df['No_nodes_around']=np.zeros(len(filtered_df))
# loop through each unique LU_DESC value in filtered_df
for lu in filtered_df['LU_DESC'].unique():
    # Subset filtered_df to only rows with matching LU_DESC value
    filtered_lu = filtered_df[filtered_df['LU_DESC'] == lu]
    # Create a new column "EP" with zero values for all LU_DESC values
    node_df["EP_"+lu] = np.zeros(len(node_df))
    # loop through each row in filtered_lu
    for index, row in filtered_lu.iterrows():
        # initialize variables to keep track of nodes around
        nodes_around = []
        # loop through each row in the updated_node_df
        for node_index, node_row in node_df.iterrows():
            # calculate distance between current node and current row in filtered_df
            dist = haversine(row['longitude'], row['latitude'], node_row['longitude'], node_row['latitude'])
            # if distance is around, add node to list
            if dist <= 350:
                nodes_around.append(node_index)
        if len(nodes_around) == 0:
            filtered_df.loc[filtered_lu.index, 'No_nodes_around'] = 1
        # add EP values of filtered_lu to node_df for nodes around
        if len(nodes_around) > 0:
            node_df.loc[nodes_around, 'EP_'+lu] += row['EP'] / len(nodes_around)
#print the number that have nodes around
print('the number that have nodes around',filtered_df['No_nodes_around'].value_counts()[0],'total number',len(filtered_df))
ep_columns = [col for col in node_df.columns if "EP_" in col]
# use the max function to find the maximum value for each row, only including the specified columns
node_df["max_EP"] = node_df[ep_columns].max(axis=1)
# get the column index for the maximum value of the specified columns
max_cols = node_df[ep_columns].idxmax(axis=1)
# extract the relevant part of the column name to get the corresponding LU_DESC value
max_EP_lu = max_cols.apply(lambda col: col.split("_")[1])
# add the new column to the dataframe
node_df["max_EP_lu"] = max_EP_lu
node_df.loc[node_df["max_EP"] == 0, "max_EP_lu"] = 'None'
unique_max_EP_lu = node_df['max_EP_lu'].unique()
# Create a scatter plot with different colors based on the LU_DESC column
sns.scatterplot(x="longitude", y="latitude", hue='max_EP_lu', palette=palette[:len(unique_max_EP_lu)], data=node_df)

# Set the title and axis labels
plt.title("Road segement")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Adjust legend position
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# display the plot
plt.show()
#%%
#建立网络(有向)
# Create a new column 'group' by encoding the 'max_EP_lu' values as integers
node_df['group'], _ = pd.factorize(node_df['max_EP_lu'])

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
            G.add_edge(next_row['LinkID'], row['LinkID'], weight=distance)
node_df.to_csv('output.csv')
#%%
#计算和画中心度点
# Generate a colormap with as many colors as the number of unique groups
cmap = plt.cm.get_cmap('tab20', len(node_df['max_EP_lu'].unique()))

# Create a dictionary mapping each group to a color
color_dict = {group: cmap(i/len(node_df['max_EP_lu'].unique())) for i, group in enumerate(node_df['max_EP_lu'].unique())}

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the nodes with their respective positions and color them by group
pos = {node_id: (node_df.loc[node_df['LinkID'] == node_id, 'longitude'].iloc[0], node_df.loc[node_df['LinkID'] == node_id, 'latitude'].iloc[0]) for node_id in G.nodes()}

# Create a subset of nodes with max_EP_lu values 'RESIDENTIAL' and 'COMMERCIAL'
source_nodes = node_df[node_df['max_EP_lu'] == 'RESIDENTIAL']['LinkID'].tolist()
target_nodes = node_df[(node_df['max_EP_lu'] == 'COMMERCIAL')]['LinkID'].tolist()
# target_nodes = node_df[(node_df['max_EP_lu'] == 'COMMERCIAL') | (node_df['max_EP_lu'] == 'BUISNESS 1')]['LinkID'].tolist()

# Calculate betweenness centrality for the subset of nodes
betweenness_subset = nx.betweenness_centrality_subset(G, source_nodes, target_nodes, weight='weight')
# Calculate betweenness centrality for the subset of nodes
betweenness_subset_inverse = nx.betweenness_centrality_subset(G, target_nodes, source_nodes, weight='weight')

# Get the 20 nodes with the highest betweenness centrality
top_nodes = sorted(betweenness_subset, key=betweenness_subset.get, reverse=True)[:20]

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
#查询逆方向的中心性是不是一样的
subset_nodes = set(source_nodes + target_nodes)
diff_count = 0

for node in subset_nodes:
    if node in betweenness_subset and node in betweenness_subset_inverse:
        if betweenness_subset[node] != betweenness_subset_inverse[node]:
            diff_count += 1

if diff_count == 0:
    print("The two betweenness centrality calculations are the same.")
else:
    print("The two betweenness centrality calculations are different. Number of different rows:", diff_count)
#%%
# #画那些百分比以下速度时间最长的点
# # Get the top 30 nodes with the highest values in the 'num_speed_below_max' column
# top_nodes = node_df.sort_values('num_speed_below_max', ascending=False).head(30)

# # Create a figure with a single subplot
# fig, ax = plt.subplots(figsize=(12, 8))

# node_Colors = []
# node_Size = []
# for node_id in G.nodes():
#     if node_id in top_nodes['LinkID'].values:
#         node_Colors.append('red')
#         node_Size.append(100)
#     else:
#         node_Colors.append('green')
#         node_Size.append(2)

# # Plot the nodes with their respective positions and color them by group
# nx.draw_networkx_nodes(G, pos, node_color=node_Colors, ax=ax, node_size=node_Size)

# nx.draw_networkx_edges(G, pos)

# # Show the plot
# plt.show()
#%%
# #画那些最高中心度的点 每个点的速度变化情况
# # Get the 20 nodes with the highest betweenness centrality
# top_nodes = sorted(betweenness_subset, key=betweenness_subset.get, reverse=True)[:20]
# for node in top_nodes:
#     df_node = df_road[df_road['LinkID'] == node] # subset dataframe for a specific node
    
#     plt.plot(df_node['RequestTimestamp'], df_node['SpeedBand'])
#     plt.xlabel('RequestTimestamp')
#     plt.ylabel('SpeedBand')
#     plt.title(f'SpeedBand vs. RequestTimestamp for LinkID {node}')
#     plt.show()
#%%
#画区域一天里的平均速度的变化情况和预计最堵的点的变化情况
import matplotlib.dates as mdates

# Group the DataFrame by RequestTimestamp and calculate the mean value of SpeedBand for each group
mean_speed = df_road[df_road['LinkID'].isin(node_df['LinkID'])].groupby('RequestTimestamp')['SpeedBand'].mean()

plt.show()

bottom_nodes = sorted(betweenness_subset, key=betweenness_subset.get, reverse=False)[:20]
top_nodes = sorted(betweenness_subset, key=betweenness_subset.get, reverse=True)[:20]
# Group the DataFrame by RequestTimestamp and calculate the mean value of SpeedBand for each group
mean_speed_top = df_road[df_road['LinkID'].isin(top_nodes)].groupby('RequestTimestamp')['SpeedBand'].mean()
mean_speed_bottom = df_road[df_road['LinkID'].isin(bottom_nodes)].groupby('RequestTimestamp')['SpeedBand'].mean()

# Plot the mean speed for the top nodes
fig, ax = plt.subplots()
ax.plot(mean_speed_top.index, mean_speed_top.values, color='red', label='Top Nodes')
ax.set_xlabel('RequestTimestamp (hour:minute)')
ax.set_ylabel('Mean SpeedBand(top 20 nodes)')

# Format the x-tick labels to show only the hour and minute
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Plot the mean speed for the bottom nodes
ax.plot(mean_speed_bottom.index, mean_speed_bottom.values, color='blue', label='Bottom Nodes')
ax.plot(mean_speed.index, mean_speed.values,color='green',label='All Nodes')
ax.legend()
plt.show()










