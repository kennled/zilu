import matplotlib.dates as mdates
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

node_df = pd.read_csv('betweenness_land_find_road_200.csv')
# Read the CSV file
df_road = pd.read_csv('../2019.02.19.csv')

# Convert the 'RequestTimestamp' column to a pandas datetime object
df_road['RequestTimestamp'] = pd.to_datetime(df_road['RequestTimestamp'], format='%Y.%m.%d.%H.%M.%S')
# Group the DataFrame by RequestTimestamp and calculate the mean value of SpeedBand for each group
mean_speed = df_road[df_road['LinkID'].isin(node_df['LinkID'])].groupby('RequestTimestamp')['SpeedBand'].mean()

# Sort the nodes based on their betweenness centrality
node_df = node_df.sort_values(by='my_betweenness', ascending=False)

# Get the top 20 and bottom 20 nodes
top_nodes = node_df.head(100)['LinkID'].tolist()
bottom_nodes = node_df.tail(100)['LinkID'].tolist()

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
ax.plot(mean_speed.index, mean_speed.values, color='green', label='All Nodes')
ax.legend()
plt.show()
