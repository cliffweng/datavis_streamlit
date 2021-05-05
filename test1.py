import matplotlib.pyplot as plt
import networkx as nx

'''
G = nx.grid_2d_graph(5, 5)  # 5x5 grid
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")
nx.draw(H)
plt.show()
'''


G = nx.grid_2d_graph(5, 5)  # 5x5 grid
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

fig, ax = plt.subplots()
nx.draw(H, ax = ax, with_labels = True)
fig.show()
plt.show()
