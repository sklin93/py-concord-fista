import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint

def plot_degree_distribution(G):
    degs = {}
    for n in G.nodes():
        deg = G.degree(n)
        # pprint(deg)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1
    items = sorted(degs.items())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([k for (k,v) in items], [v for (k,v) in  items ])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title("Degree Distribution")
    plt.show()
    fig.savefig("degree_distribution.png")

def ba_test(p=3404, m=30):
    ba = nx.random_graphs.barabasi_albert_graph(p, m)
    plot_degree_distribution(ba)
    return ba

# pos = nx.spring_layout(ba)
# nx.draw(ba, pos, node_size = 30)
# nx.draw(ba, pos)
# plt.show()

