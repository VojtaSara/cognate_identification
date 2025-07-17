import pandas as pd
import networkx as nx
import plotly.graph_objects as go


# Define only the columns we care about
columns = ["concept id", "lang 1", "word 1", "lang 2", "word 2", "translit 1", "translit 2"]

# Read only first 7 columns, skip any extras
df = pd.read_csv("CogNet-v2.0.tsv", sep="\t", names=columns, usecols=range(len(columns)), skiprows=1)


# Specify concept ID
concept_id = "n02472293"

# Filter by concept ID
filtered = df[df["concept id"] == concept_id]

# Create graph
G = nx.Graph()

# Add nodes and edges
for _, row in filtered.iterrows():
    lang1_word = f"{row['lang 1']}: {row['word 1']}"
    lang2_word = f"{row['lang 2']}: {row['word 2']}"
    
    G.add_node(lang1_word)
    G.add_node(lang2_word)
    G.add_edge(lang1_word, lang2_word)

# Layout
pos = nx.spring_layout(G, seed=42, k=1)

# Extract edge and node positions
edge_x = []
edge_y = []
for src, dst in G.edges():
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x = []
node_y = []
labels = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    labels.append(node)

# Create edge trace
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Create node trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=labels,
    textposition="top center",
    marker=dict(
        showscale=False,
        color='#1f77b4',
        size=10,  # smaller nodes
        line_width=2
    )
)

# Plot
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'Cognate Network for Concept {concept_id}',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))


# Save figure to image
fig.write_image("concept_graph.png", width=1000, height=800)

