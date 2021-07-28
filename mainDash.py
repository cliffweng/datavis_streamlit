import streamlit as st
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import seaborn as sns

import plotLib

@st.cache
def fetchDataset(dataset):
    return sns.load_dataset(dataset)

st.set_page_config(layout="wide")
st.header("Plotting with Streamlit")

genre = st.sidebar.radio("Pick a Plotting Package",('matplotlib','seaborn', 'plotly express', 'plotly', 'bokeh','altair','networkx','vega'))

if genre == 'matplotlib':
    st.header("matplotlib")
    st.write("https://matplotlib.org/stable/gallery/index.html")
    with st.echo():
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
        plotLib.matPlot(axs) # internal library
        st.write(fig)

elif genre == 'bokeh':
    st.header("bokeh")
    st.write("https://docs.bokeh.org/en/latest/docs/gallery.html")
    with st.echo():
        p = figure( title="Texas Unemployment, 2009", tools="pan,wheel_zoom,reset,hover,save",
            x_axis_location=None, y_axis_location=None,
            tooltips=[("Name", "@name"), ("Unemployment rate", "@rate%"), ("(Long, Lat)", "($x, $y)")])
        plotLib.bokehPlot(p) # internal library
        st.bokeh_chart(p, use_container_width=False)

elif genre == 'plotly express':
    st.header("plotly express")
    st.write("https://plotly.com/python/plotly-express/")
    with st.echo():
        import plotly.express as px
        df = px.data.iris()
        fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")
        st.plotly_chart(fig, use_container_width=True)
elif genre == 'plotly':
    st.header("plotly")
    st.write("https://plotly.com/python")
    with st.echo():
        fig = plotLib.plotlyPlot() # internal library
        st.plotly_chart(fig, use_container_width=True)
elif genre == 'altair':
    st.header("altair")
    st.write("https://altair-viz.github.io/gallery/")
    with st.echo():
        fig = plotLib.altairPlot()
        st.write(fig) # internal library
elif genre == 'networkx':
    st.header("networkx")
    st.write("https://networkx.org/documentation/stable/auto_examples/index.html")
    with st.echo():
        import networkx as nx
        # G = nx.grid_2d_graph(5, 5)  # 5x5 grid
        G = nx.lollipop_graph(4, 6)
        fig, ax = plt.subplots()
        nx.draw(G, ax = ax, with_labels = True)
        st.write(fig)
elif genre == 'vega':
    st.header("vega")
    st.write("https://vega.github.io/vega-lite/examples/")
    with st.echo():
        df = pd.DataFrame(np.random.randn(200, 3),columns=['a', 'b', 'c'])
        st.vega_lite_chart(df, {
        'mark': {'type': 'circle', 'tooltip': True},
        'encoding': {
            'x': {'field': 'a', 'type': 'quantitative'},
            'y': {'field': 'b', 'type': 'quantitative'},
            'size': {'field': 'c', 'type': 'quantitative'},
            'color': {'field': 'c', 'type': 'quantitative'}},
        }, True)
elif genre == 'seaborn':
    st.header("seaborn")
    st.write("https://seaborn.pydata.org/examples/index.html")
    with st.echo():
    
        penguins = fetchDataset("penguins")

        st.title("Penguins PairPlot")
        fig = sns.pairplot(penguins, hue="species")
        st.pyplot(fig)


