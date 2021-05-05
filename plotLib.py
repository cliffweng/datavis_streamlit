import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

def matPlot(axs):

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # generate some random test data
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    # plot violin plot
    axs[0].violinplot(all_data,
                    showmeans=False,
                    showmedians=True)
    axs[0].set_title('Violin plot')

    # plot box plot
    axs[1].boxplot(all_data)
    axs[1].set_title('Box plot')

    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(all_data))])
        ax.set_xlabel('Four separate samples')
        ax.set_ylabel('Observed values')

    # add x-tick labels
    plt.setp(axs, xticks=[y + 1 for y in range(len(all_data))],
            xticklabels=['x1', 'x2', 'x3', 'x4'])

def bokehPlot(p):
    from bokeh.io import show
    from bokeh.models import LogColorMapper
    from bokeh.palettes import Viridis6 as palette
    try:
        from bokeh.sampledata.unemployment import data as unemployment
        from bokeh.sampledata.us_counties import data as counties
    except:
        import bokeh
        bokeh.sampledata.download()
        from bokeh.sampledata.unemployment import data as unemployment
        from bokeh.sampledata.us_counties import data as counties

    palette = tuple(reversed(palette))

    counties = {
        code: county for code, county in counties.items() if county["state"] == "tx"
    }

    county_xs = [county["lons"] for county in counties.values()]
    county_ys = [county["lats"] for county in counties.values()]

    county_names = [county['name'] for county in counties.values()]
    county_rates = [unemployment[county_id] for county_id in counties]
    color_mapper = LogColorMapper(palette=palette)

    data=dict(
        x=county_xs,
        y=county_ys,
        name=county_names,
        rate=county_rates,
    )

    p.grid.grid_line_color = None
    p.hover.point_policy = "follow_mouse"

    p.patches('x', 'y', source=data,
            fill_color={'field': 'rate', 'transform': color_mapper},
            fill_alpha=0.7, line_color="white", line_width=0.5)

def plotlyPlot():
    import plotly.graph_objects as go
    from sklearn.neighbors import KNeighborsRegressor

    df = px.data.tips()
    X = df.total_bill.values.reshape(-1, 1)
    x_range = np.linspace(X.min(), X.max(), 100)

    # Model #1
    knn_dist = KNeighborsRegressor(10, weights='distance')
    knn_dist.fit(X, df.tip)
    y_dist = knn_dist.predict(x_range.reshape(-1, 1))

    # Model #2
    knn_uni = KNeighborsRegressor(10, weights='uniform')
    knn_uni.fit(X, df.tip)
    y_uni = knn_uni.predict(x_range.reshape(-1, 1))

    fig = px.scatter(df, x='total_bill', y='tip', color='sex', opacity=0.65)
    fig.add_traces(go.Scatter(x=x_range, y=y_uni, name='Weights: Uniform'))
    fig.add_traces(go.Scatter(x=x_range, y=y_dist, name='Weights: Distance'))
    return fig

def altairPlot():
    import altair as alt
    from vega_datasets import data

    source = data.movies.url

    stripplot =  alt.Chart(source, width=80).mark_circle(size=8).encode(
        x=alt.X(
            'jitter:Q',
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y('IMDB_Rating:Q'),
        color=alt.Color('Major_Genre:N', legend=None),
        column=alt.Column('Major_Genre:N',
            header=alt.Header(
                labelAngle=-90,
                titleOrient='top',
                labelOrient='bottom',
                labelAlign='right',
                labelPadding=3,
            ),
        ),
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    return stripplot
