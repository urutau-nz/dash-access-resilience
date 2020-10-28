import pathlib
import os

import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import geopandas as gpd


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import shapely.geometry
from dash.dependencies import Input, Output, State

amenities = ['medical_clinic', 'primary_school','supermarket']
amenity_names = {'medical_clinic':'Medical Clinic', 'primary_school':'Primary School','supermarket':'Supermarket'}

distance_column_names = ['base_supermarket','base_medical_clinic','base_primary_school','mean_supermarket','mean_medical_clinic','mean_primary_school','95_supermarket','95_medical_clinic','95_primary_school','5_supermarket','5_medical_clinic','5_primary_school']

cities = ['ch', 'wa', 'tx']
city_names = {'ch':'Christchurch, New Zealand', 'wa':'Seattle, USA', 'tx':'Houston, USA'}

hazards = ['tsunami', 'liquefaction', 'multi', 'hurricane']
hazard_names = {'tsunami':'Tsunami', 'liquefaction':'Liquefaction', 'multi':'Earthquake induced tsunami', 'hurricane':'Hurricane Inundation'}

demographics = ['total_pop', 'white', 'indigenous',	'asian', 'polynesian', 'latino', 'african_american']
demographic_names = {'total_pop':'All', 'white':'White', 'indigenous':'Maori',	'asian':'Asian', 'polynesian':'Polynesian', 'latino':'Latino', 'african_american':'African American'}

# app initialize
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=external_stylesheets,
    url_base_pathname='/access_resilience/',
)
server = app.server
app.config["suppress_callback_exceptions"] = True

app.title = 'Resilience Simulation'

# mapbox token
mapbox_access_token = open(".mapbox_token").read()

# Load data
df_dist = pd.read_csv('./data/ch_nearest_dist.csv')
df_dist[amenities] = df_dist[amenities]/1000
dff_dist = df_dist

df_dist_sim = pd.read_csv('./data/results_ch_tsunami.csv')
df_dist_sim[distance_column_names] = df_dist_sim[distance_column_names]/1000

destinations = pd.read_csv('./data/ch_destinations.csv')



# Assign color to legend
colors = ['#EA5138','#E4AE36','#1F386B','#507332']
colormap = {}
for ind, amenity in enumerate(amenities):
    colormap[amenity] = colors[ind]

pl_deep=[[0.0, 'rgb(253, 253, 204)'],
         [0.1, 'rgb(201, 235, 177)'],
         [0.2, 'rgb(145, 216, 163)'],
         [0.3, 'rgb(102, 194, 163)'],
         [0.4, 'rgb(81, 168, 162)'],
         [0.5, 'rgb(72, 141, 157)'],
         [0.6, 'rgb(64, 117, 152)'],
         [0.7, 'rgb(61, 90, 146)'],
         [0.8, 'rgb(65, 64, 123)'],
         [0.9, 'rgb(55, 44, 80)'],
         [1.0, 'rgb(39, 26, 44)']]


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.H6("Access Resilience"),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)

def brush(trace, points, state):
    inds = np.array(points.point_inds)
    if inds.size:
        selected = np.zeros(len(trace.x))
        selected[inds] = 1
        trace.marker.color = selected # now the trace marker color is a list of 0 and 1;
                                      # we have 0 at the position of unselected
                                      # points and 1 in the position of selected points

####################################################################################################################################################################################
''' ECDF '''
####################################################################################################################################################################################

def generate_ecdf_plot(amenity_select, dff_dist, hazard_select, demographic_select, demographic_compare, city_select, btn_recent, x_range=None):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    amenity = amenity_select
    hazard = hazard_select
    demograph = demographic_select
    state = city_select

    edes = pd.read_csv(r'data/ede_results.csv')

    if state =='ch' and hazard != None:
        df_dist_sim = pd.read_csv('./data/results_{}_{}.csv'.format(state, hazard))
    if state =='ch' and hazard == None:
        df_dist_sim = pd.read_csv('./data/results_{}_{}.csv'.format(state, 'tsunami'))
    elif state == 'wa':
        df_dist_sim = pd.read_csv('./data/results_wa_liquefaction.csv')
    elif state == 'tx':
        df_dist_sim = pd.read_csv('./data/results_tx_hurricane.csv')


    df_dist_sim[distance_column_names] = df_dist_sim[distance_column_names]/1000

    if x_range is None:
        x_range = [dff_dist[amenity].min(), dff_dist[amenity].max()]


    layout = dict(
        xaxis=dict(
            title="distance to nearest {} (km)".format(amenity_names[amenity]).upper(),
            range=(0,10),
            ),
        yaxis=dict(
            title="% of residents".upper(),
            range=(0,100),
            fixedrange=True,
            ),
        font=dict(size=15),
        legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=1,
        bgcolor='rgba(0,0,0,0)',
        font_size=15,
    ),
        dragmode="select",
        paper_bgcolor = 'rgba(255,255,255,1)',
		plot_bgcolor = 'rgba(0,0,0,0)',
        bargap=0.05,
        showlegend=True,
        margin={'t': 10},
        transition = {'duration': 500},

    )
    data = []
    if (demographic_compare == None and btn_recent == 'reset') or (demographic_compare == None and btn_recent == 'hazard') or (demographic_compare != None and btn_recent == 'reset'):
        # add the cdf for that amenity
        counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['base_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demograph])
        dx = bin_edges[1] - bin_edges[0]
        new_trace = go.Scatter(
                x=bin_edges, y=np.cumsum(counts)*dx*100,
                opacity=1,
                line_color='rgba(224, 145, 0, 1)',
                text=np.repeat(amenity,len(dff_dist[amenity])),
                hovertemplate = "%{y:.1f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
                hoverlabel = dict(font_size=20),
                name='Pre Hazard' if demographic_compare == None else 'Pre Hazard: {}'.format(demographic_names[demograph]),
                showlegend=True
                )
        data.append(new_trace)
        ede_trace = go.Scatter(
                x=[float(edes[(edes['dest_type']==amenity) & (edes['state']==state) & (edes['hazard']==hazard) & (edes['population_group']==demograph)]['base_ede'])/1000]*100, y=np.arange(0,100),
                opacity=1,
                line_color='rgba(224, 145, 0, 1)',
                text=np.repeat(amenity,len(dff_dist[amenity])),
                hovertemplate = "skip",
                hoverlabel = dict(font_size=20),
                name='Pre Hazard EDE' if demographic_compare == None else 'Pre Hazard EDE: {}'.format(demographic_names[demograph]),
                showlegend=True,
                line={'dash':'dash', 'width':2}
                )
        data.append(ede_trace)


    if btn_recent == 'hazard':
        counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['mean_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demograph])
        dx = bin_edges[1] - bin_edges[0]
        mean_trace = go.Scatter(
                x=bin_edges, y=np.cumsum(counts)*dx*100,
                opacity=1,
                line_color='rgba(224, 145, 0, 1)',
                hovertemplate = "After a hazard %{y:.1f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
                hoverlabel = dict(font_size=20),
                name='Post Hazard' if demographic_compare == None else 'Post Hazard: {}'.format(demographic_names[demograph]),
                showlegend=True
                )
        data.append(mean_trace)
        ede_trace = go.Scatter(
                x=[float(edes[(edes['dest_type']==amenity) & (edes['state']==state) & (edes['hazard']==hazard) & (edes['population_group']==demograph)]['sim_ede'])/1000]*100, y=np.arange(0,100),
                opacity=1,
                line_color='rgba(255, 0, 0, 0.5)',
                text=np.repeat(amenity,len(dff_dist[amenity])),
                hovertemplate = "skip",
                hoverlabel = dict(font_size=20),
                name='Post Hazard EDE' if demographic_compare == None else 'Post Hazard EDE: {}'.format(demographic_names[demograph]),
                showlegend=True,
                line={'dash':'dash', 'width':2}
                )
        data.append(ede_trace)

        counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['5_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demograph])
        dx = bin_edges[1] - bin_edges[0]
        lower_trace = go.Scatter(
                x=bin_edges, y=np.cumsum(counts)*dx*100,
                name='',
                #fill='tonexty',
                #fillcolor='rgba(255, 0, 0, 0.5)',
                mode='lines',
                line_color='rgba(255, 0, 0, 0.3)' if demographic_compare == None else 'rgba(224, 145, 0, 0)',
                hoverinfo='skip',
                showlegend=False
                )
        data.append(lower_trace)

        counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['95_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demograph])
        dx = bin_edges[1] - bin_edges[0]
        upper_trace = go.Scatter(
                x=bin_edges, y=np.cumsum(counts)*dx*100,
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.45)' if demographic_compare == None else 'rgba(224, 145, 0, 0.15)',
                mode='lines',
                line_color='rgba(255, 0, 0, 0.5)' if demographic_compare == None else 'rgba(39, 25, 168, 0.0)',
                hoverinfo='skip',
                name='90% Confidence Interval',
                showlegend=True
                )
        data.append(upper_trace)

        if demographic_compare == None:
            # histogram
            multiplier = 300 if amenity=='supermarket' else 150
            counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['mean_{}'.format(amenity)], bins=25, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demograph])
            opacity = []
            for i in bin_edges:
                if i >= x_range[0] and i <= x_range[1]:
                    opacity.append(0.6)
                else:
                    opacity.append(0.1)
            new_trace = go.Bar(
                    x=bin_edges, y=counts*multiplier,
                    marker_opacity=opacity,
                    marker_color=colormap[amenity],
                    hoverinfo="skip", hovertemplate="",
                    showlegend=False)
            data.append(new_trace)

    if btn_recent == 'reset' and demographic_compare == None:
        # histogram
        multiplier = 300 if amenity=='supermarket' else 150
        counts, bin_edges = np.histogram(dff_dist[amenity], bins=25, density=True, weights=dff_dist.population)
        opacity = []
        for i in bin_edges:
            if i >= x_range[0] and i <= x_range[1]:
                opacity.append(0.6)
            else:
                opacity.append(0.1)
        new_trace = go.Bar(
                x=bin_edges, y=counts*multiplier,
                marker_opacity=opacity,
                marker_color=colormap[amenity],
                hoverinfo="skip", hovertemplate="",
                showlegend=False)
        data.append(new_trace)

    if demographic_compare != None:
        if btn_recent == 'reset':
            # add the cdf for that amenity
            counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['base_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demographic_compare])
            dx = bin_edges[1] - bin_edges[0]
            new_trace = go.Scatter(
                    x=bin_edges, y=np.cumsum(counts)*dx*100,
                    opacity=1,
                    line_color= 'rgba(39, 25, 168, 1)',
                    text=np.repeat(amenity,len(dff_dist[amenity])),
                    hovertemplate = "%{y:.1f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
                    hoverlabel = dict(font_size=20),
                    name='Pre Hazard' if demographic_compare == None else 'Pre Hazard: {}'.format(demographic_names[demographic_compare]),
                    showlegend=True
                    )
            data.append(new_trace)
            ede_trace = go.Scatter(
                    x=[float(edes[(edes['dest_type']==amenity) & (edes['state']==state) & (edes['hazard']==hazard) & (edes['population_group']==demographic_compare)]['base_ede'])/1000]*100, y=np.arange(0,100),
                    opacity=1,
                    line_color='rgba(39, 25, 168, 1)',
                    text=np.repeat(amenity,len(dff_dist[amenity])),
                    hovertemplate = "skip",
                    hoverlabel = dict(font_size=20),
                    name='Pre Hazard EDE' if demographic_compare == None else 'Pre Hazard EDE: {}'.format(demographic_names[demographic_compare]),
                    showlegend=True,
                    line={'dash':'dash', 'width':2}
                    )
            data.append(ede_trace)

        if btn_recent == 'hazard':
            counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['mean_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demographic_compare])
            dx = bin_edges[1] - bin_edges[0]
            mean_trace = go.Scatter(
                    x=bin_edges, y=np.cumsum(counts)*dx*100,
                    opacity=1,
                    line_color= 'rgba(39, 25, 168, 1)',
                    hovertemplate = "After a hazard %{y:.1f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
                    hoverlabel = dict(font_size=20),
                    name='Post Hazard: {}'.format(demographic_names[demographic_compare]),
                    showlegend=True
                    )
            data.append(mean_trace)
            ede_trace = go.Scatter(
                    x=[float(edes[(edes['dest_type']==amenity) & (edes['state']==state) & (edes['hazard']==hazard) & (edes['population_group']==demographic_compare)]['sim_ede'])/1000]*100, y=np.arange(0,100),
                    opacity=1,
                    line_color='rgba(39, 25, 168, 1)',
                    text=np.repeat(amenity,len(dff_dist[amenity])),
                    hovertemplate = "skip",
                    hoverlabel = dict(font_size=20),
                    name='Post Hazard EDE' if demographic_compare == None else 'Post Hazard EDE: {}'.format(demographic_names[demographic_compare]),
                    showlegend=True,
                    line={'dash':'dash', 'width':2}
                    )
            data.append(ede_trace)

            counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['5_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demographic_compare])
            dx = bin_edges[1] - bin_edges[0]
            lower_trace = go.Scatter(
                    x=bin_edges, y=np.cumsum(counts)*dx*100,
                    name='',
                    #fill='tonexty',
                    #fillcolor='rgba(255, 0, 0, 0.5)',
                    mode='lines',
                    line_color='rgba(39, 25, 168, 0.3)' if demographic_compare == None else 'rgba(39, 25, 168, 0.0)',
                    hoverinfo='skip',
                    showlegend=False
                    )
            data.append(lower_trace)

            counts, bin_edges = np.histogram(df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False]['95_{}'.format(amenity)], bins=100, density = True, weights=df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][demographic_compare])
            dx = bin_edges[1] - bin_edges[0]
            upper_trace = go.Scatter(
                    x=bin_edges, y=np.cumsum(counts)*dx*100,
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.45)' if demographic_compare == None else 'rgba(39, 25, 168, 0.15)',
                    mode='lines',
                    line_color='rgba(255, 0, 0, 0.5)' if demographic_compare == None else 'rgba(39, 25, 168, 0.0)',
                    hoverinfo='skip',
                    name='90% Confidence Interval',
                    showlegend=True
                    )
            data.append(upper_trace)


    return {"data": data, "layout": layout}


####################################################################################################################################################################################
''' Map Data '''
####################################################################################################################################################################################
# generate base map for diff cities

def generate_map(amenity, dff_dest, hazard_select, demographic_select, city_select, btn_recent, x_range=None):
    """
    Generate map showing the distance to services and the locations of them
    :param amenity: the service of interest.
    :param dff_dest: the lat and lons of the service.
    :param x_range: distance range to highlight.
    :return: Plotly figure object.
    """

    hazard = hazard_select
    demograph = demographic_select
    state = city_select

    if state =='ch' and hazard != None:
        df_dist_sim = pd.read_csv('./data/results_{}_{}.csv'.format(state, hazard))
    if state =='ch' and hazard == None:
        df_dist_sim = pd.read_csv('./data/results_{}_{}.csv'.format(state, 'tsunami'))
    elif state == 'wa':
        df_dist_sim = pd.read_csv('./data/results_wa_liquefaction.csv')
    elif state == 'tx':
        df_dist_sim = pd.read_csv('./data/results_tx_hurricane.csv')

    df_dist_sim[distance_column_names] = df_dist_sim[distance_column_names]/1000

    destinations = pd.read_csv('./data/{}_destinations.csv'.format(state))
    dff_dest = destinations[destinations.dest_type == amenity]

    if state == 'wa':
        lat = 47.612608
        lon = -122.331748
    elif state == 'tx':
        lat = 29.754603
        lon = -95.363616
    elif state == 'ch':
        lat = -43.530918
        lon = 172.636744

    layout = go.Layout(
        clickmode="none",
        dragmode="zoom",
        showlegend=True,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat = lat, lon = lon),
            pitch=0,
            zoom=10.5,
            style="basic", #"dark", #
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            orientation="h",
            font=dict(color="black"),
            x=0.025,
            y=0,
            yanchor="bottom",
            #bgcolor='rgba(0,0,0,0)'
        ),
    )

    if x_range:
        # get the indices of the values within the specified range
        idx = df_dist_sim.index[df_dist_sim['mean_{}'.format(amenity)].between(x_range[0],x_range[1], inclusive=True)].tolist()
    else:
        idx = df_dist_sim.index.tolist()

    data = []

    data.append(go.Scattermapbox(
        lat=dff_dest["lat"],
        lon=dff_dest["lon"],
        mode="markers",
        marker={"color": colormap[amenity], "size": 7.5},
        name=amenity_names[amenity],
        hoverinfo="skip", hovertemplate="",
        opacity=0.75,
        visible=True,
        showlegend=True
    ))

    #Plot edges
    lats = np.load(r'data/{}_{}_lat_edges.npy'.format(state, hazard), allow_pickle=True)
    lons = np.load(r'data/{}_{}_lon_edges.npy'.format(state, hazard), allow_pickle=True)
    if btn_recent == 'hazard':
        #scatterplot of the amenity locations
        data.append(go.Scattermapbox(
            lat=dff_dest["lat"],
            lon=dff_dest["lon"],
            mode="markers",
            marker={"color": colormap[amenity], "size": 7.5},
            name=amenity_names[amenity],
            hoverinfo="skip", hovertemplate="",
            opacity=0.75,
            visible=True,
            showlegend=False
        ))
        data.append(go.Scattermapbox(lat=lats, lon=lons,
                                    visible=True,
                                    mode="lines",
                                    hoverinfo="skip", hovertemplate="",
                                    line={'color':'red','width':1},
                                    name='Closed Roads',
                                    showlegend=True
                                    ))

    if state == 'ch':
        with urlopen('https://raw.githubusercontent.com/urutau-nz/dash-x-minute-city/master/data/block_chc.geojson') as response:
            blocks = json.load(response)
            featureid = 'sa12018_v1'
    else:
        with urlopen('https://github.com/urutau-nz/dash-access-resilience/raw/main/data/{}_block.geojson'.format(state)) as response:
            blocks = json.load(response)
            featureid = 'geoid10'

    if btn_recent == 'hazard':
        df_temp = df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][['mean_{}'.format(amenity)]]
        df_iso_origins = df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==True][['id_orig']]
        df_origins = df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==False][['id_orig']]
        df_temp_iso = df_dist_sim.loc[df_dist_sim['isolated_{}'.format(amenity)]==True][['mean_{}'.format(amenity)]]
        df_temp_iso['mean_{}'.format(amenity)] = 100
        data.append(go.Choroplethmapbox(
            geojson=blocks,
            featureidkey='properties.{}'.format(featureid),
            locations= df_iso_origins['id_orig'].tolist(),
            z = df_temp_iso['mean_{}'.format(amenity)].tolist(),
            colorscale = 'Bluered',
            colorbar = dict(thickness=0, ticklen=0, bgcolor='rgba(0,0,0,0)'), zmin=0, zmax=7,
            marker_line_width=0, marker_opacity=0.6,
            visible=True,
            hovertemplate="Isolated" +
                            "<extra></extra>",
            #selectedpoints=idx,
        ))
        data.append(go.Choroplethmapbox(
            geojson=blocks,
            featureidkey='properties.{}'.format(featureid),
            locations= df_origins['id_orig'].tolist(),
            z = df_temp['mean_{}'.format(amenity)].tolist(),
            colorscale = pl_deep,
            colorbar = dict(thickness=15, ticklen=3, bgcolor='rgba(0,0,0,0)'), zmin=0, zmax=7,
            marker_line_width=0, marker_opacity=0.6,
            visible=True,
            hovertemplate="Distance: %{z:.2f}km<br>" +
                            "<extra></extra>",
            #selectedpoints=idx,
        ))
    elif btn_recent == 'reset':
        data.append(go.Choroplethmapbox(
            geojson=blocks,
            featureidkey='properties.{}'.format(featureid),
            locations=df_dist_sim['id_orig'].tolist(),
            z = df_dist_sim['base_{}'.format(amenity)].tolist(),
            colorscale = pl_deep,
            colorbar = dict(thickness=15, ticklen=3, bgcolor='rgba(0,0,0,0)'), zmin=0, zmax=7,
            marker_line_width=0, marker_opacity=0.6,
            visible=True,
            hovertemplate="Distance: %{z:.2f}km<br>" +
                            "<extra></extra>",
            #selectedpoints=idx,
        ))

    return {"data": data, "layout": layout}


####################################################################################################################################################################################
''' App Layout '''
####################################################################################################################################################################################

app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children=dcc.Markdown('''
                                    Reliable access is extremely important for urban resilience,
                                    especially in the face of a disaster. Use this tool to explore just how
                                    resilient your access to essential services is under a hazardous event.
                                    '''),
                                ),
                                build_graph_title("Select City"),
                                dcc.Dropdown(
                                    id="city-select",
                                    options=[
                                        {"label": city_names[i].upper(), "value": i}
                                        for i in cities
                                    ],
                                    value=cities[0],
                                ),
                                build_graph_title("Select Hazard"),
                                dcc.Dropdown(
                                    id="hazard-select",
                                    options=[{"label": hazard_names[i].upper(), "value": i} for i in hazards],
                                    value=hazards[0],
                                ),
                                build_graph_title("Select Amenity"),
                                dcc.Dropdown(
                                    id="amenity-select",
                                    options=[
                                        {"label": amenity_names[i].upper(), "value": i}
                                        for i in amenities
                                    ],
                                    value=amenities[0],
                                ),
                                build_graph_title("Compare Demographics"),
                                dcc.Dropdown(
                                    id="demographic-select",
                                    options=[
                                        {"label": demographic_names[i].upper(), "value": i}
                                        for i in demographics
                                    ],
                                    value=demographics[0],
                                    style={'float': 'left','margin': 'auto'}
                                ),
                                dcc.Dropdown(
                                    id="demographic-select1",
                                    options=[
                                        {"label": demographic_names[i].upper(), "value": i}
                                        for i in demographics
                                    ],
                                    value=None,
                                    style={'float': 'left','margin': 'auto'}
                                ),
                                html.Div([
                                    html.Button('Run Hazard Event', id='btn-hazard', n_clicks_timestamp=1),
                                    html.Button('Reset Hazard', id='btn-reset', n_clicks_timestamp=1),
                                    html.Div(id='container-button-timestamp')
                                ])
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # Access map
                        html.Div(
                            id="map-container",
                            children=[
                                build_graph_title("Explore how far people need to travel"),
                                dcc.Graph(
                                    id="map",
                                    figure={
                                        "layout": {
                                            "paper_bgcolor": "#192444",
                                            "plot_bgcolor": "#192444",
                                        }
                                    },
                                    config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":["lasso2d","select2d"],
                                    },
                                ),
                                build_graph_title("Select a distance range"),
                                dcc.Graph(id="ecdf",
                                    figure={
                                        "layout": {
                                            'clickmode': 'event+select',
                                            "paper_bgcolor": "#192444",
                                            "plot_bgcolor": "#192444",
                                            'mode': 'markers+lines',
                                        }
                                    },
                                    config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian'],
                                    },
                                ),
                            ],
                        ),
                        # ECDF
                        # html.Div(
                        #     id="ecdf-container",
                        #     className="six columns",
                        #     children=[
                        #         build_graph_title("Select a distance range"),
                        #         dcc.Graph(id="ecdf",
                        #             figure={
                        #                 "layout": {
                        #                     'clickmode': 'event+select',
                        #                     "paper_bgcolor": "#192444",
                        #                     "plot_bgcolor": "#192444",
                        #                     'mode': 'markers+lines',
                        #                 }
                        #             },
                        #             config={"scrollZoom": True, "displayModeBar": True,
                        #                     "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian'],
                        #             },
                        #         ),
                        #     ],
                        # ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="footer-row",
            children=[
                html.P(
                    id="footer-text",
                    children=dcc.Markdown('''
                        Thank you to the developers of [Dash and Plotly]
                        (https://plotly.com/dash/), whose work made this app possible.
                        '''
                    ),
                )
            ]
        )
    ]
)

#################################################################################
''' UPDATES '''
#################################################################################

# Update access map
@app.callback(
    Output("map", "figure"),
    [
        Input("amenity-select", "value"),
        # Input("ecdf", "relayoutData"),
        Input("ecdf", "selectedData"),
        Input("hazard-select", "value"),
        Input("demographic-select", "value"),
        Input("city-select", "value"),
        Input("btn-hazard", "n_clicks_timestamp"),
        Input("btn-reset", "n_clicks_timestamp")
    ],
)
def update_map(amenity_select, ecdf_selectedData, hazard_select, demographic_select, city_select, hazard_clicks, reset_clicks):

    btn_recent = 'reset'
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-hazard' in changed_id:
        btn_recent = 'hazard'
    elif 'btn-reset' in changed_id:
        btn_recent = 'reset'

    x_range = None
    # subset the desination df
    dff_dest = destinations[destinations.dest_type == amenity_select]

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    if prop_id == 'ecdf' and prop_type == "selectedData":
        if ecdf_selectedData:
            if 'range' in ecdf_selectedData:
                x_range = ecdf_selectedData['range']['x']
            else:
                x_range = [ecdf_selectedData['points'][0]['x']]*2

    return generate_map(amenity_select, dff_dest, hazard_select, demographic_select, city_select, btn_recent, x_range=x_range)


# Update ecdf
@app.callback(
    Output("ecdf", "figure"),
    [
        Input("amenity-select", "value"),
        Input("ecdf", "selectedData"),
        Input("hazard-select", "value"),
        Input("demographic-select", "value"),
        Input("demographic-select1", "value"),
        Input("city-select", "value"),
        Input("btn-hazard", "n_clicks_timestamp"),
        Input("btn-reset", "n_clicks_timestamp")
    ],
)
def update_ecdf(
    amenity_select, ecdf_selectedData, hazard_select, demographic_select, demographic_compare, city_select, hazard_clicks, reset_clicks
    ):
    btn_recent = 'reset'
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-hazard' in changed_id:
        btn_recent = 'hazard'
    elif 'btn-reset' in changed_id:
        btn_recent = 'reset'
    x_range = None
    # day = int(day)

    # subset data
    dff_dist = df_dist#[df_dist['service']==amenity_select]

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    if prop_id == 'ecdf' and prop_type == "selectedData":
        if ecdf_selectedData:
            if 'range' in ecdf_selectedData:
                x_range = ecdf_selectedData['range']['x']
            else:
                x_range = [ecdf_selectedData['points'][0]['x']]*2
    return generate_ecdf_plot(amenity_select, dff_dist, hazard_select, demographic_select, demographic_compare, city_select, btn_recent, x_range)


@app.callback(Output('hazard-select', 'options'),
              [Input('city-select', 'value')])
def change_hazards(city_select):
    if city_select == 'ch':
        hazards = ['tsunami', 'liquefaction', 'multi']
        hazard_names = {'tsunami':'Tsunami', 'liquefaction':'Liquefaction', 'multi':'Earthquake induced tsunami'}
    if city_select == 'wa':
        hazards = ['liquefaction']
        hazard_names = {'liquefaction':'Liquefaction'}
    if city_select == 'tx':
        hazards = ['hurricane']
        hazard_names = {'hurricane':'Hurricane Inundation'}
    return [{"label": hazard_names[i].upper(), "value": i} for i in hazards]

@app.callback(Output('hazard-select', 'value'),
              [Input('city-select', 'value')])
def change_hazard_value(city_select):
    if city_select == 'ch':
        hazards = ['tsunami', 'liquefaction', 'multi']
        hazard_names = {'tsunami':'Tsunami', 'liquefaction':'Liquefaction', 'multi':'Earthquake induced tsunami'}
    if city_select == 'wa':
        hazards = ['liquefaction']
        hazard_names = {'liquefaction':'Liquefaction'}
    if city_select == 'tx':
        hazards = ['hurricane']
        hazard_names = {'hurricane':'Hurricane Inundation'}
    return hazards[0]




# Running the server
if __name__ == "__main__":
    #app.run_server(debug=True, port=8051)
    app.run_server(port=9009)
