'''
Mitchell Anderson
28/05/2021

take old results files and create cdf/histogram data
Columns: city | Hazard | Service | Ethnic | Pop_perc | Distance | pop_perc_cum
'''

import pandas as pd
import numpy as np
from tqdm import tqdm

# set up lists
citys, hazards, dest_types, ethnics, pop_percs, distances, pop_perc_cum = [], [], [], [], [], [], []

# file names to loop through
files = ['ch_liquefaction', 'ch_tsunami', 'ch_multi', 'tx_hurricane', 'wa_liquefaction']
services = ['supermarket', 'primary_school', 'medical_clinic']
ch_ethnics = ['all', 'white', 'maori', 'asian', 'polynesian', 'meela']
us_ethnics = ['all','white','american_indian','asian','native_hawaiian','latino','african_american']

# loop through files
for file in tqdm(files):
    df = pd.read_csv(r'data/results_{}.csv'.format(file))
    # pick ethnicities list and set city name
    if file[0] == 'c':
        ethnicitys = ch_ethnics
        city = 'christchurch'
    elif file[0] == 'w':
        ethnicitys = us_ethnics
        city = 'seattle'
    elif file[0] == 't':
        ethnicitys = us_ethnics
        city = 'houston'
    if (file[0:4] == 'ch_t') or (file[0:4] == 'ch_m'):
        hazard_list = [file[3:]]
    else:
        hazard_list = ['base', file[3:]]
    for hazard in hazard_list:
        if hazard != 'base':
            column_name = 'mean'
        else:
            column_name = 'base'
        for service in services:
            for ethnic in ethnicitys:
                df['{}_{}'.format(column_name, service)] = df['{}_{}'.format(column_name, service)].fillna(30000)
                # make pop_perc, distances, pop_perc_cum
                counts, bin_edges = np.histogram(np.array(df['{}_{}'.format(column_name, service)]), bins=50, density = True, weights=df[ethnic])
                dx = bin_edges[1] - bin_edges[0]
                pop_perc_cum = pop_perc_cum + list(np.cumsum(counts)*dx*100)
                distances = distances + list(bin_edges[0:-1])
                for i in np.arange(0, len(list(np.cumsum(counts)*dx*100))):
                    if i == 0:
                        pop_percs.append(list(np.cumsum(counts)*dx*100)[i])
                    else:
                        pop_percs.append(list(np.cumsum(counts)*dx*100)[i] - list(np.cumsum(counts)*dx*100)[i-1])
                ethnics = ethnics + [ethnic]*50
                dest_types = dest_types + [service]*50
                hazards = hazards + [hazard]*50
                citys = citys + [city]*50

cdf_df = pd.DataFrame(list(zip(citys, hazards, dest_types, ethnics, distances, pop_perc_cum, pop_percs)), columns = ['city', 'hazard', 'service', 'ethnicity', 'distance', 'pop_perc_cum', 'pop_perc'])
cdf_df.to_csv(r'data/d3_cdf_data.csv')

