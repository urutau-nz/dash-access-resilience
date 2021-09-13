'''
Mitchell Anderson
28/05/2021

take old results files and create cdf/histogram data
Columns: city | Hazard | Service | Ethnic | Pop_perc | Distance | pop_perc_cum | pop | 
'''

import pandas as pd
import numpy as np
from tqdm import tqdm

# set up lists
citys, hazards, dest_types, ethnics, pop_percs, distances, pop_perc_cum, pops = [], [], [], [], [], [], [], []

# file names to loop through
files = ['ch_liquefaction', 'ch_tsunami', 'ch_multi', 'wa_liquefaction'] #'tx_hurricane', 
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
                if ethnic == 'all':
                    ethnic_col = 'total_pop'
                elif ethnic == 'maori':
                    ethnic_col = 'indigenous'
                elif ethnic == 'meela':
                    ethnic_col = 'latino'
                elif ethnic == 'american_indian':
                    ethnic_col = 'indigenous'
                elif ethnic == 'native_hawaiian':
                    ethnic_col = 'polynesian'
                else:
                    ethnic_col = ethnic
                sub_pop = df[ethnic_col].sum()
                df['mean_{}'.format(service)] = df['mean_{}'.format(service)].fillna(-999)
                df['{}_{}'.format(column_name, service)] = df['{}_{}'.format(column_name, service)].fillna(30000)
                # make pop_perc, distances, pop_perc_cum
                bins = [-1000] + list(np.arange(0, df['{}_{}'.format(column_name, service)].max(), df['{}_{}'.format(column_name, service)].max()/51))
                counts, bin_edges = np.histogram(np.array(df['{}_{}'.format(column_name, service)]), bins=bins, density = True, weights=df[ethnic_col])
                dx = bin_edges[1] - bin_edges[0]
                pop_perc_cum = pop_perc_cum + list(np.cumsum(counts)*dx*100)
                distances = distances + list(bin_edges[0:-1])
                for i in np.arange(0, len(list(np.cumsum(counts)*dx*100))):
                    if i == 0:
                        pop_percs.append(list(np.cumsum(counts)*dx*100)[i])
                        pops = pops + [(list(np.cumsum(counts)*dx*100)[i]/100)*sub_pop]
                    else:
                        pop_percs.append(list(np.cumsum(counts)*dx*100)[i] - list(np.cumsum(counts)*dx*100)[i-1])
                        pops = pops + [((list(np.cumsum(counts)*dx*100)[i] - list(np.cumsum(counts)*dx*100)[i-1])/100)*sub_pop]
                distances = distances[:-1]
                pops = pops[:-1]
                pop_perc_cum = pop_perc_cum[:-1]
                ethnics = ethnics + [ethnic]*50
                dest_types = dest_types + [service]*50
                hazards = hazards + [hazard]*50
                citys = citys + [city]*50

cdf_df = pd.DataFrame(list(zip(citys, hazards, dest_types, ethnics, distances, pop_perc_cum, pop_percs, pops)), columns = ['city', 'hazard', 'service', 'ethnicity', 'distance', 'pop_perc_cum', 'pop_perc', 'pop'])
cdf_df['distance'] = cdf_df['distance']/1000
cdf_df.to_csv(r'data/d3_cdf_data_test.csv')

