import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import folium
import difflib
import geopandas as gpd

import unicodedata

#function to remove accents from states and municipios
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

#PROCESS SUI INPUT DATA, SAVE AS CSVs
def load_sui_data():
    folder  = 'sui_database_in'
    files = os.listdir(folder)
    u_codes = pd.read_excel('Listado de Agentes.xls')
    u_codes['Des Agente'] = [remove_accents(i) for i in list(u_codes['Des Agente'])]
    u_codes = u_codes.loc[u_codes['Estado'] == 'OPERACION']
    sorter = ['DISTRIBUCIÓN','COMERCIALIZACIÓN','GENERACIÓN','TRANSPORTE']
    u_codes['Actividad'] = u_codes['Actividad'].astype('category')
    u_codes['Actividad'].cat.set_categories(sorter, inplace=True)
    u_codes = u_codes.sort_values(['Actividad'])
    u_codes = u_codes.drop_duplicates(subset = ['Des Agente'], keep = 'first')
    code_dict = pd.Series(u_codes['Código'].values, index=u_codes['Des Agente']).to_dict()

    def match_empresa_to_code(row):
        """Accents, spaces, or other differences in the long Empresa names can throw off mapping.
        This function finds the closest string (slow), and just replaces it with a 4 digit acronym used by XM
        """

        empresa_str_in = row['Empresa']
        if empresa_str_in in code_dict.keys():
            return code_dict[empresa_str_in]
        else:
            empresa_str_matches = difflib.get_close_matches(empresa_str_in, list(code_dict.keys()), cutoff = 0.8, n = 1)
            if len(empresa_str_matches) > 0:
                return code_dict[empresa_str_matches[0]]
            else:
                print(empresa_str_in, 'no matches!')
                return empresa_str_in

    good_names = ['Departamento', 'Municipio', 'Empresa', 'Variable Calculada',
                   'Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5',
                   'Estrato 6', 'Total Residencial', 'Industrial', 'Comercial', 'Oficial',
                   'Otros', 'Total No Residencial']

    out_files = {}
    for f in files:
        print(f)
        in_f = pd.read_csv(os.path.join(folder, f), encoding = 'latin',
                           names = good_names + ['bogus_1','bogus_2'], skiprows = 1)
        in_f.fillna('missing', inplace = True)

        #there are some bad rows in the csv for Bogota, where there is an uneven number of columns, this block deals with that
        in_f_error = in_f[in_f['bogus_1'] != 'missing']
        in_f_error = in_f_error[in_f_error.columns[2:]]
        in_f_error.columns = good_names
        in_f_good = in_f[in_f ['bogus_1'] == 'missing'].drop(['bogus_1','bogus_2'], axis = 'columns')
        in_f = pd.concat([in_f_error, in_f_good], axis = 'rows')
        in_f = in_f.replace('missing',None)

        in_f = in_f.dropna(axis = 'index', how = 'any')
        in_f['year'] = in_f.loc[in_f['Departamento'] == 'Año', 'Municipio'].item()
        in_f['land_type'] = in_f.loc[in_f['Departamento'] == 'Ubicación', 'Municipio'].item() #keep year as a string so groupbys dont add it
        data_type = in_f.loc[in_f['Departamento'] == 'Reporte a Consultar', 'Municipio'].item()
        in_f['data_type'] = data_type

        out_f = in_f.loc[~in_f['Departamento'].isin(['Año','Período','Ubicación','Reporte a Consultar'])]

        out_f['Empresa'] = [remove_accents(i) for i in list(out_f['Empresa'])]
        out_f['Departamento'] = [remove_accents(i) for i in list(out_f['Departamento'])]
        out_f['Municipio'] = [remove_accents(i) for i in list(out_f['Municipio'])]

        out_f['Empresa'] = out_f.apply(match_empresa_to_code, axis = 1)

        #convert string columns to int if missing values (originally coded as 'ND') are present
        for c in out_f.columns:
            if 'ND' in list(in_f[c]):
                out_f[c] = out_f[c].replace('ND', 0)
                out_f[c] = pd.to_numeric(out_f[c], errors='ignore')

        if data_type not in out_files:
            out_files[data_type] = out_f
        else:
            out_files[data_type] = pd.concat([out_f, out_files[data_type]])

    out_files['Consumo'].to_csv('sui_database_out/consumption_kwh.csv')
    out_files['Factura Promedio'].to_csv('sui_database_out/average_bill.csv')
    out_files['Consumo Promedio'].to_csv('sui_database_out/average_consumption.csv')
    out_files['Tarifa Media'].to_csv('sui_database_out/average_rate.csv')
    out_files['Total Facturado'].to_csv('sui_database_out/total_billed.csv')
    out_files['Suscriptores'].to_csv('sui_database_out/subscribers.csv')
    out_files['Valor Consumo'].to_csv('sui_database_out/consumption_cost.csv')

    return out_files

def load_sui_data_from_csv():
    out_files = {}
    out_files['Consumo'] = pd.read_csv('sui_database_out/consumption_kwh.csv')
    out_files['Factura Promedio'] = pd.read_csv('sui_database_out/average_bill.csv')
    out_files['Consumo Promedio']= pd.read_csv('sui_database_out/average_consumption.csv')
    out_files['Tarifa Media'] = pd.read_csv('sui_database_out/average_rate.csv')
    out_files['Total Facturado'] = pd.read_csv('sui_database_out/total_billed.csv')
    out_files['Suscriptores'] = pd.read_csv('sui_database_out/subscribers.csv')
    out_files['Valor Consumo'] = pd.read_csv('sui_database_out/consumption_cost.csv')
    return out_files

#%%
#function to convert a complete long-df (with different entries for state, municipality, rural/urban, company)
#into one row per municipality (and state in case different municipalities have the same name)
def data_grouper(df, level = 'Municipio', land_type = 'Total', func = 'sum'):
    """
    Funciton to group data by Departamento or Municipio and perform an aggregating function.

    To Do
    -----
    -If 'func' is 'mean', then mean should be weighted for each Municipio based on total consumption. (i.e. two utilities in one Municipio with very different charachteristics and sizes)
    """
    df = df.loc[df['land_type'] == land_type]
    df = df.replace(0, np.nan)
    if level == 'Municipio':
        df_out = df.groupby(['Departamento','Municipio'], as_index = False).agg(func)
    elif level == 'Departamento':
        df_out = df.groupby('Departamento', as_index = False).agg(func)

    return df_out

#%%
#READ IN GEOJSONS AND CLEAN UP
muni_shape_in = gpd.read_file('colombia-municipios-simplified/colombia-municipios.shp', encoding = 'latin')
muni_shape_in.rename({'NOMBRE_DPT':'Departamento','NOMBRE_MPI':'Municipio'}, axis = 'columns', inplace = True)
muni_shape_in['Departamento'] = [remove_accents(i) for i in list(muni_shape_in['Departamento'])]
muni_shape_in['Departamento'] = muni_shape_in['Departamento'].replace('SANTAFE DE BOGOTA D.C', 'BOGOTA')
muni_shape_in['Municipio'] = muni_shape_in['Municipio'].replace('SANTAFE DE BOGOTA D.C.', ' D.C.')
muni_shape_in['Municipio'] = [remove_accents(i) for i in list(muni_shape_in['Municipio'])]
muni_shape_in.crs = {'init':'epsg:4326'}

#merging
dept_shape_in = gpd.read_file('colombia-departamento-simplified.json', encoding = 'latin')
dept_shape_in.rename({'NOMBRE_DPT':'Departamento','NOMBRE_MPI':'Municipio'}, axis = 'columns', inplace = True)
dept_shape_in['Departamento'] = [remove_accents(i) for i in list(dept_shape_in['Departamento'])]
dept_shape_in['Departamento'] = dept_shape_in['Departamento'].replace('SANTAFE DE BOGOTA D.C', 'BOGOTA')
dept_shape_in.crs = {'init':'epsg:4326'}

#functions to merge data df with shape gdf
def muni_shape_merger(df):
    out = muni_shape_in.merge(df, on = ['Departamento','Municipio'], how = 'inner')
    out.fillna(0, inplace = True)
    return out

def dept_shape_merger(df):
    out = dept_shape_in.merge(df, on = ['Departamento'], how = 'outer')
    out.fillna(0, inplace = True)
    return out

def hexcolormapper(df, column, colorscale):
            bins = np.percentile(df[column], np.linspace(0,100,15))
            binned = np.digitize(df[column], bins)
            norm = matplotlib.colors.Normalize(vmin=min(binned), vmax=max(binned), clip=False)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=colorscale)
            df[f'{column}_color'] = np.vectorize(lambda x: mcolors.to_hex(mapper.to_rgba(x)))(binned)
            return df

def mapper(data_df, columns, columns_cmap = None, level = 'Municipio'):
    """ Function to create a folium map with given levels of input.

    Parameters
    ----------
    data_df : pandas.DataFrame
        dataframe that has been grouped to display a metric (i.e. number of subscribers
        or total consumption). Rows are based on a geographic jurisdiction (i.e. Departamento or Municipio)
        which should be included in a column.

    columns : list
        list of columns representing rates (i.e. Estrato 1, Total Residencial, Comercial, etc.)
        each column will be its own choropleth layer. Can also be a string for pregenerated columns.

    columns_cmap : list
        list of equal length to columns, with a colar scheme to use for each column

    level : 'Departamento' or 'Municipio'
        The jurisdictional level to be plotting
    """

    if columns == 'estrata_columns':
        columns = ['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6', 'Total Residencial']
        columns_cmap = ['Greens'] * 7

    elif columns == 'all_columns':
        columns = ['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6',
                       'Total Residencial', 'Industrial','Comercial','Oficial','Otros','Total No Residencial']
        columns_cmap = ['Greens'] * 7 + ['Reds','Blues','Purples','Purples','Reds']

    elif columns == 'sec_columns':
        columns = ['Total Residencial','Industrial','Comercial']
        columns_cmap = ['Greens', 'Reds', 'Blues']

    elif columns == 'estrata_columns_lim':
        columns = ['Estrato 1','Estrato 4','Estrato 6']
        columns_cmap = ['Greens'] * 3

    elif columns == 'total_residencial':
        columns = ['Total Residencial']
        columns_cmap = ['Greens']

    if level == 'Municipio':
        data_gdf = muni_shape_merger(data_df)
    elif level == 'Departamento':
        data_gdf = dept_shape_merger(data_df)

    m = folium.Map(location=[4.57, -74.29], zoom_start=5, tiles = 'stamentoner', width = '60%',
                   min_lat = -9, max_lat = 18, min_lon = -84, max_lon = -59, min_zoom = 5, max_zoom = 9,
           attr = 'Data Source: SUI')

    count = 0
    local_gdfs = []
    for c in columns:
        cmap = columns_cmap[count]
        count += 1
        data_gdf = hexcolormapper(data_gdf, c, cmap)
        if level == 'Municipio':
            local_gdf = data_gdf[[f'{c}',f'{c}_color','geometry','Departamento','Municipio']]
            local_gdf.columns = ['data','color','geometry', 'Departamento','Municipio'] #folium doesn't like when dfs in a loop have different column names
        if level == 'Departamento':
            local_gdf = data_gdf[[f'{c}',f'{c}_color','geometry','Departamento']]
            local_gdf.columns = ['data','color','geometry', 'Departamento']
        local_gdfs.append(local_gdf)

    fgs = []
    count = 0
    for c in columns:
        local_gdf = local_gdfs[count]
        if count == 0:
            on_switch = True
        else:
            on_switch = False
        count +=1

        fg_ = folium.FeatureGroup(name=c, show = on_switch)
        for index, row in local_gdf.iterrows():
            geojson_ = folium.GeoJson(local_gdf.iloc[index:index+1],
                       style_function = lambda feature: {
                           'fillColor': feature['properties']['color'],
                           'fillOpacity' : 0.7,
                           'color': '#000000',
                           'weight':0.2
                           })

            if level == 'Departamento':
                popup_ = folium.Popup(
                      f"<b>Dep:</b> {row['Departamento']}<br>"
                      f"<b>Data:</b> {int(row['data'])}<br>"
                      )
#            elif level == 'Municipio':
#                popup_ = folium.Popup(
#                      f"<b>Dep:</b> {row['Departamento']}<br>"
#                      f"<b>Muni:</b> {row['Municipio']}<br>"
#                      f"<b>Data:</b> {int(row['data'])}<br>"
#                      )
                popup_.add_to(geojson_)
            geojson_.add_to(fg_)

        fgs.append(fg_)

    for fg in fgs:
        m.add_child(fg)

#         folium.Choropleth(
#                     geo_data=shape_df,
#                     name=c,
#                     data=data_df,
#                     columns=[key,c],
#                     key_on=f'feature.properties.{key}',
#                     fill_color=cmap,
#                     fill_opacity=0.5,
#                     line_opacity=0.5,
#                     legend_name=c,
#                     bins=9,
#                     legend = False
#                     ).add_to(m)

    m.add_child(folium.map.LayerControl(collapsed = False, autoZIndex = True))
    return m

#%%
def land_type_aggregator(df):
    """
    Accepts an ungrouped df (i.e. sui_data['Consumo']) and returns a df with rows for Municipios
    and columns for sum by land type (Centro, Urbano, Rural)
    """
    groupeddf = df.groupby(['Departamento','Municipio','land_type'], as_index = False).sum()

    municipio_lists = []
    for d in set(groupeddf['Departamento']):
        d_df = groupeddf.loc[groupeddf['Departamento'] == d]
        for m in set(d_df['Municipio']):
            m_df = d_df.loc[d_df['Municipio'] == m]

            rural_df = m_df.loc[m_df['land_type'] == 'Rural']
            rural_val = int(rural_df['Total Residencial'].item()) + int(rural_df['Total No Residencial'].item())

            centro_df = m_df.loc[m_df['land_type'] == 'Centro Poblado']
            centro_val = int(centro_df['Total Residencial'].item()) + int(centro_df['Total No Residencial'].item())

            urban_df = m_df.loc[m_df['land_type'] == 'Urbano']
            urban_val = int(urban_df['Total Residencial'].item()) + int(urban_df['Total No Residencial'].item())

            municipio_lists.append([d, m, rural_val, centro_val, urban_val])

    land_type_df = pd.DataFrame(municipio_lists, columns = ['Departamento','Municipio', 'Rural', 'Centro Poblado', 'Urbano'])

    return land_type_df

#%%

def load_sui_tariff():
    """
    Still unsure what exactly is in this database, but it seems related to tariff components.
    Unused for now.
    """
    folder  = 'colombia_eda/sui_tariffs_in'
    files = os.listdir(folder)

    #good_names = ['Departamento', 'Municipio', 'Empresa', 'Variable Calculada',
    #               'Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5',
    #               'Estrato 6', 'Total Residencial', 'Industrial', 'Comercial', 'Oficial',
    #               'Otros', 'Total No Residencial']

    tariff_dfs = []
    for f in files:
        in_f = pd.read_csv(os.path.join(folder, f), encoding = 'latin')
        tariff_dfs.append(in_f)
    tariff_df = pd.concat(tariff_dfs)
    tariff_df = tariff_df.dropna(thresh = 8)
    tariff_df['Fecha'] = pd.to_datetime(tariff_df['Fecha'])
    tariff_df['Empresa'] = [remove_accents(i) for i in list(tariff_df['Empresa'])]
    tariff_df['Mercado'] = [remove_accents(i) for i in list(tariff_df['Mercado'])]
    tariff_df = tariff_df.sort_values(by=['Empresa','Fecha','Mercado'], ascending = [True, False, True])

    recent_rates = []
    for e in set(tariff_df['Empresa']):
        e_df = tariff_df.loc[tariff_df['Empresa'] == e]
        for m in set(e_df['Mercado']):
            m_df = e_df.loc[e_df['Mercado'] == m]
            if len(m_df) > 0:
                m_df_list = list(m_df.iloc[0])
                recent_rates.append(m_df_list)
    recent_rate_df = pd.DataFrame(recent_rates, columns = tariff_df.columns)

    recent_rate_df.rename({'Gm':'generation_charge',
                           'i':'transmission_charge',
                           'j':'comercialization?',
                           })

    recent_rate_df.to_csv('/Users/skoebric/Dropbox/GitHub/colombia_eda/sui_tariffs_out/recent_tariff_structure.csv')

#%%
def demand_8760_getter(input_codes = None):
    """
    Returns a long-df with an 8760 for each Empresa's 2018 Demand.
    Empresa
    """
    retail_df = pd.read_excel('Demanda_por_OR_2018.xls')
    nonreg_df1 = pd.read_excel('Demanda_Comercial_Por_Comercializador_2018SEM1.xls')
    nonreg_df2 = pd.read_excel('Demanda_Comercial_Por_Comercializador_2018SEM2.xls')
    demand_df = pd.concat([retail_df, nonreg_df1, nonreg_df2], axis = 'rows')
    demand_df = demand_df.rename({'Codigo':'Empresa'}, axis = 'columns')

    if input_codes is not None:
        demand_df = demand_df.loc[demand_df['Empresa'].isin(input_codes)]

    demand_df['Fecha'] = pd.to_datetime(demand_df['Fecha'])
    demand_melt_df = pd.melt(demand_df, id_vars = ['Empresa','Fecha'], value_vars = [str(i) for i in range(0,24)],
                             var_name = 'hour', value_name = 'demand')
    demand_melt_df['hour'] = pd.to_numeric(demand_melt_df['hour'])
    demand_melt_df['demand'] = demand_melt_df['demand'] / 1000 #kw to MW
    return demand_melt_df

#%%

def mean_8760_getter(df, demand_melt_df):
    """
    Accepts an ungrouped dataframe with different rate classes as columns and Empresas (companies) as rows,
    returns a long-df with a scaled 8760 of consumpton for each rate class.
    """
    df = df.loc[df['land_type'] == 'Total']
    pctdf = df.groupby(['Empresa']).sum()
    pctdf = pctdf.drop(['Total Residencial','Total No Residencial'], axis = 'columns')
    pctdf = pctdf.clip(lower = 0) #there are a few weird outliers with negative consumption
    pctdf['Total Demand'] = pctdf.sum(axis = 1)
    #for each utility, calculate the percentage of load by
    for c in [f"Estrato {i}" for i in range (1, 7)] + ['Comercial','Industrial','Oficial','Otros']:
        pctdf[c] = pctdf[c] / pctdf['Total Demand']
    pctdf = round(pctdf.drop(['Total Demand'], axis = 'columns'),3)

    utility_demand_dict = {}
    for u in set(demand_melt_df['Empresa']):
        u_df = demand_melt_df.loc[demand_melt_df['Empresa'] == u]
        total_demand = u_df['demand'].sum()
        scaler = total_demand / 100000 #scale demand profile to equal 100000 MW of total demand in a year
        u_df['demand_scaled'] = u_df['demand'] / scaler
        utility_demand_dict[u] = u_df

    mean_demand_df = pd.DataFrame()
    #calculate the 10 average
    for c in pctdf.columns:
        c_df = pctdf.sort_values(c, ascending = False) #sort df by column to find the Empresas with the largest fraction of load from the given column
        top_u = [i for i in c_df.index if i in utility_demand_dict.keys()] #only include Empresas that we have 8760 for (we don't have some non regulated ones)
        c_df = c_df.loc[c_df.index.isin(top_u)]
        top_u = c_df.head(3).index
        top_u_df = pd.concat([utility_demand_dict[i] for i in top_u], axis = 'rows')
        top_u_df = top_u_df.groupby(['Fecha','hour'], as_index = False).sum()
        top_u_df['Tariff'] = c
        mean_demand_df = pd.concat([mean_demand_df, top_u_df], axis = 'rows')
    return pctdf, mean_demand_df