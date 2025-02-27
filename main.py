import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import folium_static,st_folium
from rasterio.io import MemoryFile
from rasterio.mask import mask
import numpy as np
import cv2
from folium.raster_layers import ImageOverlay
import pandas as pd
import plotly.express as px
from zona_utm import calcular_utm
from utils import color_map,value_to_class

st.header('Estudo de caso WebGIS')
st.write('')
st.write('')
st.sidebar.title('Menu')

poligono_subido = st.sidebar.file_uploader('Escolha o polígono a ser analisado')

raster_subido = st.sidebar.file_uploader('Escolha o raster a ser analisado (Mapbiomas)')

embargos = 'dados/embargos/embargos_ibama.parquet'
desmatamentos = 'dados/mapbiomas/dashboard_alerts-shapefile/mapbiomas_alertas.parquet'
tis = 'dados/tis_poligonais/tis.parquet'

if poligono_subido:
    poligono_analise = gpd.read_file(poligono_subido)

    @st.cache_resource
    def abrir_embargo():
        gdf_embargo = gpd.read_parquet(embargos)
        return gdf_embargo

    @st.cache_resource
    def abrir_desmatamento():
        gdf_desmat = gpd.read_parquet(desmatamentos)
        return gdf_desmat

    @st.cache_resource
    def abrir_tis():
        gdf_ti = gpd.read_parquet(tis)
        return gdf_ti
    
    gdf_embargo = abrir_embargo()

    gdf_desmat = abrir_desmatamento()

    gdf_ti = abrir_tis()

    gdf_embargo = gdf_embargo.drop(columns=['nom_pessoa','cpf_cnpj_i',
                            'cpf_cnpj_s','end_pessoa',
                            'des_bairro','num_cep','num_fone',
                            'data_tad','dat_altera','data_cadas',
                            'data_geom','dt_carga'])

    entrada_embargo = gpd.sjoin(gdf_embargo,poligono_analise,how='inner',predicate='intersects')
    entrada_embargo = gpd.overlay(entrada_embargo,poligono_analise,how='intersection')

    entrada_desmat = gpd.sjoin(gdf_desmat,poligono_analise,how='inner',predicate='intersects')
    entrada_desmat = gpd.overlay(entrada_desmat,poligono_analise,how='intersection')

    entrada_tis = gpd.sjoin(gdf_ti,poligono_analise,how='inner',predicate='intersects')
    entrada_tis = gpd.overlay(entrada_tis,poligono_analise,how='intersection')

    epsg_arquivo = calcular_utm(poligono_analise)

    area_desmat = entrada_desmat.dissolve(by=None)

    area_desmat = area_desmat.to_crs(epsg=epsg_arquivo)

    area_embargos = entrada_embargo.dissolve(by=None)

    area_embargos = area_embargos.to_crs(epsg=epsg_arquivo)

    area_tis = entrada_tis.dissolve(by=None)

    area_tis = area_tis.to_crs(epsg=epsg_arquivo)

    with MemoryFile(raster_subido.getvalue()) as memfile:
        with memfile.open() as src:

            if poligono_analise.crs != src.crs:
                poligono_analise = poligono_analise.to_crs(src.crs)

            geometries = poligono_analise.geometry
            out_image, out_transform = mask(src, geometries, crop=True)
            out_image = out_image[0]

            height, width = out_image.shape

            rgb_image = np.zeros((height,width,4),dtype=np.uint8)

            for value,color in color_map.items():
                rgb_image[out_image == value] = color

            resized_image = cv2.resize(rgb_image,(width,height),interpolation=cv2.INTER_NEAREST)

            min_x,min_y = out_transform * (0,0)
            max_x,max_y = out_transform * (width,height)

            bounds = [[min_y,min_x],[max_y,max_x]]

    col1,col2,col3 = st.columns(3)

    with col1:
        st.subheader('Área desmatada (ha)')
        if len(area_desmat) == 0:
            st.subheader('0')
        else:
            area_desmat['area'] = area_desmat.area / 10000
            st.subheader(str(round(area_desmat.loc[0,'area'],2)))

    with col2:
        st.subheader('Área embargada (ha)')
        if len(area_embargos) == 0:
            st.subheader('0')
        else:
            area_embargos['area'] = area_embargos.area / 10000
            st.subheader(str(round(area_embargos.loc[0,'area'],2)))

    with col3:
        st.subheader('Área de TIs (ha)')
        if len(area_tis) == 0:
            st.subheader('0')
        else:
            area_tis['area'] = area_tis.area / 10000
            st.subheader(str(round(area_tis.loc[0,'area'],2)))

    centroid_x,centroid_y = poligono_analise.centroid.x,poligono_analise.centroid.y

    m = folium.Map(location=[centroid_y,centroid_x],zoom_start=8,tiles='Esri World Imagery')

    ImageOverlay(
    image=resized_image,
    bounds=bounds,
    opacity=0.7,
    name='Mapbiomas coleção 9',
    interactive=True,
    cross_origin=False,
    zindex=1
    ).add_to(m)

    minx,miny,maxx,maxy = poligono_analise.total_bounds

    bounds = [[miny,minx],[maxy,maxx]]

    m.fit_bounds(bounds)

    def style_function_desmat(x): return{
        'fillColor':'red',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }

    entrada_desmat_geom = gpd.GeoDataFrame(entrada_desmat,columns=['geometry'])
    folium.GeoJson(entrada_desmat_geom,name='Área desmatada',style_function=style_function_desmat).add_to(m)

    def style_function_embargo(x): return{
        'fillColor':'orange',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }

    entrada_embargo_geom = gpd.GeoDataFrame(entrada_embargo,columns=['geometry'])
    folium.GeoJson(entrada_embargo_geom,name='Área embargada',style_function=style_function_embargo).add_to(m)


    def style_function_ti(x): return{
        'fillColor':'yellow',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }

    entrada_ti_geom = gpd.GeoDataFrame(entrada_tis,columns=['geometry'])
    folium.GeoJson(entrada_ti_geom,name='Área de TIs',style_function=style_function_ti).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m,width='100%')

    unique_values,counts = np.unique(out_image,return_counts=True)
    st.write('Áreas em hectares: ')
    for value,count in zip(unique_values,counts):
        class_name = value_to_class.get(value,"Unknown")
        area_ha = (count * 900) / 10000
        st.write(f"{class_name},{area_ha} (ha)")


    df_desmat = pd.DataFrame(entrada_desmat).drop(columns=['geometry'])

    df_embargo = pd.DataFrame(entrada_embargo).drop(columns=['geometry'])

    df_ti = pd.DataFrame(entrada_tis).drop(columns=['geometry'])

    col1_graf,col2_graf,col3_graf,col4_graf = st.columns(4)

    tema_grafico = col1_graf.selectbox('Selecione o tema do gráfico',options=['Embargo','Desmatamento','Terras Indígenas'])

    if tema_grafico == 'Embargo':
        df_analisado = df_embargo
    elif tema_grafico == 'Desmatamento':
        df_analisado = df_desmat
    elif tema_grafico == 'Terras Indígenas':
        df_analisado = df_ti

    tipo_grafico = col2_graf.selectbox('Selecione o tipo de gráfico',options=['box','bar','line','scatter','violin','histogram'],index=5)

    plot_func = getattr(px,tipo_grafico)

    x_val = col3_graf.selectbox('Selecione o eixo x do gráfico',options=df_analisado.columns,index=6)

    y_val = col4_graf.selectbox('Selecione o eixo y do gráfico',options=df_analisado.columns,index=5)

    plot = plot_func(df_analisado,x=x_val,y=y_val)

    st.plotly_chart(plot,use_container_width=True)

else:
    st.warning('Suba os arquivos para inciciar o WebGIS')
