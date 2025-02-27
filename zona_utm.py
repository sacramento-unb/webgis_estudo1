import geopandas as gpd
import math

def calcular_utm(gdf):
    gdf = gdf.dissolve(by=None)
    centroid = gdf.centroid.iloc[0]
    zona = (math.floor((centroid.x+180)/6))+1
    if centroid.y > 0:
        epsg = zona + 32600
    else:
        epsg = zona + 32700
    return epsg

if __name__ == '__main__':
    calcular_utm(gpd.read_file('/dados/car.geojson'))