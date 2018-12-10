import requests
import json
import time

# see http://gisrevprxy.seattle.gov/arcgis/rest/services/SPU_EXT/SP_KC_PARCELS/MapServer/0 for details

map_server_url = 'http://gisrevprxy.seattle.gov/arcgis/rest/services/SPU_EXT/SP_KC_PARCELS/MapServer/0/query'
fields = ['*']  # ['OBJECTID', 'TAX_ID', 'zipcode' ,'LANDUSE_CD' ,'USEGRP' ,'STATE_CLASS' ,'ACREAGE' ,'Shape']

landuse_geojson = None

start = time.time()
for i in range(0, 619000, 1000):
    print('request: %d' % (i / 1000 + 1))
    
    query_format = '?where={}&outFields={}&returnGeometry={}&f=geojson'
    
    where = 'OBJECTID+BETWEEN+{}+AND+{}'.format(i, i + 999)
    r = requests.get(map_server_url + query_format.format(where, ','.join(fields), True))
    curr_landuse_geojson = r.json()
    
    if 'features' not in curr_landuse_geojson.keys():
        r = requests.get(map_server_url + query_format.format(where, ','.join(fields), False))
        curr_landuse_geojson = r.json()
        print('CAUGHT THE ERROR')

    if landuse_geojson is None:
        landuse_geojson = curr_landuse_geojson
    else:
        landuse_geojson['features'].extend(curr_landuse_geojson['features'])

    if (i + 1000) % 30000 == 0:
        print('%ss elapsed\n' % (time.time() - start))
        start = time.time()

with open('kingcounty_lu.geojson', 'w') as outfile:
    json.dump(landuse_geojson, outfile)
