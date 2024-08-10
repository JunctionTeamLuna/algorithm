from requests import *

res = post('http://localhost:8099/reservation', json={'userPositionLatitude': '36.20466303166719', 'userPositionLongitude': '129.32213620361614', 'destinationLatitude': '36.16499153927738', 'destinationLongitude': '129.2194916215949'}).json()
print(res)