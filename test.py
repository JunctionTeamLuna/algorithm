from requests import *

# res = post('http://localhost:8099/reservation', json={'userPositionLatitude': '36.20466303166719', 'userPositionLongitude': '129.32213620361614', 'destinationLatitude': '36.16499153927738', 'destinationLongitude': '129.2194916215949'}).json()
# print(res)

res = get('https://apis-navi.kakaomobility.com/v1/directions?origin=127.11015314141542,37.39472714688412&destination=127.10824367964793,37.401937080111644&waypoints=&priority=RECOMMEND&car_fuel=GASOLINE&car_hipass=false&alternatives=false&road_details=false', headers={'Authorization':'db995d33002c492f731c012107bf960f'})
print(res)