from typing import Optional
from fastapi import FastAPI
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    # "http://192.168.0.13:3000", # url을 등록해도 되고
    "*" # private 영역에서 사용한다면 *로 모든 접근을 허용할 수 있다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # cookie 포함 여부를 설정한다. 기본은 False
    allow_methods=["*"],    # 허용할 method를 설정할 수 있으며, 기본값은 'GET'이다.
    allow_headers=["*"],	# 허용할 http header 목록을 설정할 수 있으며 Content-Type, Accept, Accept-Language, Content-Language은 항상 허용된다.
)

# userStartLatitude, userStartLongitude = 36.20466303166719, 129.32213620361614
# userDestination = [36.16499153927738, 129.2194916215949]
# start_point = np.array([userStartLatitude, userStartLongitude], dtype=np.float64)
# end_point = np.array(userDestination, dtype=np.float64)

def readCSV(f):
    with open(f, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = [row for row in csv_reader]
    return rows


# 방향 체크 함수
def is_facing_destination(bus_position, bus_direction, destination_position):
    # 버스 방향과 도착지 위치를 비교하여 도착지를 향하는지 판단
    bus_to_destination_vector = destination_position - bus_position
    angle = np.arctan2(bus_to_destination_vector[1], bus_to_destination_vector[0])
    
    if bus_direction == 1:  # 시계방향
        return -np.pi/2 <= angle <= np.pi/2
    else:  # 반시계방향
        return angle < -np.pi/2 or angle > np.pi/2


# CSV 파일 읽기
DRTBus = readCSV('./data/drt.csv')
touristAttraction = readCSV('./data/tourist_attraction.csv')
transportation = readCSV('./data/transportation.csv')

# 관광지 좌표 추출
touristAttractionPocs = np.array(list(map(lambda t: [float(t[1]), float(t[2])], touristAttraction)), dtype=np.float64)
transportationPocs = np.array(list(map(lambda t: [float(t[1]), float(t[2])], transportation)), dtype=np.float64)

# Convex Hull 계산
hull = ConvexHull(touristAttractionPocs)

# DRT 버스 설정 (시계방향 2대, 반시계방향 2대)
bus_positions = touristAttractionPocs[hull.vertices[:4]]  # 현재 DRT bus 위치
bus_directions = [1, 1, -1, -1]  # 1은 시계방향, -1은 반시계방향

class Item(BaseModel):
    userPositionLatitude: str
    userPositionLongitude: str
    destinationLatitude: str
    destinationLongitude: str

@app.post("/reservation/")
def reservation(item: Item):
    global bus_positions
    
    start_point = np.array([item.userPositionLatitude, item.userPositionLongitude], dtype=np.float64)
    destination_point = np.array([item.destinationLatitude, item.destinationLongitude], dtype=np.float64)
    
    min_distance = float('inf')
    closest_bus_index = None    
    
    for i in range(len(bus_positions)):
        # 사용자 출발지와 버스 간의 거리 계산
        distance_to_start = np.linalg.norm(start_point - bus_positions[i])

        # 도착지 방향 확인 및 가장 가까운 버스 찾기
        if is_facing_destination(bus_positions[i], bus_directions[i], destination_point) and distance_to_start < min_distance:
            min_distance = distance_to_start
            closest_bus_index = i

    if closest_bus_index is not None:
        closest_bus_position = bus_positions[closest_bus_index]
    print(closest_bus_index, closest_bus_position)
    
    
    # -------- 여기까지 in -------- #
    
    
    hull_vertices = hull.vertices
    hull_edges = [(hull_vertices[i], hull_vertices[i+1]) for i in range(len(hull_vertices) - 1)]
    hull_edges.append((hull_vertices[-1], hull_vertices[0]))  # 마지막 선분을 첫 번째 점과 연결

    # 점과 선분 사이의 거리 계산 함수
    def point_line_distance(point, line_start, line_end):
        # 점에서 선분까지의 거리 계산
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec), line_start
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection), projection

    # Convex Hull의 각 선분과 점 사이의 거리 계산
    min_distance = float('inf')
    closest_point = None
    for edge in hull_edges:
        start_point = touristAttractionPocs[edge[0]]
        end_point = touristAttractionPocs[edge[1]]
        distance_to_edge, proj_point = point_line_distance(destination_point, start_point, end_point)
        if distance_to_edge < min_distance:
            min_distance = distance_to_edge
            closest_point = proj_point

    distance_threshold = 0.07  # 이 거리까지는 경로 벗어나도 태워다 준다..
    transportation_distance_threshold = 0.01

    # 선분 위의 점 계산 (Convex Hull로부터 일정 거리 떨어진 점)
    direction_vector = destination_point - closest_point
    direction_length = np.linalg.norm(direction_vector)
    if direction_length > distance_threshold:
        unit_vector = direction_vector / direction_length
        target_point = closest_point + unit_vector * distance_threshold
    else:
        target_point = destination_point
    print(direction_length)
    
    

    nearest_transportation_point = None
    if np.array_equal(target_point, destination_point):
        print(1)
    else:
        # transportation 지점 중에서 target_point에 가장 가까운 지점 찾기
        nearest_transportation_distance = float('inf')
        for trans_point in transportationPocs:
            distance_to_transportation = np.linalg.norm(trans_point - target_point)
            if distance_to_transportation < transportation_distance_threshold and distance_to_transportation < nearest_transportation_distance:
                nearest_transportation_distance = distance_to_transportation
                nearest_transportation_point = trans_point

        # 만약 일정 거리 내에 교통 수단 지점이 있다면 해당 지점에서 하차
        if nearest_transportation_point is not None:
            print('일정 거리 내에 교통 수단 지점이 있다')
            target_point = nearest_transportation_point
        else:
            print('일정 거리 내에 교통 수단 지점이 없다')
            
    
    route = []
    route.append({'type':'drt','destination': target_point.tolist()})
    if np.array_equal(target_point, destination_point):
        pass
    else:
        if nearest_transportation_point is not None:
            route.append({'type':'scooter','destination': destination_point.tolist()})
        else:
            route.append({'type':'taxi','destination': destination_point.tolist()})
        

    print("Convex Hull과 점 사이의 최소 거리:", min_distance)
    print("가장 가까운 점:", closest_point)
    print("설정된 거리만큼 떨어진 점 또는 가장 가까운 교통 수단 지점:", target_point)

    return {"closest_bus_index": closest_bus_index, "closest_bus_position": closest_bus_position.tolist(), "route": route}


@app.get("/bus")
def reservation():
    global bus_positions, bus_directions
    
    return {"bus": [{'index': i, 'direction': bus_directions[i],'position': bus_positions.tolist()[i]} for i in range(len(bus_positions))]}