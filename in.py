import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation

def readCSV(f):
    with open(f, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = [row for row in csv_reader]
    return rows

userStartLatitude, userStartLongitude = 36.20466303166719, 129.32213620361614
userDestination = [36.16499153927738, 129.2194916215949]
start_point = np.array([userStartLatitude, userStartLongitude], dtype=np.float64)
end_point = np.array(userDestination, dtype=np.float64)

# CSV 파일 읽기
DRTBus = readCSV('./data/drt.csv')
touristAttraction = readCSV('./data/tourist_attraction.csv')
transportation = readCSV('./data/transportation.csv')

# 관광지 좌표 추출
touristAttractionPocs = np.array(list(map(lambda t: [float(t[1]), float(t[2])], touristAttraction)), dtype=np.float64)

# Convex Hull 계산
hull = ConvexHull(touristAttractionPocs)

# DRT 버스 설정 (시계방향 2대, 반시계방향 2대)
bus_positions = touristAttractionPocs[hull.vertices[:4]]  # 초기 위치 설정
bus_directions = [1, 1, -1, -1]  # 1은 시계방향, -1은 반시계방향
bus_vertex_indices = np.array([0, 1, 2, 3])  # 초기 버스 위치 인덱스

# 애니메이션 업데이트 함수
def update(frame):
    global bus_positions, bus_vertex_indices
    
    plt.clf()  # 기존 그림을 지우고 새로 그립니다.

    # Convex Hull 시각화
    for simplex in hull.simplices:
        plt.plot(touristAttractionPocs[simplex, 1], touristAttractionPocs[simplex, 0], 'b-')

    # 관광지와 사용자 위치 시각화
    plt.scatter(touristAttractionPocs[:, 1], touristAttractionPocs[:, 0], color='red', label='Tourist Attractions')
    plt.scatter(start_point[1], start_point[0], color='green', label='Start Location')
    plt.scatter(end_point[1], end_point[0], color='orange', label='Destination')

    min_distance = float('inf')
    closest_bus_index = None
    
    for i in range(len(bus_positions)):
        current_vertex_index = bus_vertex_indices[i]
        next_vertex_index = (current_vertex_index + bus_directions[i]) % len(hull.vertices)
        
        current_vertex_position = touristAttractionPocs[hull.vertices[current_vertex_index]]
        next_vertex_position = touristAttractionPocs[hull.vertices[next_vertex_index]]

        # 버스를 이동시킴
        bus_positions[i] += (next_vertex_position - current_vertex_position) * 0.02

        # 만약 버스가 다음 점에 도달했다면, 다음 선분으로 이동
        if np.linalg.norm(bus_positions[i] - next_vertex_position) < 0.01:
            bus_vertex_indices[i] = next_vertex_index
            bus_positions[i] = next_vertex_position  # 버스 위치를 정확히 다음 점으로 설정

        # 사용자 출발지와 버스 간의 거리 계산
        distance_to_start = np.linalg.norm(start_point - bus_positions[i])

        # 도착지 방향 확인 및 가장 가까운 버스 찾기
        if is_facing_destination(bus_positions[i], bus_directions[i], end_point) and distance_to_start < min_distance:
            min_distance = distance_to_start
            closest_bus_index = i

    # 버스 위치 시각화
    plt.scatter(bus_positions[:, 1], bus_positions[:, 0], color='blue', label='DRT Buses')

    if closest_bus_index is not None:
        closest_bus_position = bus_positions[closest_bus_index]
        plt.scatter(closest_bus_position[1], closest_bus_position[0], color='purple', label='Closest Bus')

    # 그래프 설정
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Closest DRT Bus Considering Start and Destination')
    plt.grid(True)
    plt.legend()

# 방향 체크 함수
def is_facing_destination(bus_position, bus_direction, destination_position):
    # 버스 방향과 도착지 위치를 비교하여 도착지를 향하는지 판단
    bus_to_destination_vector = destination_position - bus_position
    angle = np.arctan2(bus_to_destination_vector[1], bus_to_destination_vector[0])
    
    if bus_direction == 1:  # 시계방향
        return -np.pi/2 <= angle <= np.pi/2
    else:  # 반시계방향
        return angle < -np.pi/2 or angle > np.pi/2

# 애니메이션 생성
fig = plt.figure(figsize=(10, 8))
ani = FuncAnimation(fig, update, frames=1000, interval=50, repeat=True)
plt.show()
