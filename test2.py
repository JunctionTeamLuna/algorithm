import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def readCSV(f):
    with open(f, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = [row for row in csv_reader]
    return rows

# 사용자 위치와 목적지 설정
userLatitude, userLongitude = 36.03466303166719, 129.42213620361614
userDestination = [36.16499153927738, 129.3194916215949]
point = np.array(userDestination, dtype=np.float64)

# CSV 파일 읽기
DRTBus = readCSV('./data/drt.csv')
touristAttraction = readCSV('./data/tourist_attraction.csv')
transportation = readCSV('./data/transportation.csv')

# 관광지 좌표 추출
touristAttractionPocs = np.array(list(map(lambda t: [float(t[1]), float(t[2])], touristAttraction)), dtype=np.float64)
print(touristAttractionPocs)

# 교통 수단 지점 좌표 추출
transportationPocs = np.array(list(map(lambda t: [float(t[1]), float(t[2])], transportation)), dtype=np.float64)

# Convex Hull 계산
hull = ConvexHull(touristAttractionPocs)

# Convex Hull 선분 추출
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
    distance_to_edge, proj_point = point_line_distance(point, start_point, end_point)
    if distance_to_edge < min_distance:
        min_distance = distance_to_edge
        closest_point = proj_point

# 설정한 거리 단위
distance_threshold = 0.01  # 예를 들어 0.05 단위 거리
transportation_distance_threshold = 0.02

# 선분 위의 점 계산 (Convex Hull로부터 일정 거리 떨어진 점)
direction_vector = point - closest_point
direction_length = np.linalg.norm(direction_vector)
if direction_length > 0:
    unit_vector = direction_vector / direction_length
    target_point = closest_point + unit_vector * distance_threshold
else:
    target_point = closest_point

# transportation 지점 중에서 target_point에 가장 가까운 지점 찾기
nearest_transportation_point = None
nearest_transportation_distance = float('inf')
for trans_point in transportationPocs:
    distance_to_transportation = np.linalg.norm(trans_point - target_point)
    if distance_to_transportation < transportation_distance_threshold and distance_to_transportation < nearest_transportation_distance:
        nearest_transportation_distance = distance_to_transportation
        nearest_transportation_point = trans_point

# 만약 일정 거리 내에 교통 수단 지점이 있다면 해당 지점에서 하차
if nearest_transportation_point is not None:
    target_point = nearest_transportation_point

print("Convex Hull과 점 사이의 최소 거리:", min_distance)
print("가장 가까운 점:", closest_point)
print("설정된 거리만큼 떨어진 점 또는 가장 가까운 교통 수단 지점:", target_point)

# 시각화
plt.figure(figsize=(10, 8))

plt.scatter(touristAttractionPocs[:, 1], touristAttractionPocs[:, 0], color='red', label='DRT Bus route')
plt.scatter(point[1], point[0], color='green', label='Destination')
plt.scatter(closest_point[1], closest_point[0], color='orange', label='Closest Point')
plt.scatter(transportationPocs[:, 1], transportationPocs[:, 0], color='blue', label='Transportation Points')
plt.scatter(target_point[1], target_point[0], color='purple', label='Target/Transport Point')

# Convex Hull을 잇는 선 그리기
for simplex in hull.simplices:
    plt.plot(touristAttractionPocs[simplex, 1], touristAttractionPocs[simplex, 0], 'b-')

# point와 closest_point를 잇는 선 그리기
plt.plot([point[1], closest_point[1]], [point[0], closest_point[0]], 'k--', label='Min Distance Line')

# 그래프 설정
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('The distance between the DRT bus route and the user\'s destination (Convex Hull)')
plt.grid(True)
plt.legend()
plt.show()
