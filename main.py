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


userLatitude,userLongitude = 36.03466303166719,129.42213620361614
userDestination = [36.16499153927738, 129.3194916215949]
point = np.array(userDestination,  dtype=np.float64)

DRTBus = readCSV('./data/drt.csv')
touristAttraction = readCSV('./data/tourist_attraction.csv')
transportation = readCSV('./data/transportation.csv')

touristAttractionPocs = np.array(list(map(lambda t: [t[1], t[2]], touristAttraction)), dtype=np.float64)
print(touristAttractionPocs)

hull = ConvexHull(touristAttractionPocs)


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
        return np.linalg.norm(point_vec)
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

# Convex Hull의 각 선분과 점 사이의 거리 계산
min_distance = float('inf')
for edge in hull_edges:
    start_point = touristAttractionPocs[edge[0]]
    end_point = touristAttractionPocs[edge[1]]
    distance_to_edge = point_line_distance(point, start_point, end_point)
    if distance_to_edge < min_distance:
        min_distance = distance_to_edge

print("Convex Hull과 점 사이의 최소 거리:", min_distance)

plt.figure(figsize=(10, 8))

plt.scatter(touristAttractionPocs[:, 1], touristAttractionPocs[:, 0], color='red', label='DRT Bus route')
plt.scatter(point[1], point[0], color='green', label='Destination')

for simplex in hull.simplices:
    plt.plot(touristAttractionPocs[simplex, 1], touristAttractionPocs[simplex, 0], 'b-')

# 그래프 설정
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('The distance between the DRT bus route and the user\'s destination (Convex Hull)')
plt.grid(True)
plt.legend()
plt.show()



# 가장 가까운 DRT 버스 찾고, 만약 주변에 타는 사람이 별로 없으면 킥보드나 택시타고 DRT 버스 기다리는 사람이 많은 곳으로 이동.
# DRT 버스가 사용자 위치로 접근 후 탑승시킴
# 만약 사용자 위치가 관광지라면) DRT 버스가 사용자 목적지로 이동 후 하차시킴
# 만약 사용자 위치가 관광지가 아니라면) 근처에 내려주고 킥보드와 택시로 이동시킴

