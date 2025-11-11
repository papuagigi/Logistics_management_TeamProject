# -*- coding: utf-8 -*-
"""
Jupyter Notebook을 Python 스크립트로 변환한 파일입니다.
"""

# --- 셀 1: 데이터 로드 및 이상적 거점 탐색 ---
import pandas as pd

# 1. 최종 배송 목록 파일 경로 설정
file_path = '최종_배송목록_20231201_강남구.csv'

# 2. CSV 파일 불러오기 (인코딩 자동 탐색)
def read_csv_safe(p):
    for enc in ["utf-8-sig", "euc-kr", "cp949"]:
        try:
            df = pd.read_csv(p, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(p)  # 마지막 시도
    df.columns = df.columns.str.strip()
    return df

df_deliveries = read_csv_safe(file_path)

# 3. 위도(latitude)와 경도(longitude) 칼럼 확인 및 숫자형 변환
if '위도' not in df_deliveries.columns or '경도' not in df_deliveries.columns:
    raise KeyError("필수 칼럼(위도, 경도)이 최종 배송 목록에 없습니다.")

df_deliveries['위도'] = pd.to_numeric(df_deliveries['위도'], errors='coerce')
df_deliveries['경도'] = pd.to_numeric(df_deliveries['경도'], errors='coerce')

# 4. 유효한 좌표만 사용하여 무게 중심 계산
df_valid_coords = df_deliveries.dropna(subset=['위도', '경도'])

center_lat = df_valid_coords['위도'].mean()
center_lon = df_valid_coords['경도'].mean()

print(f"강남구 배송 수요의 이상적인 무게 중심 (Center of Gravity):")
print(f"- 위도: {center_lat:.6f}, 경도: {center_lon:.6f}")


# --- 셀 2: 현실적 거점 후보군 도출 ---
# 스크립트 실행 전 터미널에서 라이브러리를 설치해주세요: pip install osmnx
import osmnx as ox
import numpy as np
from scipy.spatial import cKDTree

# 1. 강남구 도로망 데이터 다운로드 및 저장
place_name = 'Gangnam-gu, Seoul, South Korea'
graph_path = 'gangnam_drive_network.graphml'
try:
    G = ox.load_graphml(graph_path)
    print(f"'{graph_path}' 파일에서 도로망 데이터를 불러왔습니다.")
except FileNotFoundError:
    print(f"'{graph_path}' 파일이 없어 새로 다운로드합니다...")
    G = ox.graph_from_place(place_name, network_type='drive')
    ox.save_graphml(G, graph_path)
    print(f"도로망 데이터를 '{graph_path}'에 저장했습니다.")

# 2. 이상적 거점 위치(무게 중심)에서 가장 가까운 교차로(노드) 3개 찾기
num_candidates = 3
# cKDTree를 사용하여 k개의 가장 가까운 노드 찾기
nodes_coords = np.array([[G.nodes[node]['x'], G.nodes[node]['y']] for node in G.nodes()])
tree = cKDTree(nodes_coords)
distances, indices = tree.query([center_lon, center_lat], k=num_candidates)
depot_candidates = [list(G.nodes())[i] for i in indices]

print(f"\n이상적 위치에서 가장 가까운 현실적 거점 후보 노드 {num_candidates}개:")
print(depot_candidates)

# 3. (선택) 후보 노드들의 좌표 확인
candidate_nodes_data = G.nodes(data=True)
for node_id in depot_candidates:
    node_info = candidate_nodes_data[node_id]
    print(f"- 노드 ID {node_id}: (위도: {node_info['y']:.6f}, 경도: {node_info['x']:.6f})")


# --- 셀 3: 다차원 시뮬레이션 및 최적해 결정 ---
# 스크립트 실행 전 터미널에서 라이브러리를 설치해주세요: pip install ortools
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

# --- A. 시뮬레이션 파라미터 설정 ---
vehicles = {
    'diesel_1t': {
        'name': '1톤 디젤 트럭',
        'fuel_efficiency': 8.5,  # km/L
        'co2_per_liter': 2.64,   # kg/L
        'cost_per_km': 180,      # 원/km (유류비)
        'capacity': 50,          # 최대 적재 용량 (박스)
    }
}

# 탄소세율은 추후 구체적인 값으로 설정 (예: kg CO2당 50원)
current_carbon_tax_per_kg_co2 = 50 

driver_wage_per_hour = 12000 # 시간당 인건비
avg_speed_kph = 20 # 평균 주행 속도
driver_wage_per_km = driver_wage_per_hour / avg_speed_kph # km당 인건비



# --- B. 분석용 배송지 샘플링 및 수요량 설정 ---
num_samples = 100
df_sample = df_deliveries.sample(n=num_samples, random_state=42)

# 배송지_유형별 평균 수요량 설정 (단위: 박스 또는 kg)
demand_by_type = {
    '주거': 3,   # 주거지역: 평균 3박스
    '상업': 8,   # 상업지역: 평균 8박스
    '업무': 12,  # 업무지역: 평균 12박스
    '기타': 5    # 기타: 평균 5박스
}

# 각 배송지의 수요량 계산 (배송지_유형이 없으면 기본값 5 사용)
if '배송지_유형' in df_sample.columns:
    df_sample['수요량'] = df_sample['배송지_유형'].map(demand_by_type).fillna(5)
else:
    df_sample['수요량'] = 5  # 배송지_유형 정보가 없으면 모두 5로 설정

delivery_nodes = ox.distance.nearest_nodes(G, X=df_sample['경도'], Y=df_sample['위도'])
demands = df_sample['수요량'].tolist()  # 각 배송지의 수요량 리스트

print(f"\n총 {len(df_deliveries)}개 배송지 중 {num_samples}개를 샘플링하여 분석합니다.")
print(f"총 수요량: {sum(demands)}박스, 평균 수요량: {sum(demands)/len(demands):.1f}박스/지점")



# --- C. VRP 해결 함수 및 비용 매트릭스 생성 함수 정의 ---

memo_distances = {} # 거리 계산 결과를 저장할 딕셔너리 (메모이제이션)



def get_distance_km(from_node, to_node):

    if (from_node, to_node) in memo_distances:

        return memo_distances[(from_node, to_node)]

    try:

        distance_m = nx.shortest_path_length(G, source=from_node, target=to_node, weight='length')

        distance_km = distance_m / 1000

        memo_distances[(from_node, to_node)] = distance_km

        return distance_km

    except nx.NetworkXNoPath:

        return np.inf # 경로가 없으면 무한대 비용



def create_cost_matrix(vrp_nodes, vehicle_type, tax_rate_per_kg):

    num_nodes = len(vrp_nodes)

    cost_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    vehicle = vehicles[vehicle_type]

    if 'co2_per_liter' in vehicle:

        co2_per_km = vehicle['co2_per_liter'] / vehicle['fuel_efficiency']

    else:

        kwh_per_km = 1 / vehicle['energy_efficiency']

        co2_per_km = vehicle['co2_per_kwh'] * kwh_per_km

    for i in range(num_nodes):

        for j in range(num_nodes):

            if i == j: continue

            distance_km = get_distance_km(vrp_nodes[i], vrp_nodes[j])

            if distance_km == np.inf: 

                cost_matrix[i][j] = 99999999 # 매우 큰 비용

                continue

            operational_cost = (vehicle['cost_per_km'] + driver_wage_per_km) * distance_km

            carbon_cost = (co2_per_km * tax_rate_per_kg) * distance_km

            total_cost = operational_cost + carbon_cost

            cost_matrix[i][j] = int(total_cost)

    return cost_matrix



def solve_vrp(cost_matrix, num_vehicles, demands, vehicle_capacity):
    """
    용량 제약이 있는 VRP 문제를 해결합니다.
    
    Args:
        cost_matrix: 노드 간 이동 비용 매트릭스
        num_vehicles: 사용할 차량 대수
        demands: 각 노드(배송지)의 수요량 리스트 (인덱스 0은 거점이므로 0)
        vehicle_capacity: 각 차량의 최대 적재 용량
    """
    manager = pywrapcp.RoutingIndexManager(len(cost_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # 1. 비용(거리) 콜백 함수 등록
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return cost_matrix[from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 2. 수요량(용량) 콜백 함수 등록
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # 3. 용량 제약 추가
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack (용량 여유 없음)
        [vehicle_capacity] * num_vehicles,  # 각 차량의 용량
        True,  # start cumul to zero (시작 시 적재량 0)
        'Capacity'
    )
    
    # 4. 탐색 파라미터 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30  # 최대 30초 탐색
    
    # 5. 문제 해결
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        return solution.ObjectiveValue()
    return None



# --- D. 메인 시뮬레이션 루프 실행 ---
results = []
num_vehicles_range = range(5, 16) # 테스트할 차량 대수 범위 (5대 ~ 15대)
print("\n--- 다차원 시뮬레이션을 시작합니다 ---")

for depot_node in depot_candidates:
    vrp_nodes = [depot_node] + list(delivery_nodes)
    # 거점의 수요량은 0 (출발지/도착지이므로)
    vrp_demands = [0] + demands
    
    for vehicle_name, vehicle_data in vehicles.items():
        vehicle_capacity = vehicle_data['capacity']
        # 탄소세 시나리오 대신 단일 탄소세율 적용
        cost_matrix = create_cost_matrix(vrp_nodes, vehicle_name, current_carbon_tax_per_kg_co2)
        
        for num_vehicles in num_vehicles_range:
            print(f"계산 중: 거점={depot_node}, 차량={vehicle_data['name']}, "
                  f"용량={vehicle_capacity}박스, 탄소세={current_carbon_tax_per_kg_co2}원/kgCO2, 차량 수={num_vehicles}")
            
            total_cost = solve_vrp(cost_matrix, num_vehicles, vrp_demands, vehicle_capacity)
            
            if total_cost is not None:
                results.append({
                    '거점_노드_ID': depot_node,
                    '차량_종류': vehicle_data['name'],
                    '차량_용량': vehicle_capacity,
                    '탄소세_시나리오': f'{current_carbon_tax_per_kg_co2}원/kgCO2',
                    '차량_대수': num_vehicles,
                    '총_물류비용(원)': total_cost
                })



# --- E. 최종 결과 분석 ---

df_results = pd.DataFrame(results)

print("\n--- 시뮬레이션 완료. 최적 솔루션 분석 ---")



if not df_results.empty:

    # 이제 탄소세 시나리오가 하나이므로, 거점별 최적해를 찾습니다.

    optimal_solutions = df_results.loc[df_results.groupby(['거점_노드_ID'])['총_물류비용(원)'].idxmin()]

    print("\n[각 거점별 최적 솔루션]")

    print(optimal_solutions.sort_values('총_물류비용(원)'))



    # 전체 최적해 찾기

    overall_best = df_results.loc[df_results['총_물류비용(원)'].idxmin()]

    print("\n[전체 시나리오 통합 최적 솔루션]")

    print(overall_best)

else:

    print("\n모든 시나리오에서 해를 찾지 못했습니다.")
