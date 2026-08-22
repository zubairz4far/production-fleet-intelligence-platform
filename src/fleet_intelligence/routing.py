from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .geospatial import haversine_km

STOP_REQUIRED_COLUMNS = (
    "stop_id",
    "latitude",
    "longitude",
    "demand_units",
    "is_depot",
)


@dataclass(frozen=True)
class RouteMetrics:
    total_distance_km: float
    max_route_distance_km: float
    served_stops: int
    unserved_stops: int
    capacity_violation_units: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def validate_stop_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in STOP_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing routing columns: {missing}")

    data = frame.copy()
    data["stop_id"] = data["stop_id"].astype(str)
    for column in ("latitude", "longitude", "demand_units", "is_depot"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not data["latitude"].between(-90.0, 90.0).all():
        raise ValueError("latitude must be between -90 and 90")
    if not data["longitude"].between(-180.0, 180.0).all():
        raise ValueError("longitude must be between -180 and 180")
    if (data["demand_units"] < 0).any():
        raise ValueError("demand_units must be non-negative")
    if not set(data["is_depot"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("is_depot must contain only 0/1")
    if int(data["is_depot"].sum()) != 1:
        raise ValueError("Routing fixture must contain exactly one depot")
    if data["stop_id"].duplicated().any():
        raise ValueError("stop_id values must be unique")

    data["demand_units"] = data["demand_units"].astype(int)
    data["is_depot"] = data["is_depot"].astype(int)
    return data.reset_index(drop=True)


def distance_matrix_km(stops: pd.DataFrame) -> np.ndarray:
    data = validate_stop_dataset(stops)
    lat = data["latitude"].to_numpy(dtype=float)
    lon = data["longitude"].to_numpy(dtype=float)
    size = len(data)
    matrix = np.zeros((size, size), dtype=float)
    for row in range(size):
        matrix[row] = haversine_km(lat[row], lon[row], lat, lon)
    return matrix


def _route_distance(route: list[int], matrix: np.ndarray) -> float:
    return float(sum(matrix[a, b] for a, b in zip(route[:-1], route[1:], strict=True)))


def evaluate_routes(
    routes: list[list[int]],
    stops: pd.DataFrame,
    *,
    vehicle_capacity: int,
) -> RouteMetrics:
    data = validate_stop_dataset(stops)
    matrix = distance_matrix_km(data)
    depot = int(data.index[data["is_depot"] == 1][0])
    expected = set(data.index[data["is_depot"] == 0])
    served: set[int] = set()
    route_distances: list[float] = []
    violation = 0

    for route in routes:
        if len(route) < 2 or route[0] != depot or route[-1] != depot:
            raise ValueError("Every route must start and end at the depot")
        inner = [node for node in route[1:-1] if node != depot]
        served.update(inner)
        load = int(data.loc[inner, "demand_units"].sum()) if inner else 0
        violation += max(0, load - vehicle_capacity)
        route_distances.append(_route_distance(route, matrix))

    return RouteMetrics(
        total_distance_km=float(sum(route_distances)),
        max_route_distance_km=float(max(route_distances, default=0.0)),
        served_stops=len(served & expected),
        unserved_stops=len(expected - served),
        capacity_violation_units=int(violation),
    )


def greedy_capacity_routes(
    stops: pd.DataFrame,
    *,
    vehicle_count: int,
    vehicle_capacity: int,
) -> list[list[int]]:
    data = validate_stop_dataset(stops)
    if vehicle_count < 1 or vehicle_capacity < 1:
        raise ValueError("vehicle_count and vehicle_capacity must be positive")
    depot = int(data.index[data["is_depot"] == 1][0])
    matrix = distance_matrix_km(data)
    unserved = set(data.index[data["is_depot"] == 0])
    routes: list[list[int]] = []

    for _vehicle in range(vehicle_count):
        current = depot
        remaining_capacity = vehicle_capacity
        route = [depot]
        while unserved:
            feasible = [
                node
                for node in unserved
                if int(data.loc[node, "demand_units"]) <= remaining_capacity
            ]
            if not feasible:
                break
            next_node = min(feasible, key=lambda node: (matrix[current, node], node))
            route.append(next_node)
            remaining_capacity -= int(data.loc[next_node, "demand_units"])
            unserved.remove(next_node)
            current = next_node
        route.append(depot)
        routes.append(route)

    return routes


def ortools_capacity_routes(
    stops: pd.DataFrame,
    *,
    vehicle_count: int,
    vehicle_capacity: int,
    search_seconds: int = 2,
) -> list[list[int]]:
    data = validate_stop_dataset(stops)
    if vehicle_count < 1 or vehicle_capacity < 1:
        raise ValueError("vehicle_count and vehicle_capacity must be positive")
    depot = int(data.index[data["is_depot"] == 1][0])
    matrix_m = np.rint(distance_matrix_km(data) * 1000.0).astype(int)
    demands = data["demand_units"].to_numpy(dtype=int)

    manager = pywrapcp.RoutingIndexManager(len(data), vehicle_count, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix_m[from_node, to_node])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(from_index: int) -> int:
        return int(demands[manager.IndexToNode(from_index)])

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [vehicle_capacity] * vehicle_count,
        True,
        "Capacity",
    )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromSeconds(search_seconds)
    parameters.log_search = False

    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("OR-Tools could not find a feasible routing solution")

    routes: list[list[int]] = []
    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def benchmark_routing(
    stops: pd.DataFrame,
    *,
    vehicle_count: int = 3,
    vehicle_capacity: int = 6,
) -> dict[str, object]:
    data = validate_stop_dataset(stops)
    non_depot = data[data["is_depot"] == 0]
    total_demand = int(non_depot["demand_units"].sum())
    if total_demand > vehicle_count * vehicle_capacity:
        raise ValueError("Total demand exceeds available vehicle capacity")
    if int(non_depot["demand_units"].max()) > vehicle_capacity:
        raise ValueError("At least one stop demand exceeds vehicle capacity")

    baseline_routes = greedy_capacity_routes(
        data,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
    )
    candidate_routes = ortools_capacity_routes(
        data,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
    )
    baseline_metrics = evaluate_routes(
        baseline_routes,
        data,
        vehicle_capacity=vehicle_capacity,
    )
    candidate_metrics = evaluate_routes(
        candidate_routes,
        data,
        vehicle_capacity=vehicle_capacity,
    )

    criteria = {
        "all_stops_served": candidate_metrics.unserved_stops == 0,
        "capacity_satisfied": candidate_metrics.capacity_violation_units == 0,
        "distance_non_regression": (
            candidate_metrics.total_distance_km <= baseline_metrics.total_distance_km + 1e-6
        ),
    }
    promoted = all(criteria.values())
    improvement = 0.0
    if baseline_metrics.total_distance_km > 0:
        improvement = (
            baseline_metrics.total_distance_km - candidate_metrics.total_distance_km
        ) / baseline_metrics.total_distance_km * 100.0

    stop_ids = data["stop_id"].to_dict()

    def readable(routes: list[list[int]]) -> list[list[str]]:
        return [[str(stop_ids[node]) for node in route] for route in routes]

    return {
        "task": "capacitated_vehicle_routing",
        "problem": {
            "stops_excluding_depot": len(non_depot),
            "vehicle_count": vehicle_count,
            "vehicle_capacity": vehicle_capacity,
            "total_demand_units": total_demand,
            "constraints": ["single_depot", "vehicle_capacity", "serve_each_stop_once"],
        },
        "baseline": {
            "solver": "deterministic_nearest_feasible_greedy",
            "routes": readable(baseline_routes),
            "metrics": baseline_metrics.to_dict(),
        },
        "candidate": {
            "solver": "ortools_guided_local_search",
            "routes": readable(candidate_routes),
            "metrics": candidate_metrics.to_dict(),
        },
        "promotion": {
            "decision": "promote" if promoted else "reject",
            "criteria": criteria,
            "distance_improvement_percent": float(improvement),
        },
        "limitations": [
            "Synthetic coordinates are CI regression evidence, not a live road network.",
            "Distance uses great-circle geometry rather than road travel distance.",
            "v0.4 routing constrains capacity; time windows are reserved for a later release.",
        ],
    }
