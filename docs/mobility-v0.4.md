# v0.4 ETA + Geospatial Routing Protocol

v0.4 adds two linked fleet capabilities: ETA prediction and constrained route optimization. The synthetic fixture is for CI/software regression only and cannot establish production fleet performance.

## ETA prediction contract

Trip observations are sorted by `timestamp` and split chronologically into train, development, and untouched test windows.

The target is `actual_travel_minutes`. It is excluded from model features.

Available-at-dispatch features include:

- planned route distance;
- Haversine great-circle distance;
- planned-distance / great-circle detour ratio;
- traffic index;
- weather severity;
- vehicle load;
- remaining stops;
- cyclic hour/day-of-week encoding;
- route bearing encoded as sine/cosine.

The baseline estimates a single median effective speed using training data only. The candidate is a fixed `HistGradientBoostingRegressor`. Candidate hyperparameters and promotion criteria are frozen before the final test window is observed.

### ETA promotion rule

The candidate is promoted only if all three conditions hold on the untouched chronological test window:

1. MAE is lower than the train-fitted median-speed baseline;
2. RMSE is lower than the baseline;
3. p90 absolute error is lower than the baseline.

Development metrics are recorded but the v0.4 synthetic candidate is not tuned after test observation.

## Routing contract

The routing fixture contains one depot and a set of delivery stops. Each stop has a non-negative integer demand.

The benchmark uses a fixed fleet count and identical vehicle capacity.

Constraints:

- one depot;
- every delivery stop must be served exactly once;
- every route starts and ends at the depot;
- route demand must not exceed vehicle capacity.

The deterministic baseline repeatedly selects the nearest currently feasible stop. The candidate uses Google OR-Tools `RoutingModel` with a capacity dimension, `PATH_CHEAPEST_ARC` initialization, and guided local search.

Distance is computed using Haversine geometry. This is intentionally not presented as road-network routing.

### Routing promotion rule

The OR-Tools candidate is promoted only if:

1. every stop is served;
2. no vehicle capacity is violated;
3. total route distance is less than or equal to the deterministic greedy baseline.

## Release gate

v0.4 is promoted only when both the ETA gate and routing gate pass.

## Non-claims

The synthetic fixture does not prove:

- real ETA accuracy;
- real traffic or weather robustness;
- road-network optimality;
- production route savings;
- time-window feasibility.

Time windows, road-network travel matrices, live traffic, and operational service-time constraints require later real-data/integration work.
