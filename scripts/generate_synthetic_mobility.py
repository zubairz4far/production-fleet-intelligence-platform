from __future__ import annotations

import argparse
from pathlib import Path

from fleet_intelligence.synthetic_mobility import generate_eta_trips, generate_routing_stops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips-output", default="data/synthetic_eta_trips.csv")
    parser.add_argument("--stops-output", default="data/synthetic_routing_stops.csv")
    parser.add_argument("--rows", type=int, default=1600)
    parser.add_argument("--stops", type=int, default=18)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    trips_output = Path(args.trips_output)
    stops_output = Path(args.stops_output)
    trips_output.parent.mkdir(parents=True, exist_ok=True)
    stops_output.parent.mkdir(parents=True, exist_ok=True)

    trips = generate_eta_trips(rows=args.rows, seed=args.seed)
    stops = generate_routing_stops(stop_count=args.stops, seed=args.seed)
    trips.to_csv(trips_output, index=False)
    stops.to_csv(stops_output, index=False)
    print(f"wrote {len(trips)} ETA trips to {trips_output}")
    print(f"wrote {len(stops) - 1} delivery stops + depot to {stops_output}")


if __name__ == "__main__":
    main()
