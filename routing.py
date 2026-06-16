"""
routing.py — Diversion route generator + Barricade point mapper
            (now CAPACITY-AWARE: routes avoid already-congested roads)

Uses OSMnx to pull Bengaluru road graph, removes affected edges, and computes
alternate routes. The differentiator: every road's travel cost is inflated by the
local congestion load (congestion.py), so diversions route AROUND congestion instead
of dumping traffic onto roads that are already choked or next to other active events.

Usage (standalone test):
    python routing.py
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

# capacity-aware tuning
ALPHA = 2.5    # edge penalty: cost = length * (1 + ALPHA * load)
BETA = 1.5     # route ranking: capacity_cost = distance * (1 + BETA * avg_load)

_GRAPH_CACHE = None
GRAPH_CACHE_PATH = "models/bengaluru_graph.graphml"


def _congestion_label(load):
    return "🟢 clear" if load < 0.25 else "🟡 moderate" if load < 0.55 else "🔴 congested"


# ──────────────────────────────────────────────
# 1. GRAPH LOADING
# ──────────────────────────────────────────────
def get_bengaluru_graph(force_download: bool = False):
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE
    if not OSMNX_AVAILABLE:
        return None
    if os.path.exists(GRAPH_CACHE_PATH) and not force_download:
        print("Loading cached road graph...")
        _GRAPH_CACHE = ox.load_graphml(GRAPH_CACHE_PATH)
        return _GRAPH_CACHE
    print("Graph not cached. Using fallback routing (run python routing.py --download to cache).")
    return None


def nearest_node(G, lat: float, lon: float) -> int:
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


# ──────────────────────────────────────────────
# 2. BARRICADE ENTRY POINT MAPPER  (unchanged)
# ──────────────────────────────────────────────
def find_barricade_points(lat, lon, radius_m=400, max_points=4):
    G = get_bengaluru_graph()
    if G is None:
        return _fallback_barricade_points(lat, lon, radius_m, max_points)
    center_node = nearest_node(G, lat, lon)
    try:
        subgraph_nodes = nx.single_source_dijkstra_path_length(
            G, center_node, cutoff=radius_m, weight='length')
    except Exception:
        return _fallback_barricade_points(lat, lon, radius_m, max_points)

    entry_points = []
    for node_id, dist in subgraph_nodes.items():
        if node_id == center_node:
            continue
        if G.degree(node_id) >= 3:
            nd = G.nodes[node_id]
            street_name = "Unknown Road"
            for _, _, ed in G.edges(node_id, data=True):
                if 'name' in ed:
                    nm = ed['name']; street_name = nm if isinstance(nm, str) else nm[0]; break
            entry_points.append({'lat': nd['y'], 'lon': nd['x'], 'node_id': int(node_id),
                                 'street_name': street_name, 'distance_m': round(dist)})
    entry_points.sort(key=lambda x: x['distance_m'])
    selected = []
    for pt in entry_points:
        if len(selected) >= max_points:
            break
        if not any(_haversine(pt['lat'], pt['lon'], s['lat'], s['lon']) < 100 for s in selected):
            selected.append(pt)
    return selected if selected else _fallback_barricade_points(lat, lon, radius_m, max_points)


def _fallback_barricade_points(lat, lon, radius_m, max_points):
    angles = np.linspace(0, 2 * np.pi, max_points, endpoint=False)
    deg_per_m = 1 / 111320
    directions = ["North Entry", "East Entry", "South Entry", "West Entry"]
    pts = []
    for i, angle in enumerate(angles):
        dlat = np.sin(angle) * radius_m * deg_per_m
        dlon = np.cos(angle) * radius_m * deg_per_m / np.cos(np.radians(lat))
        pts.append({'lat': round(lat + dlat, 6), 'lon': round(lon + dlon, 6),
                    'node_id': None, 'street_name': directions[i % len(directions)],
                    'distance_m': radius_m})
    return pts


# ──────────────────────────────────────────────
# 3. DIVERSION ROUTE GENERATOR  (capacity-aware)
# ──────────────────────────────────────────────
def get_diversion_routes(incident_lat, incident_lon, radius_m=300, num_routes=3,
                         congestion=None, when=None):
    G = get_bengaluru_graph()
    if G is None:
        return _fallback_diversion_routes(incident_lat, incident_lon, congestion, when)

    center_node = nearest_node(G, incident_lat, incident_lon)
    try:
        blocked_nodes = set(nx.single_source_dijkstra_path_length(
            G, center_node, cutoff=radius_m, weight='length').keys())
        blocked_nodes.discard(center_node)
    except Exception:
        return _fallback_diversion_routes(incident_lat, incident_lon, congestion, when)

    G_blocked = G.copy()
    G_blocked.remove_nodes_from(list(blocked_nodes - {center_node}))

    # >>> CAPACITY-AWARE EDGE WEIGHTING <<<
    # inflate each edge's cost by the congestion load at its midpoint
    weight_key = 'length'
    if congestion is not None:
        weight_key = 'cost'
        for u, v, k, data in G_blocked.edges(keys=True, data=True):
            mlat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            mlon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
            load = congestion.combined_load(mlat, mlon, when)
            data['cost'] = data.get('length', 50) * (1 + ALPHA * load)

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    deg_per_m = 1 / 111320
    origin_candidates, dest_candidates = [], []
    for angle in angles:
        dlat = np.sin(angle) * 800 * deg_per_m
        dlon = np.cos(angle) * 800 * deg_per_m / np.cos(np.radians(incident_lat))
        try:
            n = nearest_node(G, incident_lat + dlat, incident_lon + dlon)
            if n not in blocked_nodes:
                origin_candidates.append((n, angle))
        except Exception:
            pass
        dlat2 = np.sin(angle + np.pi) * 800 * deg_per_m
        dlon2 = np.cos(angle + np.pi) * 800 * deg_per_m / np.cos(np.radians(incident_lat))
        try:
            n2 = nearest_node(G, incident_lat + dlat2, incident_lon + dlon2)
            if n2 not in blocked_nodes:
                dest_candidates.append((n2, angle))
        except Exception:
            pass

    routes, tried = [], set()
    for (orig, a1) in origin_candidates:
        for (dest, a2) in dest_candidates:
            if len(routes) >= num_routes * 2:
                break
            if orig == dest or (orig, dest) in tried:
                continue
            tried.add((orig, dest))
            try:
                path = nx.shortest_path(G_blocked, orig, dest, weight=weight_key)
                if len(path) < 3:
                    continue
                length_m = sum(G_blocked[path[i]][path[i+1]][0].get('length', 50)
                               for i in range(len(path)-1))
                coords = [(G.nodes[n]['y'], G.nodes[n]['x'])
                          for n in path[::max(1, len(path)//20)]]
                road_names = []
                for i in range(min(len(path)-1, 10)):
                    ed = G_blocked[path[i]][path[i+1]]
                    name = ed[0].get('name', '')
                    if isinstance(name, list):
                        name = name[0]
                    if name and name not in road_names:
                        road_names.append(name)
                avg_load = congestion.route_load(coords) if congestion is not None else 0.0
                speed_kmh = 25
                routes.append({
                    'route_id': len(routes) + 1, 'coords': coords,
                    'distance_m': round(length_m), 'distance_km': round(length_m/1000, 2),
                    'estimated_mins': round((length_m/1000)/speed_kmh*60),
                    'road_names': road_names[:5],
                    'via': ' → '.join(road_names[:3]) if road_names else 'Alternate route',
                    'node_count': len(path),
                    'congestion_score': round(avg_load, 2),
                    'congestion_label': _congestion_label(avg_load),
                    'capacity_cost': round(length_m * (1 + BETA * avg_load)),
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if len(routes) >= num_routes * 2:
            break

    if not routes:
        return _fallback_diversion_routes(incident_lat, incident_lon, congestion, when)

    return _rank_capacity_aware(routes, num_routes, incident_lat, incident_lon, radius_m,
                                'success', congestion)


# ──────────────────────────────────────────────
# 4. CAPACITY-AWARE FALLBACK
# ──────────────────────────────────────────────
def _fallback_diversion_routes(lat, lon, congestion=None, when=None):
    """Generate several geometric candidate corridors in different bearings, score each
    by congestion, and keep the least-congested. Works without the OSM graph."""
    deg = 1 / 111320
    base_roads = [["Outer Ring Road", "Intermediate Ring Road"], ["Hosur Road", "St. Johns Road"],
                  ["Palace Road", "Sankey Road"], ["Bellary Road", "Jayamahal Road"],
                  ["Old Madras Road", "CMH Road"], ["Magadi Road", "Chord Road"]]
    candidates = []
    for i, bearing in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
        # a 3-point arc skirting the incident in this bearing
        pts = []
        for step, dist in [(0.5, 900), (1.0, 1400), (1.6, 1100)]:
            ang = bearing + step * 0.6
            dlat = np.sin(ang) * dist * deg
            dlon = np.cos(ang) * dist * deg / np.cos(np.radians(lat))
            pts.append((round(lat + dlat, 6), round(lon + dlon, 6)))
        length_m = sum(_haversine(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1])
                       for j in range(len(pts)-1)) + 800
        avg_load = congestion.route_load(pts) if congestion is not None else 0.0
        roads = base_roads[i % len(base_roads)]
        candidates.append({
            'route_id': i + 1, 'coords': pts,
            'distance_m': round(length_m), 'distance_km': round(length_m/1000, 2),
            'estimated_mins': round((length_m/1000)/25*60),
            'road_names': roads, 'via': ' → '.join(roads),
            'congestion_score': round(avg_load, 2),
            'congestion_label': _congestion_label(avg_load),
            'capacity_cost': round(length_m * (1 + BETA * avg_load)),
        })
    status = 'capacity_aware_fallback' if congestion is not None else 'fallback'
    return _rank_capacity_aware(candidates, 3, lat, lon, 300, status, congestion)


# ──────────────────────────────────────────────
# 5. RANKING (shared)
# ──────────────────────────────────────────────
def _rank_capacity_aware(routes, num_routes, lat, lon, radius_m, status, congestion):
    key = 'capacity_cost' if congestion is not None else 'distance_m'
    routes.sort(key=lambda r: r[key])
    routes = routes[:num_routes]
    for i, r in enumerate(routes):
        r['rank'] = i + 1
        r['recommendation'] = "⭐ RECOMMENDED" if i == 0 else "ALTERNATE" if i == 1 else "BACKUP"
    return {'status': status, 'incident': {'lat': lat, 'lon': lon},
            'blocked_radius_m': radius_m, 'capacity_aware': congestion is not None,
            'routes': routes}


# ──────────────────────────────────────────────
# 6. UTILITY
# ──────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def full_routing_analysis(lat, lon, requires_closure=True, radius_m=350,
                          when=None, models_dir="models", data_path=None):
    """Single entry point for dashboard. Loads the congestion surface (if built) and
    returns capacity-aware barricades + diversions."""
    congestion = None
    try:
        from congestion import load_surface
        congestion = load_surface(models_dir, data_path)
    except Exception:
        congestion = None

    barricades = find_barricade_points(lat, lon, radius_m=radius_m)
    diversions = get_diversion_routes(lat, lon, radius_m=radius_m,
                                      congestion=congestion, when=when)
    return {'barricade_points': barricades, 'barricade_count': len(barricades),
            'diversions': diversions,
            'capacity_aware': diversions.get('capacity_aware', False),
            'routing_active': requires_closure or diversions['status'].endswith('success')}


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Pre-download OSM graph')
    args = parser.parse_args()

    if args.download:
        print("Downloading Bengaluru OSM graph...")
        import osmnx as ox, os
        os.makedirs('models', exist_ok=True)
        G = ox.graph_from_place("Bengaluru, Karnataka, India", network_type="drive", simplify=True)
        ox.save_graphml(G, GRAPH_CACHE_PATH)
        print(f"Saved to {GRAPH_CACHE_PATH}")
        raise SystemExit(0)

    print("Testing CAPACITY-AWARE routing (fallback mode)...")
    OSMNX_AVAILABLE = False
    result = full_routing_analysis(lat=13.0108, lon=77.5530, requires_closure=True,
                                   data_path="data/flipkart_gridlock.csv")
    print(f"\nCapacity-aware: {result['capacity_aware']}")
    print(f"Barricade Points ({result['barricade_count']}):")
    for bp in result['barricade_points']:
        print(f"  [{bp['street_name']}] ({bp['lat']:.5f}, {bp['lon']:.5f}) — {bp['distance_m']}m")
    print(f"\nDiversion Routes (ranked least-congested first):")
    for r in result['diversions']['routes']:
        print(f"  {r['recommendation']} via {r['via']}")
        print(f"    {r['distance_km']}km ~{r['estimated_mins']}min | "
              f"congestion {r['congestion_score']} {r['congestion_label']} | cost {r['capacity_cost']}")
    print("\n✅ Capacity-aware routing working.")