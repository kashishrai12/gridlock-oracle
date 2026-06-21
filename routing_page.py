"""
routing_page.py — Diversion + Barricade map page
Imported into dashboard.py as a page component.

In-process: calls routing.full_routing_analysis() directly — no API server,
no localhost dependency, nothing to fail during a live demo.

Themed to match the rest of the app: uses the shared card_header() helper and
CSS variables (var(--surface), var(--border), var(--ink)...) instead of the
default st.title / hardcoded colors, so it sits on-theme in both light and dark.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium

import routing   # local module — direct call, no HTTP

# Route accent colors (used on the folium map polylines AND the route cards,
# kept consistent so the map and the cards read as the same three routes).
ROUTE_COLORS = ['#16A34A', '#D97706', '#DC2626']
ROUTE_LABELS = ['RECOMMENDED', 'ALTERNATE', 'BACKUP']


def _fallback_card_header(icon_name, title, sub=None):
    """Used only if dashboard didn't pass its card_header in (keeps page importable standalone)."""
    st.markdown(f"**{title}**" + (f"  \n<span style='color:var(--muted);font-size:.8rem'>{sub}</span>" if sub else ""),
                unsafe_allow_html=True)


def _base_map(lat, lon, zoom=14):
    """Folium map. Uses reliable CartoDB tiles as the base (Mappls raster tiles require
    separate Map-Tiles API access and a version-specific URL, so we don't depend on them).
    If Mappls is configured AND a traffic-tile URL is set, we add it as an optional overlay
    layer the user can toggle — but the base map always renders."""
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='CartoDB dark_matter')
    try:
        import mappls
        mc = mappls.client()
        tt = mc.traffic_tile_url() if mc.is_configured() else None
        if tt:
            folium.TileLayer(tt, attr="Mappls Traffic", name="Mappls Live Traffic",
                             overlay=True, control=True, show=False).add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)
    except Exception:
        pass
    return m


def render_routing_page(theme=None, card_header=None, alert=None, **kwargs):
    # use the shared helpers from dashboard if provided, else degrade gracefully
    ch = card_header or _fallback_card_header

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            ch("map_pin", "Incident details", "Coordinates to generate diversions & barricades")
            lat = st.number_input("Latitude", value=13.0108, format="%.6f", key="r_lat")
            lon = st.number_input("Longitude", value=77.5530, format="%.6f", key="r_lon")
            closure = st.checkbox("Road closure required", value=True, key="r_closure")
            radius = st.slider("Block radius (metres)", 100, 800, 350, 50, key="r_radius")

            st.markdown("**Quick locations**")
            presets = {
                "Mekhri Circle": (13.0108, 77.5530),
                "Silk Board": (12.9172, 77.6210),
                "Hebbal Flyover": (13.0450, 77.5970),
                "Marathahalli": (12.9591, 77.6974),
            }
            for name, (plat, plon) in presets.items():
                if st.button(name, use_container_width=True, key=f"preset_{name}"):
                    st.session_state['preset_lat'] = plat
                    st.session_state['preset_lon'] = plon

            analyze = st.button("Generate routes & barricades",
                                type="primary", use_container_width=True)

    with col2:
        # Use preset coords if a preset was clicked
        final_lat = st.session_state.get('preset_lat', lat)
        final_lon = st.session_state.get('preset_lon', lon)

        if analyze:
            with st.spinner("Computing diversion routes..."):
                try:
                    # direct in-process call — replaces the old HTTP request to :8000
                    data = routing.full_routing_analysis(
                        lat=final_lat,
                        lon=final_lon,
                        requires_closure=closure,
                        radius_m=radius,
                    )
                    st.session_state['routing_result'] = data
                    st.session_state['routing_coords'] = (final_lat, final_lon)
                except Exception as e:
                    st.error(f"Routing error: {e}")
                    st.stop()

        if 'routing_result' in st.session_state:
            data = st.session_state['routing_result']
            clat, clon = st.session_state['routing_coords']

            routes = data['diversions']['routes']
            barricades = data['barricade_points']

            if data['diversions'].get('live_traffic'):
                st.success("🟢 **LIVE traffic-aware routing (Mappls)** — route congestion "
                           "reflects current on-road traffic from Mappls, not historical "
                           "averages. Routes re-ranked by live conditions.")
            elif data['diversions'].get('capacity_aware'):
                st.success("🧠 **Capacity-aware routing active** — routes ranked to avoid "
                           "already-congested corridors and roads near other active events, "
                           "not just by distance.")
            if data['diversions'].get('status') == 'fallback':
                st.caption("ℹ️ Geometric fallback (no congestion surface). Run "
                           "`python congestion.py --data data/flipkart_gridlock.csv` to enable "
                           "capacity-aware ranking, and `python routing.py --download` for real roads.")

            _render_map(clat, clon, routes, barricades, radius)
            _render_route_cards(routes, ch)
            _render_barricade_table(barricades, ch)
        else:
            # Empty map centered on Bengaluru
            m = _base_map(12.9716, 77.5946, zoom=13)
            st_folium(m, height=480, use_container_width=True)
            st.info("Enter coordinates above and click Generate.")


def _render_map(lat, lon, routes, barricades, radius):
    m = _base_map(lat, lon, zoom=14)

    # Incident zone (red circle)
    folium.Circle(
        [lat, lon], radius=radius,
        color='#DC2626', fill=True, fill_opacity=0.25,
        tooltip="Incident zone — blocked"
    ).add_to(m)

    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color='red', icon='exclamation-sign', prefix='glyphicon'),
        tooltip="Incident location"
    ).add_to(m)

    # Diversion routes
    for i, route in enumerate(routes):
        coords = route.get('coords', [])
        if not coords:
            continue
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        label = route.get('recommendation', f'Route {i+1}')

        folium.PolyLine(
            coords,
            color=color,
            weight=5 if i == 0 else 3,
            opacity=0.9 if i == 0 else 0.65,
            tooltip=f"{label}: {route['via']} ({route['distance_km']} km, ~{route['estimated_mins']} min)"
        ).add_to(m)

        # Route start marker
        if coords:
            folium.Marker(
                coords[0],
                icon=folium.DivIcon(html=f"""
                    <div style="background:{color};color:#fff;font-weight:700;
                                padding:3px 7px;border-radius:4px;font-size:11px;
                                white-space:nowrap;">
                        R{route['route_id']}
                    </div>"""),
                tooltip=f"Route {route['route_id']} start"
            ).add_to(m)

    # Barricade points
    for bp in barricades:
        folium.Marker(
            [bp['lat'], bp['lon']],
            icon=folium.DivIcon(html="""
                <div style="background:#D97706;color:#fff;font-weight:700;
                            padding:3px 8px;border-radius:4px;font-size:12px;">
                    B
                </div>"""),
            tooltip=f"Barricade: {bp['street_name']} ({bp['distance_m']}m)"
        ).add_to(m)

    st_folium(m, height=480, use_container_width=True)


def _render_route_cards(routes, ch):
    with st.container(border=True):
        ch("route", "Diversion routes", "Ranked by congestion-aware travel time")
        cols = st.columns(len(routes))
        for i, (col, route) in enumerate(zip(cols, routes)):
            color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            live_bit = (f"&nbsp;·&nbsp;live {route.get('live_factor')}×"
                        if route.get('live_traffic') else "")
            with col:
                st.markdown(f"""
                <div class="route-card">
                    <div class="route-rec" style="color:{color};">{route['recommendation']}</div>
                    <div class="route-via">{route['via']}</div>
                    <div class="route-meta">{route['distance_km']} km &nbsp;|&nbsp; ~{route['estimated_mins']} min</div>
                    <div class="route-cong">congestion: {route.get('congestion_label', 'n/a')}
                        ({route.get('congestion_score', 0)}){live_bit}</div>
                </div>
                """, unsafe_allow_html=True)


def _render_barricade_table(barricades, ch):
    with st.container(border=True):
        ch("octagon", "Barricade entry points", "Seal every entry into the incident zone")
        if not barricades:
            st.info("No barricade points computed.")
            return

        import pandas as pd
        df = pd.DataFrame(barricades)[['street_name', 'lat', 'lon', 'distance_m']]
        df.columns = ['Street / Road', 'Latitude', 'Longitude', 'Distance from incident (m)']
        df['Latitude'] = df['Latitude'].round(5)
        df['Longitude'] = df['Longitude'].round(5)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Deploy barricades at these coordinates to seal all entry points into the incident zone.")