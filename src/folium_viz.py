"""Interactive Folium visualisation with GPS, road network, reference route, predicted route, and legend."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import folium

from src.preprocessing import TripData, clean_track


def create_folium_map(
    trip: TripData,
    out_html: Path,
    predicted_arcs: Optional[List[int]] = None,
    title: str = "PathSnap Route Map",
) -> None:
    track = clean_track(trip.track)
    m = folium.Map(
        location=[float(track["lat"].mean()), float(track["lon"].mean())],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    node_lon = trip.nodes["lon"].to_numpy()
    node_lat = trip.nodes["lat"].to_numpy()
    from_ids = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    to_ids = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)

    # Layer 1: Relevant road network near trajectory (capped for smooth browser rendering)
    net_group = folium.FeatureGroup(name="Road Network (.arcs)", show=True)
    pad = 0.0012
    lon_lo, lon_hi = track["lon"].min() - pad, track["lon"].max() + pad
    lat_lo, lat_hi = track["lat"].min() - pad, track["lat"].max() + pad
    in_box = (
        (node_lon[from_ids] >= lon_lo)
        & (node_lon[from_ids] <= lon_hi)
        & (node_lat[from_ids] >= lat_lo)
        & (node_lat[from_ids] <= lat_hi)
    )
    box_indices = np.flatnonzero(in_box)
    if len(box_indices) > 3500:
        box_indices = box_indices[:: max(1, len(box_indices) // 3500)]
    for idx in box_indices:
        f_id, t_id = from_ids[idx], to_ids[idx]
        folium.PolyLine(
            [[node_lat[f_id], node_lon[f_id]], [node_lat[t_id], node_lon[t_id]]],
            color="#94a3b8",
            weight=1.8,
            opacity=0.45,
        ).add_to(net_group)
    net_group.add_to(m)

    # Layer 2: Reference route (training / student_test only)
    if trip.route is not None:
        ref_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for arc_id in trip.route:
            f_id, t_id = from_ids[arc_id], to_ids[arc_id]
            folium.PolyLine(
                [[node_lat[f_id], node_lon[f_id]], [node_lat[t_id], node_lon[t_id]]],
                color="#16a34a",
                weight=5.0,
                opacity=0.85,
            ).add_to(ref_group)
        ref_group.add_to(m)

    # Layer 3: Predicted route
    if predicted_arcs is not None:
        pred_group = folium.FeatureGroup(name="Predicted Route", show=True)
        for arc_id in np.unique(predicted_arcs):
            f_id, t_id = from_ids[arc_id], to_ids[arc_id]
            folium.PolyLine(
                [[node_lat[f_id], node_lon[f_id]], [node_lat[t_id], node_lon[t_id]]],
                color="#dc2626",
                weight=3.0,
                dash_array="6,4",
                opacity=0.92,
            ).add_to(pred_group)
        pred_group.add_to(m)

    # Layer 4: GPS observations (downsample markers slightly only if >2500 pts)
    gps_group = folium.FeatureGroup(name="Observed GPS (.track)", show=True)
    step = max(1, len(track) // 2500)
    for row in track.iloc[::step].itertuples(index=False):
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=2.8,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.8,
            tooltip=f"t={row.timestamp}s",
        ).add_to(gps_group)
    gps_group.add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 2px solid #334155; border-radius: 6px; font-size: 13px;">
      <b>{title} (Trip {trip.trip_id})</b><br>
      <span style="color:#2563eb;">&#9679;</span> Observed GPS (.track)<br>
      <span style="color:#94a3b8;">&#9473;</span> Road Network (.arcs)<br>
      <span style="color:#16a34a;">&#9473;</span> Reference Route (.route)<br>
      <span style="color:#dc2626;">- -</span> Predicted Route
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track["lat"].min(), track["lon"].min()], [track["lat"].max(), track["lon"].max()]])

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
