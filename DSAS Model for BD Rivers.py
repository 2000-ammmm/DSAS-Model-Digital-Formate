"""
PADMA RIVER MIGRATION ANALYSIS - BANGLADESH
DSAS with STRAIGHT PARALLEL BASELINE (constant offset, straight line)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy import stats
import math
import warnings
import re
warnings.filterwarnings('ignore')

print("="*70)
print("PADMA RIVER MIGRATION ANALYSIS - BANGLADESH")
print("DSAS with STRAIGHT PARALLEL BASELINE")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = r"F:\Padma River Project\BD_Vector Files\Merged Vectors BD River Cleaned"
TARGET_CRS = 'EPSG:32646'
TRANSECT_SPACING = 500  # meters between transects
TRANSECT_LENGTH = 10000  # transect length in meters
BASELINE_OFFSET = 2000  # offset distance from river (meters)
CONFIDENCE_INTERVAL = 90

print(f"\nData folder: {DATA_FOLDER}")
print(f"Transect spacing: {TRANSECT_SPACING} meters")
print(f"Transect length: {TRANSECT_LENGTH} meters")
print(f"Baseline offset: {BASELINE_OFFSET} meters")
print(f"Confidence interval: {CONFIDENCE_INTERVAL}%")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_xy_coords(geometry):
    """Extract x and y coordinates from any geometry type"""
    x_coords = []
    y_coords = []
    
    if geometry.geom_type == 'LineString':
        x, y = geometry.xy
        return list(x), list(y)
    
    elif geometry.geom_type == 'MultiLineString':
        for line in geometry.geoms:
            x, y = line.xy
            x_coords.extend(x)
            y_coords.extend(y)
        return x_coords, y_coords
    
    elif geometry.geom_type == 'Polygon':
        x, y = geometry.exterior.xy
        return list(x), list(y)
    
    elif geometry.geom_type == 'MultiPolygon':
        for poly in geometry.geoms:
            x, y = poly.exterior.xy
            x_coords.extend(x)
            y_coords.extend(y)
        return x_coords, y_coords
    
    else:
        return [], []

def extract_single_linestring(geometry):
    """Extract a single LineString from any geometry type"""
    
    if geometry.geom_type == 'LineString':
        return geometry
    
    elif geometry.geom_type == 'MultiLineString':
        lengths = [line.length for line in geometry.geoms]
        longest_idx = np.argmax(lengths)
        return geometry.geoms[longest_idx]
    
    elif geometry.geom_type == 'Polygon':
        return geometry.boundary
    
    elif geometry.geom_type == 'MultiPolygon':
        areas = [poly.area for poly in geometry.geoms]
        largest_idx = np.argmax(areas)
        return geometry.geoms[largest_idx].boundary
    
    else:
        return geometry

def get_overall_direction_and_extent(geometry):
    """Get the overall direction, start and end points of a river"""
    
    line = extract_single_linestring(geometry)
    
    # Get all coordinates
    coords = list(line.coords)
    
    if len(coords) < 2:
        return None, None, None
    
    # Find the two most distant points (overall extent)
    min_x = min(c[0] for c in coords)
    max_x = max(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    max_y = max(c[1] for c in coords)
    
    # Determine if river is more horizontal or vertical
    width_x = max_x - min_x
    width_y = max_y - min_y
    
    if width_x > width_y:
        # River flows primarily East-West
        start_point = Point(min_x, (min_y + max_y) / 2)
        end_point = Point(max_x, (min_y + max_y) / 2)
    else:
        # River flows primarily North-South
        start_point = Point((min_x + max_x) / 2, min_y)
        end_point = Point((min_x + max_x) / 2, max_y)
    
    # Calculate direction angle
    dx = end_point.x - start_point.x
    dy = end_point.y - start_point.y
    angle = math.atan2(dy, dx)
    
    return start_point, end_point, angle

def create_straight_baseline(reference_geometry, offset_meters, offset_side='left'):
    """
    Create a straight baseline parallel to the river at a constant offset
    """
    # Get overall direction and extent
    start_point, end_point, river_angle = get_overall_direction_and_extent(reference_geometry)
    
    if start_point is None or end_point is None:
        raise ValueError("Could not determine river direction")
    
    # Calculate perpendicular angle for offset
    perp_angle = river_angle + math.pi/2
    
    # Offset the start and end points
    if offset_side == 'left':
        dx_offset = offset_meters * math.cos(perp_angle)
        dy_offset = offset_meters * math.sin(perp_angle)
    else:
        dx_offset = -offset_meters * math.cos(perp_angle)
        dy_offset = -offset_meters * math.sin(perp_angle)
    
    baseline_start = Point(start_point.x + dx_offset, start_point.y + dy_offset)
    baseline_end = Point(end_point.x + dx_offset, end_point.y + dy_offset)
    
    # Create straight baseline
    baseline = LineString([baseline_start, baseline_end])
    
    return baseline, river_angle

def convert_utm_to_latlon(x, y):
    """Convert UTM coordinates to latitude/longitude"""
    from pyproj import Transformer
    transformer = Transformer.from_crs(TARGET_CRS, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat

# ============================================================
# DSAS STATISTICAL FUNCTIONS
# ============================================================

def calculate_nsm(distances_dict):
    years = sorted([y for y in distances_dict.keys() if not np.isnan(distances_dict[y])])
    if len(years) < 2:
        return np.nan
    return distances_dict[years[-1]] - distances_dict[years[0]]

def calculate_sce(distances_dict):
    valid_dists = [d for d in distances_dict.values() if not np.isnan(d)]
    if len(valid_dists) < 2:
        return np.nan
    return max(valid_dists) - min(valid_dists)

def calculate_epr(distances_dict):
    years = sorted([y for y in distances_dict.keys() if not np.isnan(distances_dict[y])])
    if len(years) < 2:
        return np.nan
    nsm = calculate_nsm(distances_dict)
    if np.isnan(nsm):
        return np.nan
    return nsm / (years[-1] - years[0])

def calculate_epr_uncertainty(distances_dict, uncertainties_dict):
    years = sorted([y for y in distances_dict.keys() if not np.isnan(distances_dict[y])])
    if len(years) < 2:
        return np.nan
    unc_oldest = uncertainties_dict.get(years[0], np.nan)
    unc_newest = uncertainties_dict.get(years[-1], np.nan)
    if np.isnan(unc_oldest) or np.isnan(unc_newest):
        return np.nan
    return np.sqrt(unc_oldest**2 + unc_newest**2) / (years[-1] - years[0])

def calculate_lrr(years, distances):
    valid = [(x, y) for x, y in zip(years, distances) if not np.isnan(y)]
    if len(valid) < 3:
        return np.nan
    x = np.array([p[0] for p in valid])
    y = np.array([p[1] for p in valid])
    n = len(x)
    slope = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return slope

def calculate_lse(years, distances):
    valid = [(x, y) for x, y in zip(years, distances) if not np.isnan(y)]
    if len(valid) < 3:
        return np.nan
    x = np.array([p[0] for p in valid])
    y = np.array([p[1] for p in valid])
    slope = calculate_lrr(years, distances)
    if np.isnan(slope):
        return np.nan
    intercept = np.mean(y) - slope * np.mean(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    ss_residuals = np.sum(residuals**2)
    return np.sqrt(ss_residuals / (len(x) - 2))

def calculate_lci(years, distances, confidence_interval=90):
    valid = [(x, y) for x, y in zip(years, distances) if not np.isnan(y)]
    if len(valid) < 3:
        return np.nan
    x = np.array([p[0] for p in valid])
    y = np.array([p[1] for p in valid])
    lse = calculate_lse(years, distances)
    if np.isnan(lse):
        return np.nan
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean)**2)
    se_slope = lse / np.sqrt(ss_x)
    alpha = 1 - (confidence_interval / 100)
    t_critical = stats.t.ppf(1 - alpha/2, len(x) - 2)
    return t_critical * se_slope

def calculate_lr2(years, distances):
    valid = [(x, y) for x, y in zip(years, distances) if not np.isnan(y)]
    if len(valid) < 3:
        return np.nan
    x = np.array([p[0] for p in valid])
    y = np.array([p[1] for p in valid])
    slope = calculate_lrr(years, distances)
    if np.isnan(slope):
        return np.nan
    intercept = np.mean(y) - slope * np.mean(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

# ============================================================
# REPROJECTION FUNCTION
# ============================================================

def reproject_to_utm(gdf, target_crs=TARGET_CRS):
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    if gdf.crs.is_projected:
        return gdf
    return gdf.to_crs(target_crs)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "-"*50)
print("STEP 1: Loading shapefiles")
print("-"*50)

shorelines = {}
years = []

for item in os.listdir(DATA_FOLDER):
    if re.match(r'^\d{4}$', item):
        folder_path = os.path.join(DATA_FOLDER, item)
        shp_files = [f for f in os.listdir(folder_path) if f.endswith('.shp')]
        
        if shp_files:
            filepath = os.path.join(folder_path, shp_files[0])
            print(f"Loading {item}: {shp_files[0]}")
            
            try:
                gdf = gpd.read_file(filepath)
                gdf = reproject_to_utm(gdf)
                
                geom = unary_union(gdf.geometry.tolist())
                line_geom = extract_single_linestring(geom)
                
                shorelines[int(item)] = line_geom
                years.append(int(item))
                print(f"  Success: {line_geom.geom_type}, length: {line_geom.length:.2f}m")
                
            except Exception as e:
                print(f"  Error: {e}")

years = sorted(years)
print(f"\nLoaded years: {years}")

if len(shorelines) == 0:
    print("No data loaded!")
    exit()

# ============================================================
# CREATE STRAIGHT PARALLEL BASELINE
# ============================================================

print("\n" + "-"*50)
print("STEP 2: Creating STRAIGHT PARALLEL baseline")
print("-"*50)

oldest_year = years[0]
reference_shoreline = shorelines[oldest_year]

print(f"Using {oldest_year} shoreline as reference")
print(f"Reference shoreline length: {reference_shoreline.length:.2f}m")

# Create straight baseline parallel to river
try:
    baseline, river_angle = create_straight_baseline(reference_shoreline, BASELINE_OFFSET, 'left')
    print(f"  Left offset successful")
except Exception as e:
    print(f"  Left offset failed: {e}")
    baseline, river_angle = create_straight_baseline(reference_shoreline, BASELINE_OFFSET, 'right')
    print(f"  Using right offset")

baseline_length = baseline.length
print(f"  Baseline length: {baseline_length:.2f}m")
print(f"  River direction angle: {math.degrees(river_angle):.1f} degrees")

# Get start and end points for reference
start_point, end_point, _ = get_overall_direction_and_extent(reference_shoreline)
print(f"  River start point (approx): ({start_point.x:.0f}, {start_point.y:.0f})")
print(f"  River end point (approx): ({end_point.x:.0f}, {end_point.y:.0f})")

# ============================================================
# CAST TRANSECTS
# ============================================================

print("\n" + "-"*50)
print("STEP 3: Casting transects")
print("-"*50)

transects = []
transect_info = []

num_transects = int(baseline_length / TRANSECT_SPACING)

if num_transects < 1:
    print(f"WARNING: Baseline too short. Adjusting spacing to create at least 5 transects")
    actual_spacing = baseline_length / 5
    num_transects = 5
else:
    actual_spacing = TRANSECT_SPACING

print(f"Baseline length: {baseline_length:.2f}m")
print(f"Transect spacing: {actual_spacing:.2f}m")
print(f"Number of transects: {num_transects}")

# Baseline direction angle (constant for straight line)
dx = baseline.coords[-1][0] - baseline.coords[0][0]
dy = baseline.coords[-1][1] - baseline.coords[0][1]
baseline_angle = math.atan2(dy, dx)

for i in range(num_transects):
    distance_along = i * actual_spacing
    if distance_along > baseline_length:
        break
        
    center_point = baseline.interpolate(distance_along)
    
    # Perpendicular angle
    perp_angle = baseline_angle + math.pi/2
    half_length = TRANSECT_LENGTH / 2
    
    x1 = center_point.x + half_length * math.cos(perp_angle)
    y1 = center_point.y + half_length * math.sin(perp_angle)
    x2 = center_point.x - half_length * math.cos(perp_angle)
    y2 = center_point.y - half_length * math.sin(perp_angle)
    
    transect = LineString([(x1, y1), (x2, y2)])
    transects.append(transect)
    transect_info.append({
        'transect_id': i,
        'distance_along_baseline': distance_along,
        'center_x': center_point.x,
        'center_y': center_point.y,
        'baseline_angle': baseline_angle
    })

print(f"Created {len(transects)} transects")

# ============================================================
# MEASURE DISTANCES AND COLLECT INTERSECTION POINTS
# ============================================================

print("\n" + "-"*50)
print("STEP 4: Measuring distances and collecting intersection points")
print("-"*50)

measurements = []
intersection_points = {year: [] for year in years}

for t_info in transect_info:
    transect_id = t_info['transect_id']
    transect = transects[transect_id]
    center_point = Point(t_info['center_x'], t_info['center_y'])
    baseline_angle = t_info['baseline_angle']
    
    # Get Lat/Lon for transect center
    center_lon, center_lat = convert_utm_to_latlon(t_info['center_x'], t_info['center_y'])
    
    row = {
        'transect_id': transect_id, 
        'along_baseline_m': t_info['distance_along_baseline'],
        'transect_center_x_utm': t_info['center_x'],
        'transect_center_y_utm': t_info['center_y'],
        'transect_center_longitude': center_lon,
        'transect_center_latitude': center_lat
    }
    
    for year, shoreline in shorelines.items():
        try:
            intersection = transect.intersection(shoreline)
            
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    point = intersection
                elif intersection.geom_type == 'MultiPoint':
                    point = intersection.geoms[0] if len(intersection.geoms) > 0 else None
                else:
                    point = intersection.centroid if hasattr(intersection, 'centroid') else None
                
                if point is not None:
                    # Store intersection point with lat/lon
                    inter_lon, inter_lat = convert_utm_to_latlon(point.x, point.y)
                    intersection_points[year].append({
                        'transect_id': transect_id,
                        'year': year,
                        'x_utm': point.x,
                        'y_utm': point.y,
                        'longitude': inter_lon,
                        'latitude': inter_lat,
                        'distance_from_baseline_m': center_point.distance(point)
                    })
                    
                    raw_distance = center_point.distance(point)
                    
                    dx = point.x - center_point.x
                    dy = point.y - center_point.y
                    point_angle = math.atan2(dy, dx)
                    angle_diff = abs(point_angle - baseline_angle)
                    sign = -1 if angle_diff < math.pi/2 else 1
                    
                    row[year] = sign * raw_distance
                else:
                    row[year] = np.nan
            else:
                row[year] = np.nan
        except Exception as e:
            row[year] = np.nan
    
    measurements.append(row)

df_measurements = pd.DataFrame(measurements)

print("Valid measurements per year:")
for year in years:
    if year in df_measurements.columns:
        valid_count = df_measurements[year].notna().sum()
        print(f"  {year}: {valid_count}")
    else:
        print(f"  {year}: 0")

# ============================================================
# ASSIGN UNCERTAINTIES
# ============================================================

print("\n" + "-"*50)
print("STEP 5: Assigning uncertainties")
print("-"*50)

uncertainties = {}
for year in years:
    if year <= 1960:
        uncertainties[year] = 15.0
    elif year <= 2000:
        uncertainties[year] = 10.0
    else:
        uncertainties[year] = 5.0

for year, unc in uncertainties.items():
    print(f"  {year}: +- {unc} meters")

# ============================================================
# CALCULATE DSAS STATISTICS
# ============================================================

print("\n" + "-"*50)
print("STEP 6: Calculating DSAS statistics")
print("-"*50)

results_list = []

for idx, row in df_measurements.iterrows():
    transect_id = row['transect_id']
    
    distances_dict = {}
    years_list = []
    distances_list = []
    uncertainties_list = []
    
    for year in years:
        if year in row:
            dist = row[year]
            if not np.isnan(dist):
                distances_dict[year] = dist
                years_list.append(year)
                distances_list.append(dist)
                uncertainties_list.append(uncertainties[year])
    
    if len(distances_dict) < 2:
        continue
    
    result = {
        'transect_id': transect_id,
        'along_baseline_m': row['along_baseline_m'],
        'transect_center_x_utm': row['transect_center_x_utm'],
        'transect_center_y_utm': row['transect_center_y_utm'],
        'transect_center_longitude': row['transect_center_longitude'],
        'transect_center_latitude': row['transect_center_latitude']
    }
    
    # Add distance measurements for each year
    for year in years:
        result[f'distance_to_{year}_m'] = row.get(year, np.nan)
    
    result['NSM_meters'] = calculate_nsm(distances_dict)
    result['SCE_meters'] = calculate_sce(distances_dict)
    result['EPR_m_per_year'] = calculate_epr(distances_dict)
    result['EPRunc_m_per_year'] = calculate_epr_uncertainty(distances_dict, uncertainties)
    
    if len(years_list) >= 3:
        result['LRR_m_per_year'] = calculate_lrr(years_list, distances_list)
        result['LSE_meters'] = calculate_lse(years_list, distances_list)
        result['LCI_m_per_year'] = calculate_lci(years_list, distances_list, CONFIDENCE_INTERVAL)
        result['LR2'] = calculate_lr2(years_list, distances_list)
    else:
        result['LRR_m_per_year'] = np.nan
        result['LSE_meters'] = np.nan
        result['LCI_m_per_year'] = np.nan
        result['LR2'] = np.nan
    
    result['num_intersections'] = len(years_list)
    results_list.append(result)

df_results = pd.DataFrame(results_list)
print(f"Calculated statistics for {len(df_results)} transects")

# ============================================================
# CREATE VISUALIZATION MAP
# ============================================================

print("\n" + "-"*50)
print("STEP 7: Creating visualization map")
print("-"*50)

try:
    fig, ax = plt.subplots(figsize=(18, 14))
    
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(years)))
    
    # Plot each shoreline
    for i, year in enumerate(years):
        shoreline = shorelines[year]
        x_coords, y_coords = get_xy_coords(shoreline)
        if x_coords and y_coords:
            ax.plot(x_coords, y_coords, color=colors[i], linewidth=1.5, alpha=0.7, label=f'Shoreline {year}')
    
    # Plot straight baseline (thick red line)
    x_coords, y_coords = baseline.xy
    ax.plot(x_coords, y_coords, 'r-', linewidth=3, label=f'Straight Baseline (offset {BASELINE_OFFSET}m)')
    
    # Plot transects (every 10th to avoid clutter)
    step = max(1, len(transects)//50)
    for i in range(0, len(transects), step):
        if i < len(transects):
            x_coords, y_coords = transects[i].xy
            ax.plot(x_coords, y_coords, color='gray', linewidth=0.5, alpha=0.4)
    
    # Add reference shoreline (1952) in black
    ref_x, ref_y = get_xy_coords(reference_shoreline)
    if ref_x and ref_y:
        ax.plot(ref_x, ref_y, 'k--', linewidth=2, label=f'Reference ({oldest_year})')
    
    ax.set_xlabel('Easting (meters - UTM Zone 46N)', fontsize=12)
    ax.set_ylabel('Northing (meters - UTM Zone 46N)', fontsize=12)
    ax.set_title(f'Padma River Migration Analysis\nStraight Baseline at {BASELINE_OFFSET}m offset', fontsize=14)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    map_path = os.path.join(DATA_FOLDER, 'Padma_River_Straight_Parallel_Baseline_Map.png')
    plt.savefig(map_path, dpi=200)
    print(f"Map saved to: {map_path}")
    plt.close()
    
except Exception as e:
    print(f"Could not create map: {e}")

# ============================================================
# CREATE MIGRATION RATE PLOT
# ============================================================

try:
    if len(df_results) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        valid_epr = df_results.dropna(subset=['EPR_m_per_year'])
        if len(valid_epr) > 0:
            ax1 = axes[0]
            colors = ['red' if x < 0 else 'blue' for x in valid_epr['EPR_m_per_year']]
            ax1.bar(valid_epr['transect_id'], valid_epr['EPR_m_per_year'], color=colors, alpha=0.7, width=1.0)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_ylabel('End Point Rate (m/year)', fontsize=12)
            ax1.set_xlabel('Transect ID', fontsize=12)
            ax1.set_title(f'Padma River Migration Rate: {years[0]} to {years[-1]}', fontsize=14)
            ax1.grid(True, alpha=0.3)
        
        valid_nsm = df_results.dropna(subset=['NSM_meters'])
        if len(valid_nsm) > 0:
            ax2 = axes[1]
            colors = ['red' if x < 0 else 'blue' for x in valid_nsm['NSM_meters']]
            ax2.bar(valid_nsm['transect_id'], valid_nsm['NSM_meters'], color=colors, alpha=0.7, width=1.0)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('Net Shoreline Movement (meters)', fontsize=12)
            ax2.set_xlabel('Transect ID', fontsize=12)
            ax2.set_title(f'Total River Migration: {years[0]} to {years[-1]}', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(DATA_FOLDER, 'Padma_River_Migration_Rates.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Migration plot saved to: {plot_path}")
        plt.close()
        
except Exception as e:
    print(f"Could not create migration plot: {e}")

# ============================================================
# SAVE ALL RESULTS WITH LAT/LON
# ============================================================

print("\n" + "-"*50)
print("STEP 8: Saving results with Lat/Lon coordinates")
print("-"*50)

# Save main DSAS results
results_path = os.path.join(DATA_FOLDER, 'Padma_River_DSAS_Straight_Parallel_Results.csv')
df_results.to_csv(results_path, index=False)
print(f"DSAS Results saved to: {results_path}")

# Save measurements
measurements_path = os.path.join(DATA_FOLDER, 'Padma_River_All_Measurements.csv')
df_measurements.to_csv(measurements_path, index=False)
print(f"Measurements saved to: {measurements_path}")

# Save intersection points for each year with Lat/Lon
for year, points in intersection_points.items():
    if points:
        df_points = pd.DataFrame(points)
        points_path = os.path.join(DATA_FOLDER, f'Intersection_Points_{year}.csv')
        df_points.to_csv(points_path, index=False)
        print(f"Intersection points for {year} saved to: {points_path}")

# ============================================================
# SUMMARY REPORT
# ============================================================

print("\n" + "="*70)
print("SUMMARY REPORT - PADMA RIVER MIGRATION")
print("="*70)

print(f"\nAnalysis Period: {years[0]} to {years[-1]}")
print(f"Years analyzed: {years}")
print(f"Number of transects: {len(df_results)}")
print(f"Transect spacing: {actual_spacing:.2f} meters")
print(f"Baseline type: STRAIGHT PARALLEL (offset {BASELINE_OFFSET}m)")

if len(df_results) > 0:
    valid_epr = df_results['EPR_m_per_year'].dropna()
    if len(valid_epr) > 0:
        print("\nEND POINT RATE (EPR):")
        print(f"  Average: {valid_epr.mean():.4f} m/year")
        print(f"  Standard deviation: {valid_epr.std():.4f} m/year")
        print(f"  Minimum (erosion): {valid_epr.min():.4f} m/year")
        print(f"  Maximum (accretion): {valid_epr.max():.4f} m/year")
        
        erosion_count = len(valid_epr[valid_epr < 0])
        accretion_count = len(valid_epr[valid_epr > 0])
        print(f"  Eroding transects: {erosion_count} ({100*erosion_count/len(valid_epr):.1f}%)")
        print(f"  Accreting transects: {accretion_count} ({100*accretion_count/len(valid_epr):.1f}%)")
    
    valid_nsm = df_results['NSM_meters'].dropna()
    if len(valid_nsm) > 0:
        print("\nNET SHORELINE MOVEMENT (NSM):")
        print(f"  Average: {valid_nsm.mean():.2f} meters")
        print(f"  Minimum: {valid_nsm.min():.2f} meters")
        print(f"  Maximum: {valid_nsm.max():.2f} meters")
        direction = "accretion (toward baseline)" if valid_nsm.mean() > 0 else "erosion (away from baseline)"
        print(f"  Direction: {direction}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

print(f"\nOutput files saved in: {DATA_FOLDER}")
print("  - Padma_River_DSAS_Straight_Parallel_Results.csv (All DSAS statistics)")
print("  - Padma_River_All_Measurements.csv (Raw distance measurements)")
print("  - Intersection_Points_YYYY.csv (Intersection points with Lat/Lon for each year)")
print("  - Padma_River_Straight_Parallel_Baseline_Map.png (Visualization map)")
print("  - Padma_River_Migration_Rates.png (Rate bar chart)")








