#!/usr/bin/env python3
"""
FIT Toolkit
A web-based application to adjust timestamps in Garmin FIT files.
Supports batch processing and setting a new start date/time.

Zero dependencies — uses only the Python standard library.

Usage:
    python3 fit_time_adjuster.py

Then open http://localhost:5050 in your browser (opens automatically).
"""

import struct
import os
import io
import json
import uuid
import zipfile
import webbrowser
import threading
import urllib.parse
import sys
import subprocess
import signal
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone


# ==============================================================================
# FIT Binary Format Constants & CRC
# ==============================================================================

GARMIN_EPOCH_OFFSET = 631065600
FIT_SIGNATURE = b'.FIT'

TIMESTAMP_FIELD_MAP = {
    253: None,
    4: {0},
    2: {18, 19, 21, 23, 34, 49, 79},
    5: {34},
}

CRC_TABLE = [
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400
]

INVALID_TIMESTAMP = 0xFFFFFFFF
INVALID_LATLON = 0x7FFFFFFF  # Sentinel for "no GPS fix"
SEMICIRCLES_TO_DEG = 180.0 / (2**31)
RECORD_MSG_NUM = 20   # Global message number for record (per-second data)
SESSION_MSG_NUM = 18   # Global message number for session summary
LAP_MSG_NUM = 19       # Global message number for lap summary

# Invalid sentinel values by size
INVALID_UINT8 = 0xFF
INVALID_UINT16 = 0xFFFF
INVALID_UINT32 = 0xFFFFFFFF
INVALID_SINT8 = 0x7F

# Session fields to extract: field_def_num → (name, format, scale_divisor)
# format: 'B'=uint8, 'H'=uint16, 'I'=uint32, 'b'=sint8
SESSION_FIELDS = {
    7:  ('total_elapsed_time', 'I', 1000),   # ms → s
    8:  ('total_timer_time', 'I', 1000),     # ms → s
    9:  ('total_distance', 'I', 100),        # cm → m
    11: ('total_calories', 'H', 1),
    14: ('avg_speed', 'H', 1000),            # mm/s → m/s
    15: ('max_speed', 'H', 1000),
    16: ('avg_heart_rate', 'B', 1),
    17: ('max_heart_rate', 'B', 1),
    18: ('avg_cadence', 'B', 1),
    19: ('max_cadence', 'B', 1),
    20: ('avg_power', 'H', 1),
    21: ('max_power', 'H', 1),
    22: ('total_ascent', 'H', 1),
    23: ('total_descent', 'H', 1),
    34: ('normalized_power', 'H', 1),
    57: ('avg_temperature', 'b', 1),         # sint8
    124: ('enhanced_avg_speed', 'I', 1000),  # prefer over field 14
    125: ('enhanced_max_speed', 'I', 1000),  # prefer over field 15
}

# Lap fields to extract: field_def_num → (name, format, scale_divisor)
LAP_FIELDS = {
    253: ('timestamp', 'I', 1),
    7:   ('total_elapsed_time', 'I', 1000),   # ms → s
    8:   ('total_timer_time', 'I', 1000),     # ms → s
    9:   ('total_distance', 'I', 100),        # cm → m
    11:  ('total_calories', 'H', 1),
    14:  ('avg_speed', 'H', 1000),            # mm/s → m/s
    15:  ('max_speed', 'H', 1000),
    16:  ('avg_heart_rate', 'B', 1),
    17:  ('max_heart_rate', 'B', 1),
    18:  ('avg_cadence', 'B', 1),
    19:  ('max_cadence', 'B', 1),
    20:  ('avg_power', 'H', 1),
    21:  ('max_power', 'H', 1),
    22:  ('total_ascent', 'H', 1),
    23:  ('total_descent', 'H', 1),
    34:  ('normalized_power', 'H', 1),
    124: ('enhanced_avg_speed', 'I', 1000),
    125: ('enhanced_max_speed', 'I', 1000),
}

# Default thresholds for moment detection
DEFAULT_SPEED_SURGE_THRESHOLD = 13.89   # m/s (50 km/h)
DEFAULT_POWER_SPIKE_THRESHOLD = 400     # watts
DEFAULT_SPRINT_POWER_THRESHOLD = 400    # watts
DEFAULT_SPRINT_ACCEL_THRESHOLD = 1.0    # m/s²
DEFAULT_SPRINT_MIN_DURATION = 3         # seconds
DEFAULT_CLIMB_GRADIENT_THRESHOLD = 0.05 # 5%
DEFAULT_CLIMB_MIN_DURATION = 30         # seconds
DEFAULT_CLIMB_MIN_ELEVATION_GAIN = 10   # meters
DEFAULT_GROUP_TIME_WINDOW = 60          # seconds

# Record fields to extract: field_def_num → (name, format, scale_divisor, offset_sub)
RECORD_FIELDS = {
    253: ('timestamp', 'I', 1, 0),
    2:   ('altitude', 'H', 5, 500),          # (raw/5) - 500 → m
    3:   ('heart_rate', 'B', 1, 0),
    4:   ('cadence', 'B', 1, 0),
    5:   ('distance', 'I', 100, 0),          # cm → m
    6:   ('speed', 'H', 1000, 0),            # mm/s → m/s
    7:   ('power', 'H', 1, 0),
    13:  ('temperature', 'b', 1, 0),         # sint8
    73:  ('enhanced_speed', 'I', 1000, 0),   # mm/s → m/s, prefer over field 6
    78:  ('enhanced_altitude', 'I', 5, 500), # prefer over field 2
}


def crc16_fit(data: bytes) -> int:
    crc = 0
    for byte in data:
        tmp = CRC_TABLE[crc & 0xF]
        crc = ((crc >> 4) & 0x0FFF) ^ tmp ^ CRC_TABLE[byte & 0xF]
        tmp = CRC_TABLE[crc & 0xF]
        crc = ((crc >> 4) & 0x0FFF) ^ tmp ^ CRC_TABLE[(byte >> 4) & 0xF]
    return crc & 0xFFFF


def garmin_to_datetime(garmin_ts: int) -> datetime:
    if garmin_ts == INVALID_TIMESTAMP or garmin_ts == 0:
        return None
    return datetime.fromtimestamp(garmin_ts + GARMIN_EPOCH_OFFSET, tz=timezone.utc)


# ==============================================================================
# FIT Binary Parser & Patcher
# ==============================================================================

class FITFile:
    def __init__(self, data: bytes):
        self.data = bytearray(data)
        self.header_size = 0
        self.data_size = 0
        self.data_start = 0
        self.data_end = 0
        self.definitions = {}
        self.first_timestamp = None
        self.timestamp_locations = []
        self.gps_points = []  # List of (lat, lon) in decimal degrees
        self.session_stats = {}  # Summary stats from session message
        self.records = []  # Time-series data from record messages
        self.laps = []  # Per-lap summaries from lap messages
        self._parse()

    def _parse(self):
        self._parse_header()
        self._scan_records()

    def _parse_header(self):
        if len(self.data) < 12:
            raise ValueError("File too small to be a valid FIT file")
        self.header_size = self.data[0]
        if self.header_size not in (12, 14):
            raise ValueError(f"Unexpected header size: {self.header_size}")
        if self.data[8:12] != FIT_SIGNATURE:
            raise ValueError("Invalid FIT file signature")
        self.data_size = struct.unpack_from('<I', self.data, 4)[0]
        self.data_start = self.header_size
        self.data_end = self.data_start + self.data_size

        if self.header_size == 14:
            stored = struct.unpack_from('<H', self.data, 12)[0]
            if stored != 0x0000:  # Header CRC is optional; 0x0000 means not set
                calc = crc16_fit(self.data[0:12])
                if stored != calc:
                    raise ValueError(f"Header CRC mismatch: stored=0x{stored:04X}, calc=0x{calc:04X}")

        if len(self.data) >= self.data_end + 2:
            stored = struct.unpack_from('<H', self.data, self.data_end)[0]
            calc = crc16_fit(self.data[0:self.data_end])
            if stored != calc:
                raise ValueError(f"File CRC mismatch: stored=0x{stored:04X}, calc=0x{calc:04X}")

    def _is_timestamp_field(self, field_def_num, global_msg_num):
        if field_def_num not in TIMESTAMP_FIELD_MAP:
            return False
        allowed = TIMESTAMP_FIELD_MAP[field_def_num]
        return allowed is None or global_msg_num in allowed

    def _scan_records(self):
        pos = self.data_start
        self.definitions = {}
        self.timestamp_locations = []
        self.first_timestamp = None
        self.gps_points = []
        self.session_stats = {}
        self.records = []
        self.laps = []

        while pos < self.data_end and pos < len(self.data):
            rh = self.data[pos]
            pos += 1

            if rh & 0x80:
                lmt = (rh >> 5) & 0x03
                if lmt not in self.definitions:
                    raise ValueError(f"Compressed msg refs undefined type {lmt} at 0x{pos-1:X}")
                defn = self.definitions[lmt]
                self._collect_timestamps(pos, defn)
                self._collect_gps(pos, defn)
                self._collect_session(pos, defn)
                self._collect_record_data(pos, defn)
                self._collect_lap_data(pos, defn)
                pos += defn['total_size']

            elif rh & 0x40:
                lmt = rh & 0x0F
                has_dev = bool(rh & 0x20)
                pos += 1  # reserved
                arch = self.data[pos]; pos += 1
                endian = '<' if arch == 0 else '>'
                gmn = struct.unpack_from(f'{endian}H', self.data, pos)[0]; pos += 2
                nf = self.data[pos]; pos += 1

                total = 0
                ts_fields = []
                gps_fields = {}
                session_fields = {}  # fdn → (offset, size, fmt)
                record_fields = {}   # fdn → (offset, size, fmt)
                lap_fields = {}      # fdn → (offset, size)
                for _ in range(nf):
                    fdn, fsz, ftype = self.data[pos], self.data[pos+1], self.data[pos+2]
                    pos += 3
                    if fsz == 4 and self._is_timestamp_field(fdn, gmn):
                        ts_fields.append((total, fsz, fdn))
                    if gmn == RECORD_MSG_NUM and fdn in (0, 1) and fsz == 4:
                        gps_fields[fdn] = total
                    # Track session fields
                    if gmn == SESSION_MSG_NUM and fdn in SESSION_FIELDS:
                        session_fields[fdn] = (total, fsz)
                    # Track lap fields
                    if gmn == LAP_MSG_NUM and fdn in LAP_FIELDS:
                        lap_fields[fdn] = (total, fsz)
                    # Track record fields
                    if gmn == RECORD_MSG_NUM and fdn in RECORD_FIELDS:
                        record_fields[fdn] = (total, fsz)
                    total += fsz

                dev_total = 0
                if has_dev:
                    ndf = self.data[pos]; pos += 1
                    for _ in range(ndf):
                        dev_total += self.data[pos+1]; pos += 3

                self.definitions[lmt] = {
                    'endian': endian, 'total_size': total + dev_total,
                    'global_msg_num': gmn,
                    'timestamp_fields': ts_fields,
                    'gps_fields': gps_fields,
                    'session_fields': session_fields,
                    'record_fields': record_fields,
                    'lap_fields': lap_fields,
                }
            else:
                lmt = rh & 0x0F
                if lmt not in self.definitions:
                    raise ValueError(f"Data msg refs undefined type {lmt} at 0x{pos-1:X}")
                defn = self.definitions[lmt]
                self._collect_timestamps(pos, defn)
                self._collect_gps(pos, defn)
                self._collect_session(pos, defn)
                self._collect_record_data(pos, defn)
                self._collect_lap_data(pos, defn)
                pos += defn['total_size']

    def _collect_timestamps(self, data_start, defn):
        endian = defn['endian']
        for offset, size, fdn in defn['timestamp_fields']:
            aoff = data_start + offset
            if aoff + 4 <= len(self.data):
                val = struct.unpack_from(f'{endian}I', self.data, aoff)[0]
                if val != INVALID_TIMESTAMP and val != 0:
                    self.timestamp_locations.append({
                        'offset': aoff, 'endian': endian, 'value': val,
                    })
                    if self.first_timestamp is None:
                        self.first_timestamp = val

    def _collect_gps(self, data_start, defn):
        gps_fields = defn.get('gps_fields', {})
        if 0 not in gps_fields or 1 not in gps_fields:
            return
        endian = defn['endian']
        lat_off = data_start + gps_fields[0]
        lon_off = data_start + gps_fields[1]
        if lat_off + 4 > len(self.data) or lon_off + 4 > len(self.data):
            return
        # Read as signed int32
        lat_raw = struct.unpack_from(f'{endian}i', self.data, lat_off)[0]
        lon_raw = struct.unpack_from(f'{endian}i', self.data, lon_off)[0]
        # Skip invalid GPS fixes
        if lat_raw == INVALID_LATLON or lon_raw == INVALID_LATLON:
            return
        if lat_raw == 0 and lon_raw == 0:
            return
        lat = lat_raw * SEMICIRCLES_TO_DEG
        lon = lon_raw * SEMICIRCLES_TO_DEG
        # Store record index for map-chart linking
        rec_idx = len(self.records)  # current record count (GPS collected before record in same msg)
        self.gps_points.append((round(lat, 6), round(lon, 6), rec_idx))

    def _read_field(self, data_start, offset, size, fmt, endian):
        """Read a single field value, applying the correct struct format."""
        aoff = data_start + offset
        if aoff + size > len(self.data):
            return None
        fmt_map = {'B': 'B', 'H': 'H', 'I': 'I', 'b': 'b'}
        sf = fmt_map.get(fmt)
        if sf is None:
            return None
        # Ensure size matches expected
        expected = struct.calcsize(sf)
        if size < expected:
            return None
        val = struct.unpack_from(f'{endian}{sf}', self.data, aoff)[0]
        # Check invalid sentinels
        if fmt == 'B' and val == INVALID_UINT8:
            return None
        if fmt == 'H' and val == INVALID_UINT16:
            return None
        if fmt == 'I' and val == INVALID_UINT32:
            return None
        if fmt == 'b' and val == INVALID_SINT8:
            return None
        return val

    def _collect_session(self, data_start, defn):
        """Extract summary stats from session message (global msg 18)."""
        session_fields = defn.get('session_fields', {})
        if not session_fields:
            return
        endian = defn['endian']
        for fdn, (offset, size) in session_fields.items():
            meta = SESSION_FIELDS[fdn]
            name, fmt, scale = meta
            val = self._read_field(data_start, offset, size, fmt, endian)
            if val is not None:
                scaled = val / scale if scale > 1 else val
                self.session_stats[name] = round(scaled, 3) if isinstance(scaled, float) else scaled
        # Prefer enhanced fields
        if 'enhanced_avg_speed' in self.session_stats:
            self.session_stats['avg_speed'] = self.session_stats.pop('enhanced_avg_speed')
        if 'enhanced_max_speed' in self.session_stats:
            self.session_stats['max_speed'] = self.session_stats.pop('enhanced_max_speed')

    def _collect_record_data(self, data_start, defn):
        """Extract time-series data from record messages (global msg 20)."""
        record_fields = defn.get('record_fields', {})
        if not record_fields:
            return
        endian = defn['endian']
        rec = {}
        for fdn, (offset, size) in record_fields.items():
            name, fmt, scale, off_sub = RECORD_FIELDS[fdn]
            val = self._read_field(data_start, offset, size, fmt, endian)
            if val is not None:
                scaled = val / scale if scale > 1 else val
                if off_sub:
                    scaled = scaled - off_sub
                rec[name] = round(scaled, 2) if isinstance(scaled, float) else scaled
        # Prefer enhanced fields
        if 'enhanced_altitude' in rec:
            rec['altitude'] = rec.pop('enhanced_altitude')
        if 'enhanced_speed' in rec:
            rec['speed'] = rec.pop('enhanced_speed')
        if rec:
            self.records.append(rec)

    def _collect_lap_data(self, data_start, defn):
        """Extract lap summaries from lap messages (global msg 19)."""
        lap_fields = defn.get('lap_fields', {})
        if not lap_fields:
            return
        endian = defn['endian']
        lap = {}
        for fdn, (offset, size) in lap_fields.items():
            name, fmt, scale = LAP_FIELDS[fdn]
            val = self._read_field(data_start, offset, size, fmt, endian)
            if val is not None:
                scaled = val / scale if scale > 1 else val
                lap[name] = round(scaled, 3) if isinstance(scaled, float) else scaled
        # Prefer enhanced fields
        if 'enhanced_avg_speed' in lap:
            lap['avg_speed'] = lap.pop('enhanced_avg_speed')
        if 'enhanced_max_speed' in lap:
            lap['max_speed'] = lap.pop('enhanced_max_speed')
        if lap:
            self.laps.append(lap)

    def get_start_datetime(self):
        if self.first_timestamp is None:
            return None
        return garmin_to_datetime(self.first_timestamp)

    def adjust_timestamps(self, offset_seconds):
        count = 0
        for loc in self.timestamp_locations:
            aoff, endian = loc['offset'], loc['endian']
            val = struct.unpack_from(f'{endian}I', self.data, aoff)[0]
            if val != INVALID_TIMESTAMP and val != 0:
                new_val = max(0, val + offset_seconds) & 0xFFFFFFFF
                struct.pack_into(f'{endian}I', self.data, aoff, new_val)
                count += 1
        return count

    def recalculate_crcs(self):
        if self.header_size == 14:
            struct.pack_into('<H', self.data, 12, crc16_fit(self.data[0:12]))
        if len(self.data) >= self.data_end + 2:
            struct.pack_into('<H', self.data, self.data_end, crc16_fit(self.data[0:self.data_end]))

    def get_bytes(self):
        return bytes(self.data)


def process_fit_bytes(file_bytes, new_start_dt):
    fit = FITFile(file_bytes)
    original_dt = fit.get_start_datetime()
    if original_dt is None:
        raise ValueError("No timestamps found in FIT file")
    offset_seconds = int((new_start_dt - original_dt).total_seconds())
    if offset_seconds == 0:
        raise ValueError("New start time is the same as the original")
    count = fit.adjust_timestamps(offset_seconds)
    fit.recalculate_crcs()
    return {
        'output_bytes': fit.get_bytes(),
        'original_start': original_dt,
        'new_start': new_start_dt,
        'offset_seconds': offset_seconds,
        'timestamps_modified': count,
    }


# ==============================================================================
# Route Similarity (Fréchet distance + overlap)
# ==============================================================================

import math

def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between two (lat, lon) points."""
    R = 6_371_000  # Earth radius in metres
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Forward azimuth in degrees (0=N, 90=E, 180=S, 270=W)."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _describe_route_shape(records, gps_lookup, start_idx, end_idx):
    """Analyze road shape over a record index range. Returns dict with turn/gradient description."""
    # Collect GPS points in padded range
    pad_start = max(0, start_idx - 5)
    pad_end = min(len(records) - 1, end_idx + 5)
    pts = []
    for idx in range(pad_start, pad_end + 1):
        gps = gps_lookup.get(idx)
        if gps:
            pts.append((gps[0], gps[1], idx))

    # Compute total bearing change
    total_bearing_change = 0.0
    if len(pts) >= 3:
        bearings = []
        for k in range(len(pts) - 1):
            bearings.append(_bearing_deg(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]))
        for k in range(len(bearings) - 1):
            delta = bearings[k + 1] - bearings[k]
            # Normalize to -180..+180
            while delta > 180:
                delta -= 360
            while delta < -180:
                delta += 360
            total_bearing_change += delta

    abs_turn = abs(total_bearing_change)
    if abs_turn < 15:
        turn_label = 'straight road'
    elif abs_turn < 45:
        turn_label = 'gentle curve ' + ('right' if total_bearing_change > 0 else 'left')
    elif abs_turn < 90:
        turn_label = 'curve ' + ('right' if total_bearing_change > 0 else 'left')
    elif abs_turn < 135:
        turn_label = 'sharp turn ' + ('right' if total_bearing_change > 0 else 'left')
    elif abs_turn < 170:
        turn_label = 'hairpin ' + ('right' if total_bearing_change > 0 else 'left')
    else:
        turn_label = 'U-turn'

    # Compute gradient from altitude
    gradient_pct = 0.0
    gradient_desc = 'flat'
    s_idx = max(0, start_idx)
    e_idx = min(len(records) - 1, end_idx)
    start_alt = None
    end_alt = None
    for idx in range(s_idx, e_idx + 1):
        alt = records[idx].get('altitude')
        if alt is not None:
            if start_alt is None:
                start_alt = alt
            end_alt = alt

    if start_alt is not None and end_alt is not None:
        # Compute horizontal distance
        h_dist = 0.0
        prev_pt = None
        for idx in range(s_idx, e_idx + 1):
            gps = gps_lookup.get(idx)
            if gps and prev_pt:
                h_dist += _haversine_m(prev_pt[0], prev_pt[1], gps[0], gps[1])
            if gps:
                prev_pt = gps
        if h_dist > 5:
            gradient_pct = ((end_alt - start_alt) / h_dist) * 100
            g = gradient_pct
            if g < -3:
                gradient_desc = 'downhill'
            elif g < 1:
                gradient_desc = 'flat'
            elif g < 4:
                gradient_desc = 'slight uphill'
            elif g < 8:
                gradient_desc = 'uphill'
            elif g < 12:
                gradient_desc = 'steep climb'
            else:
                gradient_desc = 'very steep'

    # Compose description
    gradient_pct = round(gradient_pct, 1)
    if gradient_desc == 'flat':
        desc = turn_label + ' on flat road'
    elif gradient_desc == 'downhill':
        desc = turn_label + ', ' + gradient_desc + ' ' + str(gradient_pct) + '%'
    else:
        desc = turn_label + ' on ' + gradient_desc + ' ' + str(gradient_pct) + '%'
        if gradient_desc in ('steep climb', 'very steep'):
            desc = turn_label + ' on ' + gradient_desc + ' ' + str(gradient_pct) + '% climb'

    return {
        'description': desc,
        'turn': turn_label,
        'turn_angle': round(total_bearing_change, 1),
        'gradient_pct': gradient_pct,
        'gradient_desc': gradient_desc,
    }


def _discrete_frechet(P, Q, dist_fn, max_pts=500):
    """Compute the discrete Fréchet distance between two polylines.
    P, Q: lists of (lat, lon) tuples.
    Returns distance in the same unit as dist_fn (metres)."""
    # Downsample for performance — O(n*m) memory + time
    if len(P) > max_pts:
        step = len(P) / max_pts
        P = [P[int(i * step)] for i in range(max_pts)]
    if len(Q) > max_pts:
        step = len(Q) / max_pts
        Q = [Q[int(i * step)] for i in range(max_pts)]
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float('inf')
    # DP table (flat list for speed)
    ca = [-1.0] * (n * m)
    def _c(i, j):
        idx = i * m + j
        if ca[idx] >= 0:
            return ca[idx]
        d = dist_fn(P[i][0], P[i][1], Q[j][0], Q[j][1])
        if i == 0 and j == 0:
            ca[idx] = d
        elif i == 0:
            ca[idx] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[idx] = max(_c(i - 1, 0), d)
        else:
            ca[idx] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[idx]
    # Avoid Python recursion limit — use iterative bottom-up instead
    for i in range(n):
        for j in range(m):
            d = dist_fn(P[i][0], P[i][1], Q[j][0], Q[j][1])
            idx = i * m + j
            if i == 0 and j == 0:
                ca[idx] = d
            elif i == 0:
                ca[idx] = max(ca[j - 1], d)
            elif j == 0:
                ca[idx] = max(ca[(i - 1) * m], d)
            else:
                ca[idx] = max(min(ca[(i - 1) * m + j], ca[(i - 1) * m + (j - 1)], ca[i * m + (j - 1)]), d)
    return ca[n * m - 1]


def _overlap_pct(P, Q, threshold_m=50.0, max_pts=1000):
    """Compute what percentage of points in P are within threshold_m of any point in Q.
    Returns (pct_P_near_Q, pct_Q_near_P) as 0-100 floats."""
    if not P or not Q:
        return 0.0, 0.0
    # Downsample for performance
    if len(P) > max_pts:
        step = len(P) / max_pts
        P = [P[int(i * step)] for i in range(max_pts)]
    if len(Q) > max_pts:
        step = len(Q) / max_pts
        Q = [Q[int(i * step)] for i in range(max_pts)]

    def _near_count(source, target):
        count = 0
        for sp in source:
            for tp in target:
                if _haversine_m(sp[0], sp[1], tp[0], tp[1]) <= threshold_m:
                    count += 1
                    break
        return count

    near_pq = _near_count(P, Q)
    near_qp = _near_count(Q, P)
    return round(near_pq / len(P) * 100, 1), round(near_qp / len(Q) * 100, 1)


def compute_route_similarity(gps_a, gps_b):
    """Compute similarity between two GPS point lists.
    Each list contains (lat, lon, rec_idx) tuples.
    Returns dict with frechet_m, frechet_score (0-100), overlap_a, overlap_b, overlap_avg."""
    # Strip rec_idx for distance calculations
    P = [(p[0], p[1]) for p in gps_a]
    Q = [(p[0], p[1]) for p in gps_b]
    if len(P) < 2 or len(Q) < 2:
        return None

    frechet_m = _discrete_frechet(P, Q, _haversine_m)
    # Convert Fréchet distance to a 0-100 score:
    # 0m → 100%, 50m → ~95%, 200m → ~80%, 1000m → ~37%, 5000m → ~1%
    # Using exponential decay: score = 100 * exp(-d / 300)
    frechet_score = round(100 * math.exp(-frechet_m / 300), 1)
    frechet_score = max(0.0, min(100.0, frechet_score))

    overlap_a, overlap_b = _overlap_pct(P, Q, threshold_m=50.0)
    overlap_avg = round((overlap_a + overlap_b) / 2, 1)

    return {
        'frechet_m': round(frechet_m, 1),
        'frechet_score': frechet_score,
        'overlap_a': overlap_a,
        'overlap_b': overlap_b,
        'overlap_avg': overlap_avg,
    }


# ==============================================================================
# Moment Detection & Group Analysis
# ==============================================================================

def _parse_thresholds(qs):
    """Extract moment detection thresholds from query string with defaults."""
    def _f(key, default):
        return float(qs.get(key, [default])[0])
    return {
        'speed_surge': _f('speed_surge', DEFAULT_SPEED_SURGE_THRESHOLD),
        'power_spike': _f('power_spike', DEFAULT_POWER_SPIKE_THRESHOLD),
        'sprint_power': _f('sprint_power', DEFAULT_SPRINT_POWER_THRESHOLD),
        'sprint_accel': _f('sprint_accel', DEFAULT_SPRINT_ACCEL_THRESHOLD),
        'sprint_min_duration': _f('sprint_min_duration', DEFAULT_SPRINT_MIN_DURATION),
        'climb_gradient': _f('climb_gradient', DEFAULT_CLIMB_GRADIENT_THRESHOLD),
        'climb_min_duration': _f('climb_min_duration', DEFAULT_CLIMB_MIN_DURATION),
        'climb_min_elevation_gain': _f('climb_min_elevation_gain', DEFAULT_CLIMB_MIN_ELEVATION_GAIN),
    }


def detect_moments(records, gps_points, thresholds):
    """Detect notable moments (speed surges, power spikes, sprints, climbs) from records."""
    # Build rec_idx → (lat, lon) lookup
    gps_lookup = {}
    if gps_points:
        for pt in gps_points:
            gps_lookup[pt[2]] = (pt[0], pt[1])

    moments = []

    # Sprint state machine
    sprint_active = False
    sprint_start_ts = 0
    sprint_start_idx = 0
    sprint_peak_power = 0
    sprint_peak_speed = 0
    sprint_last_ts = 0

    # Climb state machine
    climb_active = False
    climb_start_ts = 0
    climb_start_idx = 0
    climb_start_alt = 0
    climb_elevation_gain = 0
    climb_last_alt = 0
    climb_last_ts = 0

    prev_speed = None
    prev_ts = None

    for i, rec in enumerate(records):
        ts = rec.get('timestamp')
        speed = rec.get('speed')
        power = rec.get('power')
        alt = rec.get('altitude')
        dist = rec.get('distance')

        gps = gps_lookup.get(i)
        lat = gps[0] if gps else None
        lon = gps[1] if gps else None

        if ts is None:
            continue

        # Speed surge detection
        if speed is not None and speed > thresholds['speed_surge']:
            moments.append({
                'type': 'speed_surge',
                'timestamp': ts,
                'value': round(speed, 2),
                'lat': lat, 'lon': lon,
                'route_shape': _describe_route_shape(records, gps_lookup, i - 5, i + 5),
            })

        # Power spike detection
        if power is not None and power > 0 and power > thresholds['power_spike']:
            moments.append({
                'type': 'power_spike',
                'timestamp': ts,
                'value': round(power, 1),
                'lat': lat, 'lon': lon,
                'route_shape': _describe_route_shape(records, gps_lookup, i - 5, i + 5),
            })

        # Acceleration calculation
        accel = 0
        if speed is not None and prev_speed is not None and prev_ts is not None:
            dt = ts - prev_ts
            if 0 < dt <= 5:
                accel = (speed - prev_speed) / dt

        # Sprint state machine
        is_sprint_trigger = (
            (power is not None and power > thresholds['sprint_power']) or
            (accel > thresholds['sprint_accel']) or
            (speed is not None and speed > thresholds['speed_surge'])
        )

        if sprint_active:
            gap = ts - sprint_last_ts if sprint_last_ts else 0
            if gap > 5 or not is_sprint_trigger:
                # End sprint
                duration = sprint_last_ts - sprint_start_ts
                if duration >= thresholds['sprint_min_duration']:
                    s_gps = gps_lookup.get(sprint_start_idx)
                    moments.append({
                        'type': 'sprint',
                        'timestamp': sprint_start_ts,
                        'value': round(sprint_peak_power, 1),
                        'peak_speed': round(sprint_peak_speed, 2),
                        'duration': round(duration, 1),
                        'lat': s_gps[0] if s_gps else None,
                        'lon': s_gps[1] if s_gps else None,
                        'route_shape': _describe_route_shape(records, gps_lookup, sprint_start_idx, i),
                    })
                sprint_active = False
            else:
                if power is not None and power > sprint_peak_power:
                    sprint_peak_power = power
                if speed is not None and speed > sprint_peak_speed:
                    sprint_peak_speed = speed
                sprint_last_ts = ts
        elif is_sprint_trigger:
            sprint_active = True
            sprint_start_ts = ts
            sprint_start_idx = i
            sprint_peak_power = power if power else 0
            sprint_peak_speed = speed if speed else 0
            sprint_last_ts = ts

        # Climb state machine
        if alt is not None and prev_ts is not None:
            dt = ts - prev_ts
            if climb_active:
                if dt > 0 and climb_last_alt is not None:
                    alt_delta = alt - climb_last_alt
                    if alt_delta > 0:
                        climb_elevation_gain += alt_delta
                    # Check gradient over recent segment
                    # Use distance if available, otherwise estimate
                    if speed is not None and speed > 0.5 and dt > 0:
                        h_dist = speed * dt
                        if h_dist > 0:
                            gradient = alt_delta / h_dist
                        else:
                            gradient = 0
                    else:
                        gradient = 0

                    # End climb if gradient drops below threshold
                    if gradient < -thresholds['climb_gradient'] or (ts - climb_last_ts > 10 and gradient < 0):
                        duration = climb_last_ts - climb_start_ts
                        if duration >= thresholds['climb_min_duration'] and climb_elevation_gain >= thresholds['climb_min_elevation_gain']:
                            c_gps = gps_lookup.get(climb_start_idx)
                            moments.append({
                                'type': 'climb',
                                'timestamp': climb_start_ts,
                                'value': round(climb_elevation_gain, 1),
                                'duration': round(duration, 1),
                                'lat': c_gps[0] if c_gps else None,
                                'lon': c_gps[1] if c_gps else None,
                                'route_shape': _describe_route_shape(records, gps_lookup, climb_start_idx, i),
                            })
                        climb_active = False
                climb_last_alt = alt
                climb_last_ts = ts
            else:
                # Check if climb starts
                if speed is not None and speed > 0.5 and dt > 0:
                    h_dist = speed * dt
                    if h_dist > 0 and climb_last_alt is not None:
                        gradient = (alt - climb_last_alt) / h_dist
                        if gradient > thresholds['climb_gradient']:
                            climb_active = True
                            climb_start_ts = ts
                            climb_start_idx = i
                            climb_start_alt = alt
                            climb_elevation_gain = 0
                            climb_last_ts = ts
                climb_last_alt = alt

        if speed is not None:
            prev_speed = speed
        prev_ts = ts

    # Flush active sprint
    if sprint_active:
        duration = sprint_last_ts - sprint_start_ts
        if duration >= thresholds['sprint_min_duration']:
            s_gps = gps_lookup.get(sprint_start_idx)
            moments.append({
                'type': 'sprint',
                'timestamp': sprint_start_ts,
                'value': round(sprint_peak_power, 1),
                'peak_speed': round(sprint_peak_speed, 2),
                'duration': round(duration, 1),
                'lat': s_gps[0] if s_gps else None,
                'lon': s_gps[1] if s_gps else None,
                'route_shape': _describe_route_shape(records, gps_lookup, sprint_start_idx, len(records) - 1),
            })

    # Flush active climb
    if climb_active:
        duration = climb_last_ts - climb_start_ts
        if duration >= thresholds['climb_min_duration'] and climb_elevation_gain >= thresholds['climb_min_elevation_gain']:
            c_gps = gps_lookup.get(climb_start_idx)
            moments.append({
                'type': 'climb',
                'timestamp': climb_start_ts,
                'value': round(climb_elevation_gain, 1),
                'duration': round(duration, 1),
                'lat': c_gps[0] if c_gps else None,
                'lon': c_gps[1] if c_gps else None,
                'route_shape': _describe_route_shape(records, gps_lookup, climb_start_idx, len(records) - 1),
            })

    return moments


def detect_achievements(records, session_stats, thresholds, gps_points=None):
    """Detect personal achievements from session data."""
    achievements = []
    max_speed = session_stats.get('max_speed')
    if max_speed is not None and max_speed > thresholds['speed_surge']:
        # Find the record with max speed for timestamp and GPS
        best_idx = None
        best_rec = None
        for i, r in enumerate(records):
            spd = r.get('speed')
            if spd is not None and abs(spd - max_speed) < 0.01:
                best_idx = i
                best_rec = r
                break
        entry = {
            'type': 'speed_demon',
            'value': round(max_speed * 3.6, 1),  # km/h
        }
        if best_rec:
            if best_rec.get('timestamp') is not None:
                entry['timestamp'] = best_rec['timestamp']
            # Find GPS from gps_points using record index
            if gps_points and best_idx is not None:
                gps_lookup = {pt[2]: (pt[0], pt[1]) for pt in gps_points}
                gps = gps_lookup.get(best_idx)
                if gps:
                    entry['lat'] = gps[0]
                    entry['lon'] = gps[1]
        achievements.append(entry)
    return achievements


def correlate_moments(file_moments, time_window):
    """Correlate moments across multiple files to find group moments."""
    # Gather all moments with file_id tag
    all_by_type = {}
    for file_id, moments in file_moments.items():
        for m in moments:
            t = m['type']
            if t not in all_by_type:
                all_by_type[t] = []
            all_by_type[t].append({**m, 'file_id': file_id})

    group_moments = []
    for mtype, items in all_by_type.items():
        items.sort(key=lambda x: x['timestamp'])
        consumed = set()

        for i, anchor in enumerate(items):
            if i in consumed:
                continue
            members = [anchor]
            member_files = {anchor['file_id']}

            for j in range(i + 1, len(items)):
                if j in consumed:
                    continue
                candidate = items[j]
                if candidate['timestamp'] - anchor['timestamp'] > time_window:
                    break
                if candidate['file_id'] not in member_files:
                    members.append(candidate)
                    member_files.add(candidate['file_id'])
                    consumed.add(j)

            if len(member_files) >= 2:
                consumed.add(i)
                avg_ts = sum(m['timestamp'] for m in members) / len(members)
                max_val = max(m['value'] for m in members)
                max_dur = max((m.get('duration') or 0) for m in members)
                lats = [m['lat'] for m in members if m.get('lat') is not None]
                lons = [m['lon'] for m in members if m.get('lon') is not None]
                group_moments.append({
                    'type': mtype,
                    'timestamp': round(avg_ts),
                    'value': max_val,
                    'duration': round(max_dur, 1) if max_dur > 0 else None,
                    'lat': round(sum(lats) / len(lats), 6) if lats else None,
                    'lon': round(sum(lons) / len(lons), 6) if lons else None,
                    'route_shape': members[0].get('route_shape'),
                    'members': [{
                        'file_id': m['file_id'],
                        'value': m['value'],
                        'timestamp': m['timestamp'],
                        'route_shape': m.get('route_shape'),
                    } for m in members],
                    'member_count': len(members),
                })

    group_moments.sort(key=lambda x: -x['member_count'])
    return group_moments


def correlate_achievements(file_achievements):
    """Correlate achievements across multiple files."""
    by_type = {}
    for file_id, achs in file_achievements.items():
        for a in achs:
            t = a['type']
            if t not in by_type:
                by_type[t] = {}
            # Keep highest value per file per type
            if file_id not in by_type[t] or a['value'] > by_type[t][file_id]['value']:
                by_type[t][file_id] = {**a, 'file_id': file_id}

    group_achievements = []
    for atype, file_map in by_type.items():
        if len(file_map) >= 2:
            members = list(file_map.values())
            group_achievements.append({
                'type': atype,
                'members': members,
                'member_count': len(members),
                'max_value': max(m['value'] for m in members),
            })
    return group_achievements


# ==============================================================================
# Report Export (self-contained HTML → Print to PDF)
# ==============================================================================

def _fmt_dur(sec):
    """Format seconds as H:MM:SS or M:SS."""
    if not sec:
        return '\u2014'
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def _fmt_dist_km(meters):
    return f'{meters / 1000:.2f}'


def _fmt_speed_kmh(ms):
    return f'{ms * 3.6:.1f}'


def _render_gps_svg(gps_points, width=540, height=340):
    """Render GPS route as an inline SVG element."""
    if not gps_points or len(gps_points) < 2:
        return '<div style="color:#94a3b8;text-align:center;padding:40px">No GPS data available</div>'

    # Downsample for SVG
    pts = gps_points
    if len(pts) > 500:
        step = len(pts) / 500
        pts = [pts[int(i * step)] for i in range(500)]
        if pts[-1] != gps_points[-1]:
            pts.append(gps_points[-1])

    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Padding
    lat_range = (max_lat - min_lat) or 0.001
    lon_range = (max_lon - min_lon) or 0.001
    pad = 0.08
    min_lat -= lat_range * pad
    max_lat += lat_range * pad
    min_lon -= lon_range * pad
    max_lon += lon_range * pad
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Aspect ratio correction
    cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
    effective_lon_range = lon_range * cos_lat

    # Fit into viewbox maintaining aspect
    if effective_lon_range / lat_range > width / height:
        view_w = width
        view_h = int(lat_range / (effective_lon_range / width))
        if view_h < 100:
            view_h = 100
    else:
        view_h = height
        view_w = int(effective_lon_range / (lat_range / height))
        if view_w < 100:
            view_w = 100

    def project(lat, lon):
        x = (lon - min_lon) / lon_range * view_w
        y = (1 - (lat - min_lat) / lat_range) * view_h
        return f'{x:.1f},{y:.1f}'

    polyline_pts = ' '.join(project(p[0], p[1]) for p in pts)
    start = project(pts[0][0], pts[0][1])
    end = project(pts[-1][0], pts[-1][1])
    sx, sy = start.split(',')
    ex, ey = end.split(',')

    # Distance scale
    mid_lat = (min_lat + max_lat) / 2
    km_per_deg_lon = 111.32 * math.cos(math.radians(mid_lat))
    map_km = lon_range * km_per_deg_lon
    scale_km = 1
    for s in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]:
        if s / map_km * view_w > 40:
            scale_km = s
            break
    scale_px = scale_km / map_km * view_w

    svg = f'''<svg viewBox="0 0 {view_w} {view_h}" width="{width}" height="{height}"
     xmlns="http://www.w3.org/2000/svg" style="background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0">
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.15"/>
    </filter>
  </defs>
  <polyline points="{polyline_pts}" fill="none" stroke="#2563eb" stroke-width="2.5"
    stroke-linecap="round" stroke-linejoin="round" filter="url(#shadow)" opacity="0.85"/>
  <circle cx="{sx}" cy="{sy}" r="6" fill="#22c55e" stroke="#fff" stroke-width="2"/>
  <circle cx="{ex}" cy="{ey}" r="6" fill="#ef4444" stroke="#fff" stroke-width="2"/>
  <g transform="translate(10,{view_h - 15})">
    <line x1="0" y1="0" x2="{scale_px:.1f}" y2="0" stroke="#64748b" stroke-width="2"/>
    <line x1="0" y1="-3" x2="0" y2="3" stroke="#64748b" stroke-width="1.5"/>
    <line x1="{scale_px:.1f}" y1="-3" x2="{scale_px:.1f}" y2="3" stroke="#64748b" stroke-width="1.5"/>
    <text x="{scale_px / 2:.1f}" y="-5" text-anchor="middle" font-size="9" fill="#64748b">{scale_km} km</text>
  </g>
  <g transform="translate({view_w - 22},12)">
    <text text-anchor="middle" font-size="9" fill="#22c55e" font-weight="600">S</text>
  </g>
  <g transform="translate({view_w - 10},12)">
    <text text-anchor="middle" font-size="9" fill="#ef4444" font-weight="600">F</text>
  </g>
</svg>'''
    return svg


def _render_profile_svg(elapsed, values, color, label, unit, width=540, height=150):
    """Render a timeseries profile as an inline SVG area chart."""
    if not elapsed or not values or len(values) < 2:
        return ''
    n = len(values)
    min_v = min(v for v in values if v is not None) if any(v is not None for v in values) else 0
    max_v = max(v for v in values if v is not None) if any(v is not None for v in values) else 1
    v_range = max_v - min_v or 1
    max_t = elapsed[-1] if elapsed[-1] > 0 else 1
    pad_l, pad_r, pad_t, pad_b = 45, 10, 20, 25
    cw = width - pad_l - pad_r
    ch = height - pad_t - pad_b

    def px(t, v):
        x = pad_l + (t / max_t) * cw
        y = pad_t + (1 - (v - min_v) / v_range) * ch
        return x, y

    # Build path
    path_pts = []
    area_pts = []
    for i in range(n):
        v = values[i] if values[i] is not None else min_v
        t = elapsed[i]
        x, y = px(t, v)
        path_pts.append(f'{x:.1f},{y:.1f}')
        area_pts.append(f'{x:.1f},{y:.1f}')

    # Area fill
    first_x = pad_l + (elapsed[0] / max_t) * cw
    last_x = pad_l + (elapsed[-1] / max_t) * cw
    bottom = pad_t + ch
    area_d = f'M{first_x:.1f},{bottom} L' + ' L'.join(area_pts) + f' L{last_x:.1f},{bottom} Z'
    line_d = 'M' + ' L'.join(path_pts)

    # Y axis ticks (5 ticks)
    y_ticks = ''
    for i in range(5):
        v = min_v + (v_range * i / 4)
        _, y = px(0, v)
        y_ticks += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="0.5"/>'
        y_ticks += f'<text x="{pad_l - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="8" fill="#94a3b8">{v:.0f}</text>'

    # X axis ticks (time)
    x_ticks = ''
    for i in range(5):
        t = max_t * i / 4
        x = pad_l + (t / max_t) * cw
        x_ticks += f'<text x="{x:.1f}" y="{height - 5}" text-anchor="middle" font-size="8" fill="#94a3b8">{_fmt_dur(t)}</text>'

    return f'''<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"
     xmlns="http://www.w3.org/2000/svg" style="margin-bottom:8px">
  <text x="{pad_l}" y="13" font-size="10" font-weight="600" fill="#475569">{label} ({unit})</text>
  {y_ticks}
  {x_ticks}
  <path d="{area_d}" fill="{color}" opacity="0.12"/>
  <path d="{line_d}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>
</svg>'''


def _render_zone_bars_svg(zone_data, colors, width=540):
    """Render zone distribution as horizontal SVG bars."""
    if not zone_data:
        return ''
    labels = zone_data['labels']
    pcts = zone_data['pct']
    secs = zone_data['zones']
    n = len(labels)
    bar_h = 22
    gap = 4
    label_w = 100
    time_w = 60
    bar_w = width - label_w - time_w - 20
    total_h = n * (bar_h + gap) + 4

    bars = ''
    for i in range(n):
        y = i * (bar_h + gap)
        fill_w = max(pcts[i] / 100 * bar_w, 2)
        bars += f'''<text x="{label_w - 4}" y="{y + 15}" text-anchor="end" font-size="9" fill="#64748b">{labels[i]}</text>
  <rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h}" rx="3" fill="#f1f5f9"/>
  <rect x="{label_w}" y="{y}" width="{fill_w:.1f}" height="{bar_h}" rx="3" fill="{colors[i]}"/>
  <text x="{label_w + fill_w - 4}" y="{y + 15}" text-anchor="end" font-size="8" fill="#fff" font-weight="600">{pcts[i]}%</text>
  <text x="{label_w + bar_w + 6}" y="{y + 15}" font-size="9" fill="#94a3b8" font-family="monospace">{_fmt_dur(secs[i])}</text>'''

    return f'''<svg viewBox="0 0 {width} {total_h}" width="{width}" height="{total_h}"
     xmlns="http://www.w3.org/2000/svg">{bars}</svg>'''


def _build_report_data(file_id, thresholds=None):
    """Assemble all data needed for a single-file report."""
    entry = uploaded_files[file_id]
    stats = entry.get('session_stats', {})
    records = entry.get('records', [])
    gps_points = entry.get('gps_points', [])
    laps = entry.get('laps', [])
    timeseries = _build_timeseries(records, gps_points, max_points=500)
    zones = _compute_zones(records, gps_points)
    if thresholds is None:
        thresholds = {
            'speed_surge': DEFAULT_SPEED_SURGE_THRESHOLD,
            'power_spike': DEFAULT_POWER_SPIKE_THRESHOLD,
            'sprint_power': DEFAULT_SPRINT_POWER_THRESHOLD,
            'sprint_accel': DEFAULT_SPRINT_ACCEL_THRESHOLD,
            'sprint_min_duration': DEFAULT_SPRINT_MIN_DURATION,
            'climb_gradient': DEFAULT_CLIMB_GRADIENT_THRESHOLD,
            'climb_min_duration': DEFAULT_CLIMB_MIN_DURATION,
            'climb_min_elevation_gain': DEFAULT_CLIMB_MIN_ELEVATION_GAIN,
        }
    moments = detect_moments(records, gps_points, thresholds)
    achievements = detect_achievements(records, stats, thresholds, gps_points)
    return {
        'file_id': file_id,
        'filename': entry['filename'],
        'original_start': entry.get('original_start'),
        'stats': stats,
        'gps_points': gps_points,
        'laps': laps,
        'timeseries': timeseries,
        'zones': zones,
        'moments': moments,
        'achievements': achievements,
    }


def _generate_report_html(data, auto_print=True):
    """Generate a self-contained HTML report for a single file."""
    filename = data['filename'].replace('.fit', '').replace('.FIT', '')
    start = data.get('original_start') or 'Unknown'
    stats = data['stats']
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # Stats grid
    stat_items = []
    def add_stat(label, val, unit=''):
        if val is not None:
            stat_items.append((label, val, unit))

    if stats.get('total_distance') is not None:
        add_stat('Distance', _fmt_dist_km(stats['total_distance']), 'km')
    if stats.get('total_timer_time') is not None:
        add_stat('Duration', _fmt_dur(stats['total_timer_time']))
    if stats.get('avg_speed') is not None:
        add_stat('Avg Speed', _fmt_speed_kmh(stats['avg_speed']), 'km/h')
    if stats.get('max_speed') is not None:
        add_stat('Max Speed', _fmt_speed_kmh(stats['max_speed']), 'km/h')
    if stats.get('total_ascent') is not None:
        add_stat('Ascent', str(stats['total_ascent']), 'm')
    if stats.get('total_descent') is not None:
        add_stat('Descent', str(stats['total_descent']), 'm')
    if stats.get('avg_heart_rate') is not None:
        add_stat('Avg HR', str(stats['avg_heart_rate']), 'bpm')
    if stats.get('max_heart_rate') is not None:
        add_stat('Max HR', str(stats['max_heart_rate']), 'bpm')
    if stats.get('avg_power') is not None:
        add_stat('Avg Power', str(stats['avg_power']), 'W')
    if stats.get('max_power') is not None:
        add_stat('Max Power', str(stats['max_power']), 'W')
    if stats.get('normalized_power') is not None:
        add_stat('NP', str(stats['normalized_power']), 'W')
    if stats.get('total_calories') is not None:
        add_stat('Calories', str(stats['total_calories']), 'kcal')
    if stats.get('avg_cadence') is not None:
        add_stat('Avg Cadence', str(stats['avg_cadence']), 'rpm')
    if stats.get('avg_temperature') is not None:
        add_stat('Avg Temp', str(stats['avg_temperature']), '\u00b0C')

    stats_html = ''
    for label, val, unit in stat_items:
        stats_html += f'<div class="stat-box"><div class="stat-val">{val}<span class="stat-unit">{unit}</span></div><div class="stat-lbl">{label}</div></div>'

    # Route SVG
    route_svg = _render_gps_svg(data['gps_points'])

    # Profile charts
    ts = data['timeseries']
    profiles_html = ''
    if ts.get('elevation'):
        elev_vals = ts['elevation']
        profiles_html += _render_profile_svg(ts.get('elapsed', []), elev_vals, '#64748b', 'Elevation', 'm')
    if ts.get('speed'):
        speed_vals = [v * 3.6 for v in ts['speed']]
        profiles_html += _render_profile_svg(ts.get('elapsed', []), speed_vals, '#2563eb', 'Speed', 'km/h')
    if ts.get('heart_rate'):
        profiles_html += _render_profile_svg(ts.get('elapsed', []), ts['heart_rate'], '#dc2626', 'Heart Rate', 'bpm')
    if ts.get('power'):
        profiles_html += _render_profile_svg(ts.get('elapsed', []), ts['power'], '#ea580c', 'Power', 'W')

    # Laps table
    laps_html = ''
    laps = data.get('laps', [])
    if laps:
        laps_html = '<div class="section"><div class="section-title">Lap Splits</div><table class="lap-tbl"><thead><tr><th>#</th>'
        col_defs = [
            ('total_timer_time', 'Duration', lambda v: _fmt_dur(v)),
            ('total_distance', 'Distance', lambda v: _fmt_dist_km(v) + ' km'),
            ('avg_speed', 'Avg Speed', lambda v: _fmt_speed_kmh(v) + ' km/h'),
            ('avg_heart_rate', 'Avg HR', lambda v: f'{v} bpm'),
            ('max_heart_rate', 'Max HR', lambda v: f'{v} bpm'),
            ('avg_power', 'Avg Power', lambda v: f'{v} W'),
            ('avg_cadence', 'Cadence', lambda v: f'{v} rpm'),
            ('total_ascent', 'Ascent', lambda v: f'{v} m'),
        ]
        active_cols = [(k, l, f) for k, l, f in col_defs if any(lap.get(k) is not None for lap in laps)]
        for _, label, _ in active_cols:
            laps_html += f'<th>{label}</th>'
        laps_html += '</tr></thead><tbody>'
        for i, lap in enumerate(laps):
            laps_html += f'<tr><td>{i + 1}</td>'
            for key, _, fmt in active_cols:
                v = lap.get(key)
                laps_html += f'<td>{fmt(v) if v is not None else chr(8212)}</td>'
            laps_html += '</tr>'
        laps_html += '</tbody></table></div>'

    # Zones
    zones_html = ''
    zones = data.get('zones', {})
    hr_colors = ['#3b82f6', '#22c55e', '#eab308', '#f97316', '#ef4444']
    pw_colors = ['#93c5fd', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e3a8a']
    if zones.get('hr') or zones.get('power'):
        zones_html = '<div class="section"><div class="section-title">Zone Analysis</div>'
        if zones.get('hr'):
            zones_html += f'<div class="zone-heading">Heart Rate Zones (Max HR: {zones["hr"]["max_hr"]} bpm)</div>'
            zones_html += _render_zone_bars_svg(zones['hr'], hr_colors)
        if zones.get('power'):
            zones_html += f'<div class="zone-heading" style="margin-top:16px">Power Zones (FTP: {zones["power"]["ftp"]} W)</div>'
            zones_html += _render_zone_bars_svg(zones['power'], pw_colors)
        zones_html += '</div>'

    # Moments
    moments_html = ''
    moments = data.get('moments', [])
    achievements = data.get('achievements', [])
    moment_labels = {'speed_surge': 'Speed Surge', 'power_spike': 'Power Spike', 'sprint': 'Sprint', 'climb': 'Climb'}
    moment_colors = {'speed_surge': '#e6198a', 'power_spike': '#d97706', 'sprint': '#dc2626', 'climb': '#15803d'}
    if moments or achievements:
        by_type = {}
        for m in moments:
            by_type.setdefault(m['type'], []).append(m)
        moments_html = '<div class="section"><div class="section-title">Detected Moments</div><div class="moment-badges-r">'
        for mtype, items in by_type.items():
            c = moment_colors.get(mtype, '#666')
            moments_html += f'<span class="mbadge" style="background:{c}15;color:{c};border:1px solid {c}40">{len(items)} {moment_labels.get(mtype, mtype)}</span>'
        for a in achievements:
            moments_html += f'<span class="mbadge" style="background:#fef3c7;color:#92400e;border:1px solid #fde68a">\U0001f3c6 {a["value"]} km/h</span>'
        moments_html += '</div>'

        for mtype, items in by_type.items():
            c = moment_colors.get(mtype, '#666')
            moments_html += f'<div class="moment-group"><div class="moment-group-title" style="color:{c}">{moment_labels.get(mtype, mtype)} ({len(items)})</div>'
            for m in items[:10]:  # Cap display
                detail = ''
                if mtype == 'speed_surge':
                    detail = f'{m["value"] * 3.6:.1f} km/h'
                elif mtype == 'power_spike':
                    detail = f'{m["value"]:.0f} W'
                elif mtype == 'sprint':
                    detail = f'{m["value"]:.0f} W peak, {m["duration"]}s'
                elif mtype == 'climb':
                    detail = f'+{m["value"]} m, {_fmt_dur(m.get("duration", 0))}'
                moments_html += f'<div class="moment-row"><span>{detail}</span></div>'
            if len(items) > 10:
                moments_html += f'<div class="moment-row" style="color:#94a3b8">+{len(items) - 10} more</div>'
            moments_html += '</div>'
        moments_html += '</div>'

    print_script = '<script>window.onload=function(){window.print()}</script>' if auto_print else ''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Activity Report \u2014 {filename}</title>
<style>
  @page {{ size: A4; margin: 12mm 15mm; }}
  @media print {{
    body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .no-print {{ display: none !important; }}
    .page-break {{ page-break-before: always; }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    color: #1e293b; background: #fff; font-size: 13px; line-height: 1.5;
  }}
  .page {{ max-width: 680px; margin: 0 auto; padding: 20px; }}
  /* Header */
  .header {{
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%);
    color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px;
    position: relative; overflow: hidden;
  }}
  .header::after {{
    content: ''; position: absolute; top: -50%; right: -20%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .header h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 2px; letter-spacing: -0.02em; }}
  .header .meta {{ font-size: 12px; opacity: 0.8; }}
  .header .meta span {{ margin-right: 16px; }}
  /* Stats */
  .stats-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 24px;
  }}
  .stat-box {{
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 10px; text-align: center;
  }}
  .stat-val {{ font-size: 18px; font-weight: 700; color: #1e293b; line-height: 1.2; }}
  .stat-unit {{ font-size: 10px; font-weight: 400; color: #64748b; margin-left: 2px; }}
  .stat-lbl {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-top: 2px; }}
  /* Sections */
  .section {{
    margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid #f1f5f9;
  }}
  .section:last-child {{ border-bottom: none; }}
  .section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: #64748b; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0;
  }}
  /* Route */
  .route-container {{ text-align: center; margin-bottom: 8px; }}
  /* Laps */
  .lap-tbl {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  .lap-tbl th {{
    text-align: left; padding: 6px 8px; font-weight: 600; font-size: 9px;
    text-transform: uppercase; letter-spacing: 0.04em; color: #64748b;
    border-bottom: 2px solid #e2e8f0; background: #f8fafc;
  }}
  .lap-tbl td {{
    padding: 6px 8px; border-bottom: 1px solid #f1f5f9;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px;
  }}
  .lap-tbl tr:nth-child(even) td {{ background: #fafbfc; }}
  /* Zones */
  .zone-heading {{ font-size: 11px; font-weight: 600; color: #475569; margin-bottom: 8px; }}
  /* Moments */
  .moment-badges-r {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }}
  .mbadge {{
    display: inline-flex; align-items: center; gap: 3px;
    padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
  }}
  .moment-group {{ margin-bottom: 10px; }}
  .moment-group-title {{ font-size: 11px; font-weight: 600; margin-bottom: 4px; }}
  .moment-row {{
    font-size: 11px; padding: 2px 8px; border-left: 3px solid #e2e8f0;
    margin-bottom: 2px; font-family: 'SF Mono', 'Fira Code', monospace;
  }}
  /* Footer */
  .footer {{
    text-align: center; font-size: 10px; color: #94a3b8;
    padding: 16px 0; border-top: 1px solid #f1f5f9; margin-top: 12px;
  }}
  /* Print button */
  .print-bar {{
    position: fixed; top: 0; left: 0; right: 0; background: #1e293b; color: white;
    padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
    z-index: 1000; font-size: 13px;
  }}
  .print-bar button {{
    background: #2563eb; color: white; border: none; padding: 8px 20px;
    border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer;
  }}
  .print-bar button:hover {{ background: #1d4ed8; }}
  @media print {{ .print-bar {{ display: none; }} .page {{ padding-top: 0; }} }}
  @media screen {{ .page {{ padding-top: 56px; }} }}
</style>
</head>
<body>
<div class="print-bar no-print">
  <span>Activity Report &mdash; {filename}</span>
  <button onclick="window.print()">Save as PDF</button>
</div>
<div class="page">
  <div class="header">
    <h1>{filename}</h1>
    <div class="meta">
      <span>Start: {start} UTC</span>
      <span>Report generated: {now_str}</span>
    </div>
  </div>

  <div class="stats-grid">{stats_html}</div>

  <div class="section">
    <div class="section-title">Route</div>
    <div class="route-container">{route_svg}</div>
  </div>

  {('<div class="section"><div class="section-title">Performance Profiles</div>' + profiles_html + '</div>') if profiles_html else ''}

  {laps_html}

  {zones_html}

  {moments_html}

  <div class="footer">
    Generated by FIT Toolkit &bull; {now_str}
  </div>
</div>
{print_script}
</body>
</html>'''


def _generate_group_report_html(reports, auto_print=True):
    """Generate a self-contained HTML report for multiple files."""
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    n = len(reports)

    # File list
    file_list_html = ''
    for r in reports:
        fname = r['filename'].replace('.fit', '').replace('.FIT', '')
        start = r.get('original_start') or 'Unknown'
        dist = _fmt_dist_km(r['stats']['total_distance']) + ' km' if r['stats'].get('total_distance') else ''
        dur = _fmt_dur(r['stats'].get('total_timer_time'))
        file_list_html += f'<div class="file-entry"><strong>{fname}</strong><span class="file-detail">{start} UTC &bull; {dist} &bull; {dur}</span></div>'

    # Comparison table
    comp_html = '<table class="lap-tbl"><thead><tr><th>Metric</th>'
    for r in reports:
        comp_html += f'<th>{r["filename"].replace(".fit", "").replace(".FIT", "")}</th>'
    comp_html += '</tr></thead><tbody>'
    metrics = [
        ('Distance', lambda s: _fmt_dist_km(s['total_distance']) + ' km' if s.get('total_distance') else '\u2014'),
        ('Duration', lambda s: _fmt_dur(s.get('total_timer_time'))),
        ('Avg Speed', lambda s: _fmt_speed_kmh(s['avg_speed']) + ' km/h' if s.get('avg_speed') else '\u2014'),
        ('Max Speed', lambda s: _fmt_speed_kmh(s['max_speed']) + ' km/h' if s.get('max_speed') else '\u2014'),
        ('Ascent', lambda s: f'{s["total_ascent"]} m' if s.get('total_ascent') else '\u2014'),
        ('Avg HR', lambda s: f'{s["avg_heart_rate"]} bpm' if s.get('avg_heart_rate') else '\u2014'),
        ('Max HR', lambda s: f'{s["max_heart_rate"]} bpm' if s.get('max_heart_rate') else '\u2014'),
        ('Avg Power', lambda s: f'{s["avg_power"]} W' if s.get('avg_power') else '\u2014'),
        ('Calories', lambda s: f'{s["total_calories"]} kcal' if s.get('total_calories') else '\u2014'),
    ]
    for label, fn in metrics:
        vals = [fn(r['stats']) for r in reports]
        if all(v == '\u2014' for v in vals):
            continue
        comp_html += f'<tr><td style="font-weight:600">{label}</td>'
        for v in vals:
            comp_html += f'<td>{v}</td>'
        comp_html += '</tr>'
    comp_html += '</tbody></table>'

    # Combined route SVG
    combined_route = ''
    overlay_colors = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
    all_gps = []
    for r in reports:
        all_gps.extend(r.get('gps_points', []))
    if len(all_gps) >= 2:
        # Compute global bounds
        all_lats = [p[0] for p in all_gps]
        all_lons = [p[1] for p in all_gps]
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        lat_range = (max_lat - min_lat) or 0.001
        lon_range = (max_lon - min_lon) or 0.001
        pad = 0.08
        min_lat -= lat_range * pad
        max_lat += lat_range * pad
        min_lon -= lon_range * pad
        max_lon += lon_range * pad
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
        w, h = 540, 340
        effective_lon_range = lon_range * cos_lat
        if effective_lon_range / lat_range > w / h:
            vw = w
            vh = max(int(lat_range / (effective_lon_range / w)), 100)
        else:
            vh = h
            vw = max(int(effective_lon_range / (lat_range / h)), 100)

        def proj(lat, lon):
            x = (lon - min_lon) / lon_range * vw
            y = (1 - (lat - min_lat) / lat_range) * vh
            return f'{x:.1f},{y:.1f}'

        polylines = ''
        legend_items = ''
        for i, r in enumerate(reports):
            pts = r.get('gps_points', [])
            if len(pts) < 2:
                continue
            if len(pts) > 500:
                step = len(pts) / 500
                pts = [pts[int(j * step)] for j in range(500)]
            color = overlay_colors[i % len(overlay_colors)]
            pts_str = ' '.join(proj(p[0], p[1]) for p in pts)
            polylines += f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>'
            fname = r['filename'].replace('.fit', '').replace('.FIT', '')
            legend_items += f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px"><span style="width:12px;height:3px;background:{color};border-radius:2px;display:inline-block"></span>{fname}</span>'

        combined_route = f'''<div class="section">
  <div class="section-title">Routes</div>
  <div class="route-container">
    <svg viewBox="0 0 {vw} {vh}" width="540" height="{int(540 * vh / vw)}"
      xmlns="http://www.w3.org/2000/svg" style="background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0">
      {polylines}
    </svg>
  </div>
  <div style="font-size:10px;color:#64748b;margin-top:6px;text-align:center">{legend_items}</div>
</div>'''

    # Group moments
    file_moments = {}
    file_achievements = {}
    for r in reports:
        fid = r['file_id']
        file_moments[fid] = r.get('moments', [])
        file_achievements[fid] = r.get('achievements', [])
    group_moms = correlate_moments(file_moments, DEFAULT_GROUP_TIME_WINDOW)
    group_achs = correlate_achievements(file_achievements)

    group_html = ''
    moment_labels = {'speed_surge': 'Speed Surge', 'power_spike': 'Power Spike', 'sprint': 'Sprint', 'climb': 'Climb', 'speed_demon': 'Speed Demon'}
    moment_colors_map = {'speed_surge': '#e6198a', 'power_spike': '#d97706', 'sprint': '#dc2626', 'climb': '#15803d'}
    if group_moms or group_achs:
        group_html = '<div class="section"><div class="section-title">Group Moments</div>'
        for gm in group_moms:
            c = moment_colors_map.get(gm['type'], '#666')
            val_str = ''
            if gm['type'] == 'speed_surge':
                val_str = f'{gm["value"] * 3.6:.1f} km/h'
            elif gm['type'] in ('power_spike', 'sprint'):
                val_str = f'{gm["value"]:.0f} W'
            elif gm['type'] == 'climb':
                val_str = f'+{gm["value"]} m'
            members_str = ', '.join(
                (next((r['filename'].replace('.fit','').replace('.FIT','') for r in reports if r['file_id'] == mem['file_id']), mem['file_id']))
                for mem in gm['members']
            )
            group_html += f'<div style="padding:8px;background:#f8fafc;border-radius:6px;margin-bottom:6px;border-left:3px solid {c}">'
            group_html += f'<div style="font-weight:600;font-size:12px">{moment_labels.get(gm["type"], gm["type"])} &bull; {gm["member_count"]} riders &bull; peak {val_str}</div>'
            group_html += f'<div style="font-size:10px;color:#64748b">{members_str}</div></div>'
        if group_achs:
            for ga in group_achs:
                group_html += f'<div style="padding:8px;background:#fef3c7;border-radius:6px;margin-bottom:6px">'
                group_html += f'<div style="font-weight:600;font-size:12px">\U0001f3c6 {moment_labels.get(ga["type"], ga["type"])} &bull; {ga["member_count"]} riders</div></div>'
        group_html += '</div>'

    print_script = '<script>window.onload=function(){window.print()}</script>' if auto_print else ''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Group Activity Report</title>
<style>
  @page {{ size: A4; margin: 12mm 15mm; }}
  @media print {{
    body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .no-print {{ display: none !important; }}
    .page-break {{ page-break-before: always; }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b; background: #fff; font-size: 13px; line-height: 1.5;
  }}
  .page {{ max-width: 680px; margin: 0 auto; padding: 20px; }}
  .header {{
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%);
    color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px;
    position: relative; overflow: hidden;
  }}
  .header::after {{
    content: ''; position: absolute; top: -50%; right: -20%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .header h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 2px; }}
  .header .meta {{ font-size: 12px; opacity: 0.8; }}
  .section {{ margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid #f1f5f9; }}
  .section:last-child {{ border-bottom: none; }}
  .section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: #64748b; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0;
  }}
  .file-entry {{
    padding: 8px 12px; background: #f8fafc; border-radius: 6px;
    margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;
  }}
  .file-detail {{ font-size: 11px; color: #64748b; }}
  .lap-tbl {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  .lap-tbl th {{
    text-align: left; padding: 6px 8px; font-weight: 600; font-size: 9px;
    text-transform: uppercase; letter-spacing: 0.04em; color: #64748b;
    border-bottom: 2px solid #e2e8f0; background: #f8fafc;
  }}
  .lap-tbl td {{
    padding: 6px 8px; border-bottom: 1px solid #f1f5f9;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px;
  }}
  .lap-tbl tr:nth-child(even) td {{ background: #fafbfc; }}
  .route-container {{ text-align: center; }}
  .footer {{ text-align: center; font-size: 10px; color: #94a3b8; padding: 16px 0; border-top: 1px solid #f1f5f9; margin-top: 12px; }}
  .print-bar {{
    position: fixed; top: 0; left: 0; right: 0; background: #1e293b; color: white;
    padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
    z-index: 1000; font-size: 13px;
  }}
  .print-bar button {{
    background: #2563eb; color: white; border: none; padding: 8px 20px;
    border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer;
  }}
  .print-bar button:hover {{ background: #1d4ed8; }}
  @media print {{ .print-bar {{ display: none; }} .page {{ padding-top: 0; }} }}
  @media screen {{ .page {{ padding-top: 56px; }} }}
</style>
</head>
<body>
<div class="print-bar no-print">
  <span>Group Activity Report &mdash; {n} files</span>
  <button onclick="window.print()">Save as PDF</button>
</div>
<div class="page">
  <div class="header">
    <h1>Group Activity Report</h1>
    <div class="meta">
      <span>{n} activities</span>
      <span>&bull; Generated: {now_str}</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Activities</div>
    {file_list_html}
  </div>

  {combined_route}

  <div class="section">
    <div class="section-title">Comparison</div>
    {comp_html}
  </div>

  {group_html}

  <div class="footer">
    Generated by FIT Toolkit &bull; {now_str}
  </div>
</div>
{print_script}
</body>
</html>'''


# ==============================================================================
# Web Server (stdlib only — no Flask, no pip install)
# ==============================================================================

uploaded_files = {}
adjusted_files = {}


def parse_multipart(handler):
    """Parse a multipart/form-data upload and return (filename, file_bytes)."""
    content_type = handler.headers.get('Content-Type', '')
    if 'boundary=' not in content_type:
        return None, None

    boundary = content_type.split('boundary=')[1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    body = handler.rfile.read(int(handler.headers['Content-Length']))
    boundary_bytes = ('--' + boundary).encode()
    parts = body.split(boundary_bytes)

    for part in parts:
        if b'Content-Disposition' not in part:
            continue
        header_end = part.find(b'\r\n\r\n')
        if header_end < 0:
            continue
        header_section = part[:header_end].decode('utf-8', errors='replace')
        file_data = part[header_end + 4:]
        if file_data.endswith(b'\r\n'):
            file_data = file_data[:-2]

        if 'filename="' in header_section:
            fname_start = header_section.index('filename="') + 10
            fname_end = header_section.index('"', fname_start)
            filename = header_section[fname_start:fname_end]
            return filename, file_data

    return None, None


def _build_timeseries(records, gps_points=None, max_points=1000):
    """Build columnar time-series data from record dicts, with downsampling.
    gps_points: list of (lat, lon, rec_idx) tuples for map-chart linking."""
    if not records:
        return {'count': 0}
    # Build a rec_idx → (lat, lon) lookup from GPS points
    gps_lookup = {}
    if gps_points:
        for pt in gps_points:
            gps_lookup[pt[2]] = (pt[0], pt[1])
    # Downsample if needed, tracking original indices
    indices = list(range(len(records)))
    if len(records) > max_points:
        step = len(records) / max_points
        indices = [int(i * step) for i in range(max_points)]
        if indices[-1] != len(records) - 1:
            indices.append(len(records) - 1)
    pts = [records[i] for i in indices]
    # Compute elapsed time from first timestamp
    first_ts = None
    for r in pts:
        if 'timestamp' in r:
            first_ts = r['timestamp']
            break
    # Build columnar arrays
    result = {'count': len(pts)}
    keys = ['elapsed', 'elevation', 'heart_rate', 'speed', 'cadence', 'power', 'temperature']
    arrays = {k: [] for k in keys}
    lat_arr = []
    lon_arr = []
    for idx_pos, orig_idx in enumerate(indices):
        r = pts[idx_pos]
        ts = r.get('timestamp')
        arrays['elapsed'].append(round(ts - first_ts, 1) if ts and first_ts else 0)
        arrays['elevation'].append(r.get('altitude'))
        arrays['heart_rate'].append(r.get('heart_rate'))
        arrays['speed'].append(r.get('speed'))
        arrays['cadence'].append(r.get('cadence'))
        arrays['power'].append(r.get('power'))
        arrays['temperature'].append(r.get('temperature'))
        # GPS lookup for this record index
        gps = gps_lookup.get(orig_idx)
        lat_arr.append(gps[0] if gps else None)
        lon_arr.append(gps[1] if gps else None)
    # Only include arrays that have at least some non-null values
    for k in keys:
        non_null = [v for v in arrays[k] if v is not None]
        if non_null:
            result[k] = [v if v is not None else 0 for v in arrays[k]]
    # Include GPS arrays if any valid points
    if any(v is not None for v in lat_arr):
        result['lat'] = lat_arr
        result['lon'] = lon_arr
    return result


def _compute_zones(records, gps_points=None, max_hr=190, ftp=200):
    """Compute HR and power zone distributions from record data."""
    hr_zones = [0] * 5   # Z1-Z5 in seconds
    power_zones = [0] * 6  # Z1-Z6 in seconds
    hr_thresholds = [max_hr * p for p in [0.6, 0.7, 0.8, 0.9]]
    power_thresholds = [ftp * p for p in [0.55, 0.75, 0.90, 1.05, 1.20]]
    has_hr = False
    has_power = False
    for r in records:
        hr = r.get('heart_rate')
        pw = r.get('power')
        if hr is not None and hr > 0:
            has_hr = True
            if hr < hr_thresholds[0]:
                hr_zones[0] += 1
            elif hr < hr_thresholds[1]:
                hr_zones[1] += 1
            elif hr < hr_thresholds[2]:
                hr_zones[2] += 1
            elif hr < hr_thresholds[3]:
                hr_zones[3] += 1
            else:
                hr_zones[4] += 1
        if pw is not None and pw > 0:
            has_power = True
            if pw < power_thresholds[0]:
                power_zones[0] += 1
            elif pw < power_thresholds[1]:
                power_zones[1] += 1
            elif pw < power_thresholds[2]:
                power_zones[2] += 1
            elif pw < power_thresholds[3]:
                power_zones[3] += 1
            elif pw < power_thresholds[4]:
                power_zones[4] += 1
            else:
                power_zones[5] += 1
    result = {}
    if has_hr:
        total = sum(hr_zones) or 1
        result['hr'] = {
            'zones': hr_zones,
            'pct': [round(z / total * 100, 1) for z in hr_zones],
            'labels': ['Z1 Recovery', 'Z2 Endurance', 'Z3 Tempo', 'Z4 Threshold', 'Z5 VO2max'],
            'max_hr': max_hr,
        }
    if has_power:
        total = sum(power_zones) or 1
        result['power'] = {
            'zones': power_zones,
            'pct': [round(z / total * 100, 1) for z in power_zones],
            'labels': ['Z1 Recovery', 'Z2 Endurance', 'Z3 Tempo', 'Z4 Threshold', 'Z5 VO2max', 'Z6 Anaerobic'],
            'ftp': ftp,
        }
    return result


class FITHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_bytes, filename, content_type='application/octet-stream'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Length', str(len(file_bytes)))
        self.end_headers()
        self.wfile.write(file_bytes)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/':
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path.startswith('/download/'):
            result_id = path.split('/download/')[1]
            if result_id in adjusted_files:
                info = adjusted_files[result_id]
                self._send_file(info['bytes'], info['filename'])
            else:
                self.send_error(404, 'File not found')

        elif path.startswith('/gps/'):
            file_id = path.split('/gps/')[1]
            if file_id in uploaded_files:
                pts = uploaded_files[file_id].get('gps_points', [])
                # Downsample if too many points (keep route shape, reduce payload)
                if len(pts) > 2000:
                    step = len(pts) / 2000
                    pts = [pts[int(i * step)] for i in range(2000)] + [pts[-1]]
                # Return [lat, lon] only (strip rec_idx for map rendering)
                self._send_json({'points': [[p[0], p[1]] for p in pts]})
            else:
                self._send_json({'error': 'File not found'}, 404)

        elif path.startswith('/stats/'):
            file_id = path.split('/stats/')[1]
            if file_id in uploaded_files:
                self._send_json(uploaded_files[file_id].get('session_stats', {}))
            else:
                self._send_json({'error': 'File not found'}, 404)

        elif path.startswith('/timeseries/'):
            file_id = path.split('/timeseries/')[1]
            if file_id in uploaded_files:
                records = uploaded_files[file_id].get('records', [])
                gps_pts = uploaded_files[file_id].get('gps_points', [])
                self._send_json(_build_timeseries(records, gps_pts))
            else:
                self._send_json({'error': 'File not found'}, 404)

        elif path.startswith('/laps/'):
            file_id = path.split('/laps/')[1]
            if file_id in uploaded_files:
                self._send_json(uploaded_files[file_id].get('laps', []))
            else:
                self._send_json({'error': 'File not found'}, 404)

        elif path.startswith('/zones/'):
            file_id = path.split('/zones/')[1]
            qs = urllib.parse.parse_qs(parsed.query)
            max_hr = int(qs.get('max_hr', [190])[0])
            ftp = int(qs.get('ftp', [200])[0])
            if file_id in uploaded_files:
                records = uploaded_files[file_id].get('records', [])
                self._send_json(_compute_zones(records, max_hr=max_hr, ftp=ftp))
            else:
                self._send_json({'error': 'File not found'}, 404)

        elif path == '/timeseries-multi':
            qs = urllib.parse.parse_qs(parsed.query)
            ids = qs.get('ids', [])
            result = {}
            for fid in ids:
                if fid in uploaded_files:
                    records = uploaded_files[fid].get('records', [])
                    gps_pts = uploaded_files[fid].get('gps_points', [])
                    result[fid] = _build_timeseries(records, gps_pts)
            self._send_json(result)

        elif path == '/gps-multi':
            qs = urllib.parse.parse_qs(parsed.query)
            ids = qs.get('ids', [])
            result = {}
            for fid in ids:
                if fid in uploaded_files:
                    pts = uploaded_files[fid].get('gps_points', [])
                    if len(pts) > 2000:
                        step = len(pts) / 2000
                        pts = [pts[int(i * step)] for i in range(2000)] + [pts[-1]]
                    result[fid] = {'points': [[p[0], p[1]] for p in pts]}
            self._send_json(result)

        elif path == '/similarity':
            qs = urllib.parse.parse_qs(parsed.query)
            id_a = qs.get('a', [None])[0]
            id_b = qs.get('b', [None])[0]
            if not id_a or not id_b:
                self._send_json({'error': 'Need ?a=fileId&b=fileId'}, 400)
            elif id_a not in uploaded_files or id_b not in uploaded_files:
                self._send_json({'error': 'File not found'}, 404)
            else:
                gps_a = uploaded_files[id_a].get('gps_points', [])
                gps_b = uploaded_files[id_b].get('gps_points', [])
                result = compute_route_similarity(gps_a, gps_b)
                if result is None:
                    self._send_json({'error': 'Insufficient GPS data'}, 400)
                else:
                    result['file_a'] = uploaded_files[id_a]['filename']
                    result['file_b'] = uploaded_files[id_b]['filename']
                    self._send_json(result)

        elif path == '/download-zip':
            qs = urllib.parse.parse_qs(parsed.query)
            ids = qs.get('ids', [])
            if not ids:
                self.send_error(400, 'No file IDs')
                return
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for rid in ids:
                    if rid in adjusted_files:
                        info = adjusted_files[rid]
                        zf.writestr(info['filename'], info['bytes'])
            self._send_file(buf.getvalue(), 'adjusted_fit_files.zip', 'application/zip')

        elif path.startswith('/report/'):
            file_id = path.split('/report/')[1]
            qs = urllib.parse.parse_qs(parsed.query)
            auto_print = qs.get('print', ['1'])[0] != '0'
            if file_id not in uploaded_files:
                self.send_error(404, 'File not found')
            else:
                thresholds = _parse_thresholds(qs)
                report_data = _build_report_data(file_id, thresholds)
                html = _generate_report_html(report_data, auto_print=auto_print)
                body = html.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        elif path == '/report-group':
            qs = urllib.parse.parse_qs(parsed.query)
            ids = qs.get('ids', [])
            auto_print = qs.get('print', ['1'])[0] != '0'
            if not ids:
                self.send_error(400, 'No file IDs')
            else:
                thresholds = _parse_thresholds(qs)
                reports = []
                for fid in ids:
                    if fid in uploaded_files:
                        reports.append(_build_report_data(fid, thresholds))
                if not reports:
                    self.send_error(404, 'No valid files')
                else:
                    html = _generate_group_report_html(reports, auto_print=auto_print)
                    body = html.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        elif path.startswith('/moments/'):
            file_id = path.split('/moments/')[1]
            qs = urllib.parse.parse_qs(parsed.query)
            if file_id not in uploaded_files:
                self._send_json({'error': 'File not found'}, 404)
            else:
                thresholds = _parse_thresholds(qs)
                records = uploaded_files[file_id].get('records', [])
                gps_pts = uploaded_files[file_id].get('gps_points', [])
                session_stats = uploaded_files[file_id].get('session_stats', {})
                moments = detect_moments(records, gps_pts, thresholds)
                achievements = detect_achievements(records, session_stats, thresholds, gps_pts)
                self._send_json({'moments': moments, 'achievements': achievements})

        elif path == '/group-moments':
            qs = urllib.parse.parse_qs(parsed.query)
            ids = qs.get('ids', [])
            if len(ids) < 2:
                self._send_json({'error': 'Need at least 2 file IDs'}, 400)
            else:
                thresholds = _parse_thresholds(qs)
                time_window = float(qs.get('time_window', [DEFAULT_GROUP_TIME_WINDOW])[0])
                file_moments = {}
                file_achievements = {}
                individual = {}
                for fid in ids:
                    if fid not in uploaded_files:
                        continue
                    records = uploaded_files[fid].get('records', [])
                    gps_pts = uploaded_files[fid].get('gps_points', [])
                    session_stats = uploaded_files[fid].get('session_stats', {})
                    moments = detect_moments(records, gps_pts, thresholds)
                    achievements = detect_achievements(records, session_stats, thresholds, gps_pts)
                    file_moments[fid] = moments
                    file_achievements[fid] = achievements
                    individual[fid] = {'moments': moments, 'achievements': achievements}
                group_moments = correlate_moments(file_moments, time_window)
                group_achievements = correlate_achievements(file_achievements)
                self._send_json({
                    'group_moments': group_moments,
                    'group_achievements': group_achievements,
                    'individual': individual,
                })

        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path

        if path == '/upload':
            filename, file_bytes = parse_multipart(self)
            if not filename or not file_bytes:
                self._send_json({'error': 'No file provided'}, 400)
                return

            file_id = uuid.uuid4().hex[:8]
            try:
                fit = FITFile(file_bytes)
                start_dt = fit.get_start_datetime()
                start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S') if start_dt else None
                gps_count = len(fit.gps_points)
            except Exception as e:
                self._send_json({'error': f'Invalid FIT file: {e}'}, 400)
                return

            size_kb = len(file_bytes) / 1024
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"

            has_stats = bool(fit.session_stats or fit.records)
            has_laps = bool(fit.laps)
            uploaded_files[file_id] = {
                'filename': filename,
                'bytes': file_bytes,
                'original_start': start_str,
                'gps_points': fit.gps_points,
                'session_stats': fit.session_stats,
                'records': fit.records,
                'laps': fit.laps,
            }

            self._send_json({
                'id': file_id,
                'filename': filename,
                'original_start': start_str,
                'size_str': size_str,
                'gps_count': gps_count,
                'has_stats': has_stats,
                'has_laps': has_laps,
                'lap_count': len(fit.laps),
            })

        elif path == '/adjust':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            file_id = data.get('file_id')
            new_start_str = data.get('new_start')

            if file_id not in uploaded_files:
                self._send_json({'error': 'File not found. Please re-upload.'}, 404)
                return

            try:
                new_start_dt = datetime.strptime(new_start_str, '%Y-%m-%dT%H:%M:%S')
                new_start_dt = new_start_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                self._send_json({'error': f'Invalid datetime: {new_start_str}'}, 400)
                return

            try:
                result = process_fit_bytes(uploaded_files[file_id]['bytes'], new_start_dt)
            except Exception as e:
                self._send_json({'error': str(e)}, 400)
                return

            result_id = uuid.uuid4().hex[:8]
            orig_stem = Path(uploaded_files[file_id]['filename']).stem
            adjusted_files[result_id] = {
                'filename': f'{orig_stem}_adjusted.fit',
                'bytes': result['output_bytes'],
            }

            self._send_json({
                'result_id': result_id,
                'original_start': result['original_start'].strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                'new_start': result['new_start'].strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                'offset_seconds': result['offset_seconds'],
                'timestamps_modified': result['timestamps_modified'],
            })

        else:
            self.send_error(404)

    def do_DELETE(self):
        if self.path.startswith('/remove/'):
            file_id = self.path.split('/remove/')[1]
            uploaded_files.pop(file_id, None)
            self._send_json({'ok': True})
        else:
            self.send_error(404)


# ==============================================================================
# HTML Page
# ==============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FIT Toolkit</title>
<style>
  :root {
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --danger: #dc2626;
    --success: #16a34a;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 24px;
  }
  .container { max-width: 720px; margin: 0 auto; }
  h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: var(--text-muted); margin-bottom: 24px; font-size: 0.95rem; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; margin-bottom: 16px;
  }
  .card-title {
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 12px;
  }
  .drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 32px; text-align: center; cursor: pointer;
    transition: all 0.2s; background: var(--bg);
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--primary); background: #eff6ff;
  }
  .drop-zone-icon { font-size: 2rem; margin-bottom: 8px; }
  .drop-zone-text { color: var(--text-muted); }
  .drop-zone-text strong { color: var(--primary); }
  input[type="file"] { display: none; }
  .file-list { list-style: none; }
  .file-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 12px; border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 8px; background: var(--bg); font-size: 0.9rem;
  }
  .file-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1; }
  .file-name { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-meta { color: var(--text-muted); font-size: 0.8rem; }
  .file-actions { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
  .file-remove {
    background: none; border: none; color: var(--danger); cursor: pointer;
    font-size: 1.2rem; padding: 4px 8px; border-radius: 4px; flex-shrink: 0;
  }
  .file-remove:hover { background: #fef2f2; }
  .file-overlay-cb { width: 16px; height: 16px; cursor: pointer; accent-color: var(--primary); }
  .time-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  .time-row label { font-weight: 500; font-size: 0.9rem; min-width: 120px; }
  .time-display {
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: var(--primary); font-weight: 500; font-size: 0.95rem;
  }
  .time-inputs { display: flex; align-items: center; gap: 4px; }
  .time-inputs input {
    width: 52px; padding: 8px 4px; border: 1px solid var(--border);
    border-radius: 6px; text-align: center;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.95rem;
  }
  .time-inputs input:focus {
    outline: none; border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
  }
  .time-inputs input.wide { width: 64px; }
  .time-sep { color: var(--text-muted); font-weight: 500; font-size: 1.1rem; }
  .time-label-sm { color: var(--text-muted); font-size: 0.8rem; margin-left: 4px; }
  .btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 10px 20px; border: none; border-radius: 8px;
    font-size: 0.9rem; font-weight: 500; cursor: pointer; transition: all 0.15s;
  }
  .btn-primary { background: var(--primary); color: white; }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-primary:disabled { background: #94a3b8; cursor: not-allowed; }
  .btn-secondary { background: var(--bg); color: var(--text); border: 1px solid var(--border); }
  .btn-secondary:hover { background: #e2e8f0; }
  .btn-sm { padding: 6px 12px; font-size: 0.8rem; }
  .btn-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .log {
    background: #0f172a; color: #e2e8f0;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem;
    padding: 16px; border-radius: var(--radius);
    max-height: 200px; overflow-y: auto; line-height: 1.5;
    white-space: pre-wrap; word-break: break-word;
  }
  .log .success { color: #4ade80; }
  .log .error { color: #f87171; }
  .log .info { color: #60a5fa; }
  .progress-bar-outer {
    width: 100%; height: 6px; background: var(--border);
    border-radius: 3px; overflow: hidden; margin-bottom: 12px; display: none;
  }
  .progress-bar-outer.active { display: block; }
  .progress-bar-inner {
    height: 100%; background: var(--primary);
    border-radius: 3px; transition: width 0.3s; width: 0%;
  }
  #map-card { display: none; }
  #map-card.active { display: block; }
  #map {
    width: 100%; height: 400px; border-radius: 8px;
    border: 1px solid var(--border);
    position: relative; z-index: 0;
  }
  .leaflet-container img { max-width: none !important; }
  .map-info {
    font-size: 0.82rem; color: var(--text-muted); margin-top: 8px;
  }
  .map-legend {
    display: flex; gap: 16px; align-items: center;
    font-size: 0.8rem; color: var(--text-muted); margin-top: 6px;
  }
  .legend-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 4px; vertical-align: middle;
  }
  /* Stats card */
  #stats-card, #charts-card, #laps-card, #zones-card, #thresholds-card, #moments-card, #group-moments-card { display: none; }
  #stats-card.active, #charts-card.active, #laps-card.active, #zones-card.active,
  #thresholds-card.active, #moments-card.active, #group-moments-card.active { display: block; }
  .stats-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  .stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
  }
  .stat-item {
    background: var(--bg); border-radius: 8px; padding: 12px; text-align: center;
  }
  .stat-label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--text-muted); margin-bottom: 4px;
  }
  .stat-value {
    font-size: 1.25rem; font-weight: 700; color: var(--text);
  }
  .stat-value .stat-unit {
    font-size: 0.75rem; font-weight: 400; color: var(--text-muted); margin-left: 2px;
  }
  .unit-toggle {
    display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
  }
  .unit-toggle button {
    border: none; background: var(--card); color: var(--text-muted);
    padding: 4px 12px; font-size: 0.78rem; cursor: pointer; transition: all 0.15s;
  }
  .unit-toggle button.active {
    background: var(--primary); color: #fff;
  }
  /* Charts */
  .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px; }
  .chart-tabs {
    display: flex; gap: 4px; flex-wrap: wrap;
  }
  .chart-tab {
    border: 1px solid var(--border); background: var(--card); color: var(--text-muted);
    padding: 5px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer;
    transition: all 0.15s;
  }
  .chart-tab.active { background: var(--primary); color: #fff; border-color: var(--primary); }
  .chart-tab:hover:not(.active) { background: var(--bg); }
  .chart-container { position: relative; width: 100%; height: 280px; }
  #chartCanvas { width: 100%; height: 100%; }
  /* Laps table */
  .laps-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .laps-table th {
    text-align: left; padding: 8px 10px; font-weight: 600; color: var(--text-muted);
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em;
    border-bottom: 2px solid var(--border); background: var(--bg);
  }
  .laps-table td {
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.82rem;
  }
  .laps-table tr:last-child td { border-bottom: none; }
  .laps-table tr:hover td { background: #f1f5f9; }
  /* Zone bars */
  .zone-section { margin-bottom: 16px; }
  .zone-section-title { font-size: 0.82rem; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
  .zone-inputs { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; font-size: 0.82rem; }
  .zone-inputs label { color: var(--text-muted); font-size: 0.78rem; }
  .zone-inputs input {
    width: 56px; padding: 4px 6px; border: 1px solid var(--border); border-radius: 4px;
    text-align: center; font-size: 0.82rem;
  }
  .zone-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .zone-label { width: 100px; font-size: 0.75rem; color: var(--text-muted); text-align: right; flex-shrink: 0; }
  .zone-bar-bg { flex: 1; height: 22px; background: var(--bg); border-radius: 4px; overflow: hidden; position: relative; }
  .zone-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s; display: flex; align-items: center; justify-content: flex-end; padding-right: 6px; min-width: 2px; }
  .zone-bar-text { font-size: 0.7rem; color: white; font-weight: 600; white-space: nowrap; }
  .zone-time { width: 60px; font-size: 0.75rem; color: var(--text-muted); font-family: monospace; }
  /* Overlay legend */
  .overlay-legend { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; font-size: 0.78rem; }
  .overlay-legend-item { display: flex; align-items: center; gap: 4px; }
  .overlay-swatch { width: 14px; height: 3px; border-radius: 2px; }
  /* Similarity scores */
  .similarity-card { display: none; }
  .similarity-card.active { display: block; }
  .sim-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .sim-item { background: var(--bg); border-radius: 8px; padding: 14px; text-align: center; }
  .sim-score { font-size: 1.8rem; font-weight: 800; line-height: 1; }
  .sim-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); margin-bottom: 6px; }
  .sim-detail { font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; }
  .sim-bar-bg { height: 6px; background: var(--border); border-radius: 3px; margin-top: 8px; overflow: hidden; }
  .sim-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  /* Map cursor marker */
  .map-cursor-marker {
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--primary); border: 2px solid white;
    box-shadow: 0 0 6px rgba(37,99,235,0.5);
  }
  /* Highlight marker for selected moment */
  .moment-highlight-ring {
    border-radius: 50%;
    border: 3px solid var(--primary);
    box-shadow: 0 0 0 6px rgba(37,99,235,0.25), 0 0 16px rgba(37,99,235,0.4);
    animation: moment-pulse 1.5s ease-in-out infinite;
  }
  @keyframes moment-pulse {
    0%, 100% { box-shadow: 0 0 0 6px rgba(37,99,235,0.25), 0 0 16px rgba(37,99,235,0.4); }
    50% { box-shadow: 0 0 0 12px rgba(37,99,235,0.10), 0 0 24px rgba(37,99,235,0.2); }
  }
  .moment-list li { cursor: pointer; }
  .moment-list li:hover { background: var(--bg); }
  .moment-list li.moment-active { background: var(--bg); border-left-color: var(--primary) !important; }
  .group-moment-item { cursor: pointer; transition: box-shadow 0.15s; }
  .group-moment-item:hover { box-shadow: 0 0 0 2px var(--primary); }
  .group-moment-item.moment-active { box-shadow: 0 0 0 2px var(--primary); background: color-mix(in srgb, var(--primary) 6%, var(--bg)); }
  .route-shape-tag {
    display: inline-block; font-size: 0.72rem; padding: 2px 7px; border-radius: 10px;
    background: #f0f4ff; color: #3b5998; margin-top: 4px; font-weight: 500;
  }
  [data-theme="dark"] .route-shape-tag { background: #1e293b; color: #93c5fd; }

  /* Threshold settings */
  .th-subtitle {
    font-size: 0.84rem; color: var(--text-muted); margin-bottom: 16px; line-height: 1.4;
  }
  .th-group { margin-bottom: 16px; }
  .th-group:last-child { margin-bottom: 0; }
  .th-group-title {
    font-size: 0.78rem; font-weight: 600; color: var(--text);
    margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
  }
  .th-group-title .th-icon { font-size: 0.9rem; }
  .threshold-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
  }
  .threshold-item {
    display: flex; flex-direction: column; gap: 3px;
    background: var(--bg); border-radius: 8px; padding: 10px;
  }
  .threshold-item label {
    font-size: 0.75rem; font-weight: 600; color: var(--text);
  }
  .threshold-item .th-hint {
    font-size: 0.7rem; color: var(--text-muted); line-height: 1.3;
  }
  .threshold-item input {
    width: 100%; padding: 6px 8px; border: 1px solid var(--border);
    border-radius: 6px; font-size: 0.85rem; text-align: center;
    margin-top: 2px;
  }
  .threshold-item input:focus {
    outline: none; border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
  }
  /* Moment badges */
  .moment-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .moment-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 10px; border-radius: 16px; font-size: 0.8rem; font-weight: 600;
  }
  .moment-badge.speed_surge { background: #fce7f3; color: #be185d; }
  .moment-badge.power_spike { background: #fef3c7; color: #b45309; }
  .moment-badge.sprint { background: #fee2e2; color: #b91c1c; }
  .moment-badge.climb { background: #dcfce7; color: #166534; }
  .moment-badge.speed_demon { background: #f3e8ff; color: #7e22ce; }
  /* Moment list */
  .moment-type-section { margin-bottom: 12px; }
  .moment-type-header {
    font-size: 0.8rem; font-weight: 600; margin-bottom: 6px; cursor: pointer;
    display: flex; align-items: center; gap: 6px;
  }
  .moment-type-header:hover { color: var(--primary); }
  .moment-list { list-style: none; font-size: 0.8rem; }
  .moment-list li {
    padding: 4px 8px; border-left: 3px solid var(--border);
    margin-bottom: 4px; display: flex; justify-content: space-between;
    align-items: center; font-family: 'SF Mono', 'Fira Code', monospace;
  }
  .moment-list li.speed_surge { border-color: #e6198a; }
  .moment-list li.power_spike { border-color: #d97706; }
  .moment-list li.sprint { border-color: #dc2626; }
  .moment-list li.climb { border-color: #15803d; }
  /* Group moments */
  .group-moment-item {
    background: var(--bg); border-radius: 8px; padding: 12px; margin-bottom: 8px;
  }
  .group-moment-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
  }
  .group-moment-members {
    font-size: 0.78rem; color: var(--text-muted);
  }
  .group-moment-members span { margin-right: 12px; }
  .copy-ai-btn {
    margin-left: auto; background: none; border: 1px solid var(--border); border-radius: 6px;
    padding: 2px 8px; font-size: 0.75rem; color: var(--text-muted); cursor: pointer;
    position: relative; display: inline-flex; align-items: center; gap: 4px;
  }
  .copy-ai-btn:hover { background: var(--bg); color: var(--text); }
  .copy-ai-menu {
    display: none; position: absolute; top: 100%; right: 0; margin-top: 4px;
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 100; min-width: 140px; overflow: hidden;
  }
  .copy-ai-menu.show { display: block; }
  .copy-ai-menu div {
    padding: 8px 12px; font-size: 0.8rem; cursor: pointer; white-space: nowrap;
  }
  .copy-ai-menu div:hover { background: var(--bg); }
  .copy-toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #333; color: #fff; padding: 8px 20px; border-radius: 8px;
    font-size: 0.85rem; z-index: 9999; animation: toastFade 2.2s forwards;
  }
  @keyframes toastFade { 0%,70% { opacity: 1; } 100% { opacity: 0; } }
  .moment-copy-btn {
    background: none; border: 1px solid var(--border); cursor: pointer; color: var(--text-muted);
    font-size: 0.8rem; padding: 1px 6px; margin-left: 4px; border-radius: 4px; position: relative;
  }
  .moment-copy-btn:hover { color: var(--text); background: var(--bg); }
  .moment-list li { overflow: visible; }

  @media (max-width: 600px) {
    body { padding: 12px; }
    .time-row { flex-direction: column; align-items: flex-start; }
    .time-row label { min-width: auto; }
    #map { height: 300px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .threshold-grid { grid-template-columns: 1fr 1fr; }
    .chart-container { height: 220px; }
  }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.0/chart.umd.min.js" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.2.0/chartjs-plugin-zoom.min.js" crossorigin="anonymous"></script>
</head>
<body>
<div class="container">
  <h1>FIT Toolkit</h1>
  <p class="subtitle">A toolkit for working with Garmin FIT files.</p>

  <div class="card">
    <div class="card-title">1. Select FIT Files</div>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="drop-zone-icon">&#128228;</div>
      <div class="drop-zone-text"><strong>Click to browse</strong> or drag &amp; drop .fit files here</div>
    </div>
    <input type="file" id="fileInput" accept=".fit,.FIT" multiple>
    <ul class="file-list" id="fileList"></ul>
  </div>

  <div class="card" id="map-card">
    <div class="card-title">Route Map</div>
    <div id="map"></div>
    <div class="map-info" id="mapInfo"></div>
    <div class="map-legend" id="mapLegend">
      <span><span class="legend-dot" style="background:#22c55e"></span>Start</span>
      <span><span class="legend-dot" style="background:#ef4444"></span>Finish</span>
    </div>
    <div class="map-legend" id="momentLegend" style="display:none;margin-top:2px">
      <span><span class="legend-dot" style="background:#e6198a"></span>Speed Surge</span>
      <span><span class="legend-dot" style="background:#d97706"></span>Power Spike</span>
      <span><span class="legend-dot" style="background:#dc2626"></span>Sprint</span>
      <span><span class="legend-dot" style="background:#15803d"></span>Climb</span>
    </div>
  </div>

  <div class="card similarity-card" id="similarity-card">
    <div class="card-title">Route Similarity</div>
    <div class="sim-grid" id="simGrid"></div>
  </div>

  <div class="card" id="charts-card">
    <div class="card-header">
      <div class="card-title">Charts</div>
      <div class="chart-header">
        <div class="chart-tabs" id="chartTabs"></div>
        <button class="btn btn-secondary btn-sm" id="resetZoomBtn" onclick="resetChartZoom()" style="display:none">Reset Zoom</button>
      </div>
      <div class="overlay-legend" id="overlayLegend" style="display:none"></div>
    </div>
    <div class="chart-container"><canvas id="chartCanvas"></canvas></div>
  </div>

  <div class="card" id="stats-card">
    <div class="stats-header">
      <div class="card-title" style="margin-bottom:0">Activity Stats</div>
      <div style="display:flex;gap:8px;align-items:center">
      <button class="btn btn-secondary btn-sm" id="reportBtn" onclick="exportReport()" style="display:none">Export Report</button>
      <div class="unit-toggle">
        <button class="active" onclick="setUnits('metric')">Metric</button>
        <button onclick="setUnits('imperial')">Imperial</button>
      </div>
      </div>
    </div>
    <div class="stats-grid" id="statsGrid"></div>
  </div>

  <div class="card" id="laps-card">
    <div class="card-title">Lap Splits</div>
    <div style="overflow-x:auto"><table class="laps-table" id="lapsTable"></table></div>
  </div>

  <div class="card" id="zones-card">
    <div class="card-title">Zone Analysis</div>
    <div class="zone-inputs" id="zoneInputs">
      <label>Age:</label>
      <input type="number" id="zoneAge" value="30" min="10" max="99" onchange="reloadZones()">
      <label>Max HR:</label>
      <input type="number" id="zoneMaxHR" value="190" min="100" max="230" onchange="reloadZones()">
      <label>FTP (W):</label>
      <input type="number" id="zoneFTP" value="200" min="50" max="500" onchange="reloadZones()">
    </div>
    <div id="zoneContent"></div>
  </div>

  <div class="card" id="thresholds-card">
    <div class="card-title">Moment Detection Settings</div>
    <div class="th-subtitle">Configure how key moments are detected from your ride data. Lower thresholds will detect more events, higher thresholds only flag the most extreme efforts.</div>

    <div class="th-group">
      <div class="th-group-title"><span class="th-icon">&#x1F3CE;&#xFE0F;</span> Speed &amp; Power Thresholds</div>
      <div class="threshold-grid">
        <div class="threshold-item">
          <label>Speed Surge</label>
          <div class="th-hint">Minimum instantaneous speed to flag as a notable high-speed moment. Typical: 45&ndash;55 km/h for road cycling.</div>
          <input type="number" id="thSpeedSurge" value="50" step="1" min="1">
          <div class="th-hint" style="text-align:center;margin-top:1px">km/h</div>
        </div>
        <div class="threshold-item">
          <label>Power Spike</label>
          <div class="th-hint">Minimum wattage for a single-second power spike. Set higher for stronger riders. Typical: 350&ndash;500 W.</div>
          <input type="number" id="thPowerSpike" value="400" step="10" min="1">
          <div class="th-hint" style="text-align:center;margin-top:1px">watts</div>
        </div>
      </div>
    </div>

    <div class="th-group">
      <div class="th-group-title"><span class="th-icon">&#x1F3C3;</span> Sprint Detection</div>
      <div class="threshold-grid">
        <div class="threshold-item">
          <label>Sprint Power</label>
          <div class="th-hint">Power output that triggers sprint detection. A sprint starts when power, acceleration, or speed exceeds its threshold.</div>
          <input type="number" id="thSprintPower" value="400" step="10" min="1">
          <div class="th-hint" style="text-align:center;margin-top:1px">watts</div>
        </div>
        <div class="threshold-item">
          <label>Sprint Acceleration</label>
          <div class="th-hint">Rate of speed increase that can trigger a sprint. Higher values require sharper accelerations. Typical: 0.8&ndash;1.5 m/s&sup2;.</div>
          <input type="number" id="thSprintAccel" value="1.0" step="0.1" min="0.1">
          <div class="th-hint" style="text-align:center;margin-top:1px">m/s&sup2;</div>
        </div>
        <div class="threshold-item">
          <label>Minimum Duration</label>
          <div class="th-hint">Shortest effort that counts as a sprint. Filters out brief surges that aren&rsquo;t true sprints.</div>
          <input type="number" id="thSprintMinDur" value="3" step="1" min="1">
          <div class="th-hint" style="text-align:center;margin-top:1px">seconds</div>
        </div>
      </div>
    </div>

    <div class="th-group">
      <div class="th-group-title"><span class="th-icon">&#x26F0;&#xFE0F;</span> Climb Detection</div>
      <div class="threshold-grid">
        <div class="threshold-item">
          <label>Gradient</label>
          <div class="th-hint">Minimum road steepness to count as climbing. 5% is a moderate hill, 8%+ is steep. Lower to catch gentle slopes.</div>
          <input type="number" id="thClimbGradient" value="5" step="0.5" min="0.5">
          <div class="th-hint" style="text-align:center;margin-top:1px">%</div>
        </div>
        <div class="threshold-item">
          <label>Minimum Duration</label>
          <div class="th-hint">Shortest climb to report. Filters out brief ramps and overpasses. Typical: 20&ndash;60 seconds.</div>
          <input type="number" id="thClimbMinDur" value="30" step="5" min="5">
          <div class="th-hint" style="text-align:center;margin-top:1px">seconds</div>
        </div>
        <div class="threshold-item">
          <label>Minimum Elevation Gain</label>
          <div class="th-hint">Minimum total meters gained for a climb to be reported. Filters short steep bumps.</div>
          <input type="number" id="thClimbMinGain" value="10" step="1" min="1">
          <div class="th-hint" style="text-align:center;margin-top:1px">meters</div>
        </div>
      </div>
    </div>

    <div class="th-group">
      <div class="th-group-title"><span class="th-icon">&#x1F465;</span> Group Analysis</div>
      <div class="threshold-grid" style="grid-template-columns: 1fr">
        <div class="threshold-item">
          <label>Time Window</label>
          <div class="th-hint">When comparing multiple riders, moments within this window are grouped together. A 60s window means two riders&rsquo; sprints count as a &ldquo;group sprint&rdquo; if they happened within 60 seconds of each other.</div>
          <input type="number" id="thTimeWindow" value="60" step="5" min="5" style="max-width:120px">
          <div class="th-hint" style="max-width:120px;text-align:center;margin-top:1px">seconds</div>
        </div>
      </div>
    </div>

    <div style="margin-top:12px">
      <button class="btn btn-primary btn-sm" onclick="redetectMoments()">Re-Analyze</button>
    </div>
  </div>

  <div class="card" id="moments-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <div class="card-title" style="margin-bottom:0">Moments &amp; Achievements</div>
      <label style="display:inline-flex;align-items:center;gap:6px;font-size:0.78rem;color:var(--text-muted);cursor:pointer;user-select:none" title="Show HUD overlay (speed, power, gradient) in POV prompts">
        <input type="checkbox" id="povHudToggle" checked onchange="povHudEnabled=this.checked" style="cursor:pointer">
        POV HUD
      </label>
    </div>
    <div id="momentsContent"></div>
  </div>

  <div class="card" id="group-moments-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <div class="card-title" style="margin-bottom:0">Group Moments</div>
      <button class="btn btn-secondary btn-sm" onclick="exportGroupReport()">Export Group Report</button>
    </div>
    <div id="groupMomentsContent"></div>
  </div>

  <div class="card">
    <div class="card-title">2. Set New Start Time</div>
    <div class="time-row">
      <label>Original start:</label>
      <span class="time-display" id="originalTime">&mdash;</span>
    </div>
    <div class="time-row">
      <label>New start (UTC):</label>
      <div class="time-inputs">
        <input type="text" id="year" maxlength="4" placeholder="YYYY" class="wide">
        <span class="time-sep">-</span>
        <input type="text" id="month" maxlength="2" placeholder="MM">
        <span class="time-sep">-</span>
        <input type="text" id="day" maxlength="2" placeholder="DD">
        <span style="width:12px"></span>
        <input type="text" id="hour" maxlength="2" placeholder="HH">
        <span class="time-sep">:</span>
        <input type="text" id="minute" maxlength="2" placeholder="mm">
        <span class="time-sep">:</span>
        <input type="text" id="second" maxlength="2" placeholder="ss">
        <span class="time-label-sm">UTC</span>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn btn-secondary btn-sm" onclick="useCurrentTime()">Use Current Time</button>
      <button class="btn btn-secondary btn-sm" onclick="copyOriginal()">Use Original Time</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">3. Adjust &amp; Download</div>
    <div class="progress-bar-outer" id="progressOuter">
      <div class="progress-bar-inner" id="progressInner"></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" id="adjustBtn" onclick="adjustFiles()" disabled>Adjust Files</button>
      <button class="btn btn-secondary" id="downloadBtn" onclick="downloadResults()" style="display:none">Download Results</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Log</div>
    <div class="log" id="log">Ready. Add FIT files to begin.</div>
  </div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileList = document.getElementById('fileList');
const logEl = document.getElementById('log');
let files = {};
let resultIds = [];

// ---- Map ----
let map = null;
let routeLayer = null;
let markerLayer = null;
let cursorMarker = null;

// ---- Multi-file overlay ----
let overlayFileIds = [];  // files selected for overlay
const OVERLAY_COLORS = ['#2563eb','#dc2626','#16a34a','#f59e0b','#8b5cf6','#ec4899','#14b8a6','#f97316'];

function initMap() {
  if (map) return;
  if (typeof L === 'undefined') {
    log('Error: Leaflet library failed to load. Check your internet connection.', 'error');
    return;
  }
  map = L.map('map', { zoomControl: true, attributionControl: true });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
  }).addTo(map);
  routeLayer = L.layerGroup().addTo(map);
  markerLayer = L.layerGroup().addTo(map);

  // Cursor marker for chart-map linking
  const icon = L.divIcon({ className: 'map-cursor-marker', iconSize: [14, 14], iconAnchor: [7, 7] });
  cursorMarker = L.marker([0, 0], { icon: icon, interactive: false }).addTo(map);
  cursorMarker.setOpacity(0);
}

async function loadRoute(fileId, color, showMarkers) {
  try {
    const r = await fetch('/gps/' + fileId);
    const d = await r.json();
    if (d.error || !d.points || d.points.length < 2) return;

    const mapCard = document.getElementById('map-card');
    mapCard.classList.add('active');
    await new Promise(resolve => setTimeout(resolve, 50));
    initMap();

    const latlngs = d.points.map(p => [p[0], p[1]]);
    color = color || '#2563eb';

    L.polyline(latlngs, { color: color, weight: 3.5, opacity: 0.85 }).addTo(routeLayer);

    if (showMarkers !== false) {
      L.circleMarker(latlngs[0], {
        radius: 8, fillColor: '#22c55e', color: '#fff', weight: 2, fillOpacity: 1
      }).bindPopup('Start').addTo(markerLayer);
      L.circleMarker(latlngs[latlngs.length - 1], {
        radius: 8, fillColor: '#ef4444', color: '#fff', weight: 2, fillOpacity: 1
      }).bindPopup('Finish').addTo(markerLayer);
    }

    const bounds = L.latLngBounds(latlngs).pad(0.05);
    map.fitBounds(bounds);
    document.getElementById('mapInfo').textContent = d.points.length + ' GPS points';
    setTimeout(() => { map.invalidateSize(); map.fitBounds(bounds); }, 200);
  } catch (e) {
    console.warn('Map load error:', e);
  }
}

function clearMapLayers() {
  if (routeLayer) routeLayer.clearLayers();
  if (markerLayer) markerLayer.clearLayers();
}

function hideMap() {
  document.getElementById('map-card').classList.remove('active');
  clearMapLayers();
  document.getElementById('mapInfo').textContent = '';
}

function log(msg, cls) {
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = msg + '\\n';
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
}

function handleFiles(list) {
  for (const f of list) {
    if (!f.name.toLowerCase().endsWith('.fit')) { log('Skipped: ' + f.name, 'error'); continue; }
    uploadFile(f);
  }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) { log('Error: ' + d.error, 'error'); return; }
    files[d.id] = d;
    renderFileList();
    const gpsNote = d.gps_count > 0 ? ', ' + d.gps_count + ' GPS points' : '';
    const lapNote = d.has_laps ? ', ' + d.lap_count + ' laps' : '';
    log('Added: ' + d.filename + ' (start: ' + (d.original_start || 'N/A') + gpsNote + lapNote + ')', 'info');
    if (Object.keys(files).length === 1 && d.original_start) {
      document.getElementById('originalTime').textContent = d.original_start + ' UTC';
    }
    if (d.gps_count > 0) await loadRoute(d.id);
    if (d.has_stats) loadStats(d.id);
    if (d.has_laps) loadLaps(d.id);
    loadMoments(d.id);
    updateBtn();
  } catch (e) { log('Upload failed: ' + e.message, 'error'); }
}

function removeFile(id) {
  fetch('/remove/' + id, { method: 'DELETE' });
  delete files[id];
  delete currentMomentsData[id];
  overlayFileIds = overlayFileIds.filter(x => x !== id);
  renderFileList(); updateBtn();
  const keys = Object.keys(files);
  document.getElementById('originalTime').textContent =
    keys.length > 0 ? (files[keys[0]].original_start || '\\u2014') + ' UTC' : '\\u2014';
  if (keys.length > 0) {
    const first = files[keys[0]];
    if (first.gps_count > 0) loadRoute(keys[0]); else hideMap();
    if (first.has_stats) loadStats(keys[0]); else hideStats();
    if (first.has_laps) loadLaps(keys[0]); else hideLaps();
  } else { hideMap(); hideStats(); hideLaps(); hideZones(); hideSimilarity(); hideMoments(); }
  if (overlayFileIds.length > 1) refreshOverlay();
}

function renderFileList() {
  fileList.innerHTML = '';
  const multiFile = Object.keys(files).length > 1;
  for (const [id, f] of Object.entries(files)) {
    const li = document.createElement('li'); li.className = 'file-item';
    const overlayCheck = multiFile
      ? '<input type="checkbox" class="file-overlay-cb" ' + (overlayFileIds.includes(id) ? 'checked' : '') +
        ' onchange="toggleOverlay(\\'' + id + '\\', this.checked)" title="Include in overlay">'
      : '';
    li.innerHTML = '<div class="file-info"><span class="file-name">' + f.filename +
      '</span><span class="file-meta">Start: ' + (f.original_start || 'N/A') +
      ' UTC &middot; ' + f.size_str + '</span></div>' +
      '<div class="file-actions">' + overlayCheck +
      '<button class="file-remove" onclick="removeFile(\\'' + id + '\\')" title="Remove">&times;</button></div>';
    fileList.appendChild(li);
  }
}

function updateBtn() { document.getElementById('adjustBtn').disabled = Object.keys(files).length === 0; }

function fillTime(s) {
  const p = s.split(/[\\s\\-:T]/);
  if (p.length >= 6) {
    document.getElementById('year').value = p[0];
    document.getElementById('month').value = p[1];
    document.getElementById('day').value = p[2];
    document.getElementById('hour').value = p[3];
    document.getElementById('minute').value = p[4];
    document.getElementById('second').value = p[5];
  }
}

function useCurrentTime() {
  const now = new Date();
  document.getElementById('year').value = now.getUTCFullYear();
  document.getElementById('month').value = String(now.getUTCMonth() + 1).padStart(2, '0');
  document.getElementById('day').value = String(now.getUTCDate()).padStart(2, '0');
  document.getElementById('hour').value = String(now.getUTCHours()).padStart(2, '0');
  document.getElementById('minute').value = String(now.getUTCMinutes()).padStart(2, '0');
  document.getElementById('second').value = String(now.getUTCSeconds()).padStart(2, '0');
}

function copyOriginal() {
  const k = Object.keys(files);
  if (k.length > 0 && files[k[0]].original_start) fillTime(files[k[0]].original_start);
}

function getNewDT() {
  const g = id => (document.getElementById(id).value || '00').padStart(2, '0');
  return (document.getElementById('year').value || '2024').padStart(4, '0') +
    '-' + g('month') + '-' + g('day') + 'T' + g('hour') + ':' + g('minute') + ':' + g('second');
}

async function adjustFiles() {
  const ids = Object.keys(files);
  if (!ids.length) return;
  const newTime = getNewDT();
  const btn = document.getElementById('adjustBtn');
  const dlBtn = document.getElementById('downloadBtn');
  const pOuter = document.getElementById('progressOuter');
  const pInner = document.getElementById('progressInner');
  btn.disabled = true; dlBtn.style.display = 'none';
  pOuter.classList.add('active'); resultIds = [];

  log('\\n' + '='.repeat(48), 'info');
  log('Processing ' + ids.length + ' file(s)...', 'info');
  log('New start time: ' + newTime + ' UTC', 'info');
  log('='.repeat(48), 'info');

  for (let i = 0; i < ids.length; i++) {
    const id = ids[i], f = files[id];
    pInner.style.width = ((i / ids.length) * 100) + '%';
    try {
      const r = await fetch('/adjust', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ file_id: id, new_start: newTime })
      });
      const d = await r.json();
      if (d.error) { log('[' + (i+1) + '/' + ids.length + '] ' + f.filename + ': ERROR - ' + d.error, 'error'); continue; }
      resultIds.push(d.result_id);
      const os = d.offset_seconds, dir = os > 0 ? 'forward' : 'backward', a = Math.abs(os);
      log('[' + (i+1) + '/' + ids.length + '] ' + f.filename, 'success');
      log('  Original: ' + d.original_start);
      log('  Shifted ' + dir + ' by ' + Math.floor(a/3600) + 'h ' + Math.floor((a%3600)/60) + 'm ' + (a%60) + 's');
      log('  Timestamps modified: ' + d.timestamps_modified);
    } catch (e) { log('[' + (i+1) + '/' + ids.length + '] ' + f.filename + ': ERROR - ' + e.message, 'error'); }
  }
  pInner.style.width = '100%';
  log('\\n' + '='.repeat(48), 'info');
  log('Done! ' + resultIds.length + '/' + ids.length + ' files adjusted.', 'success');
  log('='.repeat(48), 'info');
  btn.disabled = false;
  if (resultIds.length > 0) dlBtn.style.display = 'inline-flex';
  setTimeout(() => pOuter.classList.remove('active'), 1000);
}

function downloadResults() {
  if (!resultIds.length) return;
  if (resultIds.length === 1) window.location.href = '/download/' + resultIds[0];
  else window.location.href = '/download-zip?' + resultIds.map(id => 'ids=' + id).join('&');
}

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => { handleFiles(fileInput.files); fileInput.value = ''; });
document.querySelectorAll('.time-inputs input').forEach(el => {
  el.addEventListener('input', function() { this.value = this.value.replace(/[^0-9]/g, ''); });
});

// Arrow up/down to increment/decrement time fields with carry-over
(function() {
  const fields = ['year','month','day','hour','minute','second'];
  const mins   = [1970, 1, 1, 0, 0, 0];
  const maxes  = [2099, 12, 31, 23, 59, 59];
  const pads   = [4, 2, 2, 2, 2, 2];

  function getVal(i) { return parseInt(document.getElementById(fields[i]).value, 10) || mins[i]; }
  function setVal(i, v) {
    document.getElementById(fields[i]).value = String(v).padStart(pads[i], '0');
  }
  function daysInMonth(y, m) { return new Date(y, m, 0).getDate(); }

  function adjust(idx, delta) {
    let vals = fields.map((_, i) => getVal(i));
    vals[idx] += delta;

    for (let i = 5; i >= 1; i--) {
      const mx = (i === 2) ? daysInMonth(vals[0], vals[1]) : maxes[i];
      const mn = mins[i];
      if (vals[i] > mx) { vals[i] = mn; if (i > 0) vals[i-1]++; }
      else if (vals[i] < mn) { vals[i] = mx; if (i > 0) vals[i-1]--; }
    }
    vals[0] = Math.max(mins[0], Math.min(maxes[0], vals[0]));
    vals[2] = Math.min(vals[2], daysInMonth(vals[0], vals[1]));

    vals.forEach((v, i) => setVal(i, v));
  }

  fields.forEach((fid, idx) => {
    document.getElementById(fid).addEventListener('keydown', function(e) {
      if (e.key === 'ArrowUp')   { e.preventDefault(); adjust(idx, 1); }
      if (e.key === 'ArrowDown') { e.preventDefault(); adjust(idx, -1); }
    });
  });
})();

useCurrentTime();

// ---- Stats & Charts ----
let currentStats = null;
let currentTimeseries = null;
let currentChart = null;
let activeTab = null;
let unitSystem = 'metric';
let currentFileId = null;  // track which file's data is displayed

const CONVERSIONS = {
  metric: {
    dist: v => v / 1000, distUnit: 'km',
    elev: v => v, elevUnit: 'm',
    speed: v => v * 3.6, speedUnit: 'km/h',
    temp: v => v, tempUnit: '\\u00b0C',
  },
  imperial: {
    dist: v => v / 1609.344, distUnit: 'mi',
    elev: v => v * 3.28084, elevUnit: 'ft',
    speed: v => v * 2.23694, speedUnit: 'mph',
    temp: v => v * 9/5 + 32, tempUnit: '\\u00b0F',
  }
};

function conv() { return CONVERSIONS[unitSystem]; }

function fmtDuration(sec) {
  if (!sec) return '\\u2014';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h > 0 ? h + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0')
    : m + ':' + String(s).padStart(2,'0');
}

function fmtElapsed(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return h + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
  return m + ':' + String(s).padStart(2,'0');
}

function setUnits(u) {
  unitSystem = u;
  document.querySelectorAll('.unit-toggle button').forEach(b => {
    b.classList.toggle('active', b.textContent.toLowerCase() === u);
  });
  if (currentStats) renderStats(currentStats);
  if (currentTimeseries && activeTab) renderChart(activeTab);
  if (currentLaps) renderLaps(currentLaps);
}

function renderStats(stats) {
  const c = conv();
  const items = [];
  const addStat = (label, val, unit) => {
    if (val !== undefined && val !== null) items.push({label, val, unit});
  };
  if (stats.total_distance !== undefined) addStat('Distance', c.dist(stats.total_distance).toFixed(2), c.distUnit);
  if (stats.total_timer_time !== undefined) addStat('Duration', fmtDuration(stats.total_timer_time), '');
  if (stats.avg_speed !== undefined) addStat('Avg Speed', c.speed(stats.avg_speed).toFixed(1), c.speedUnit);
  if (stats.total_ascent !== undefined) addStat('Ascent', Math.round(c.elev(stats.total_ascent)), c.elevUnit);
  if (stats.total_descent !== undefined) addStat('Descent', Math.round(c.elev(stats.total_descent)), c.elevUnit);
  if (stats.avg_heart_rate !== undefined) addStat('Avg HR', stats.avg_heart_rate, 'bpm');
  if (stats.max_heart_rate !== undefined) addStat('Max HR', stats.max_heart_rate, 'bpm');
  if (stats.avg_power !== undefined) addStat('Avg Power', stats.avg_power, 'W');
  if (stats.max_power !== undefined) addStat('Max Power', stats.max_power, 'W');
  if (stats.normalized_power !== undefined) addStat('NP', stats.normalized_power, 'W');
  if (stats.total_calories !== undefined) addStat('Calories', stats.total_calories, 'kcal');
  if (stats.avg_cadence !== undefined) addStat('Avg Cadence', stats.avg_cadence, 'rpm');
  if (stats.max_cadence !== undefined) addStat('Max Cadence', stats.max_cadence, 'rpm');
  if (stats.avg_temperature !== undefined) addStat('Avg Temp', Math.round(c.temp(stats.avg_temperature)), c.tempUnit);
  if (stats.max_speed !== undefined) addStat('Max Speed', c.speed(stats.max_speed).toFixed(1), c.speedUnit);

  const grid = document.getElementById('statsGrid');
  grid.innerHTML = items.map(i =>
    '<div class="stat-item"><div class="stat-label">' + i.label +
    '</div><div class="stat-value">' + i.val +
    (i.unit ? '<span class="stat-unit">' + i.unit + '</span>' : '') +
    '</div></div>'
  ).join('');
}

// ---- Chart definitions ----
const CHART_DEFS = {
  elevation: { key: 'elevation', label: 'Elevation', color: '#64748b', unitFn: c => c.elevUnit, convFn: (v, c) => c.elev(v) },
  speed:     { key: 'speed', label: 'Speed', color: '#2563eb', unitFn: c => c.speedUnit, convFn: (v, c) => c.speed(v) },
  heart_rate:{ key: 'heart_rate', label: 'Heart Rate', color: '#dc2626', unitFn: () => 'bpm', convFn: v => v },
  cadence:   { key: 'cadence', label: 'Cadence', color: '#9333ea', unitFn: () => 'rpm', convFn: v => v },
  power:     { key: 'power', label: 'Power', color: '#ea580c', unitFn: () => 'W', convFn: v => v },
  temperature:{ key: 'temperature', label: 'Temperature', color: '#0d9488', unitFn: c => c.tempUnit, convFn: (v, c) => c.temp(v) },
};

function buildChartTabs(ts) {
  const tabsEl = document.getElementById('chartTabs');
  tabsEl.innerHTML = '';
  let first = null;
  for (const [id, def] of Object.entries(CHART_DEFS)) {
    if (!ts[def.key]) continue;
    if (!first) first = id;
    const btn = document.createElement('button');
    btn.className = 'chart-tab';
    btn.textContent = def.label;
    btn.dataset.chart = id;
    btn.onclick = () => {
      tabsEl.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderChart(id);
    };
    tabsEl.appendChild(btn);
  }
  return first;
}

function renderChart(tabId) {
  activeTab = tabId;
  const def = CHART_DEFS[tabId];
  if (!def || !currentTimeseries || !currentTimeseries[def.key]) return;
  if (typeof Chart === 'undefined') { console.error('Chart.js not loaded'); return; }
  const c = conv();
  const elapsed = currentTimeseries.elapsed || [];
  const rawData = currentTimeseries[def.key];
  const data = rawData.map(v => {
    const cv = def.convFn(v, c);
    return Math.round(cv * 100) / 100;
  });
  const labels = elapsed.map(s => fmtElapsed(s));
  const unit = def.unitFn(c);

  if (currentChart) { currentChart.destroy(); currentChart = null; }

  const container = document.querySelector('.chart-container');
  container.innerHTML = '<canvas id="chartCanvas"></canvas>';
  const canvas = document.getElementById('chartCanvas');
  document.getElementById('resetZoomBtn').style.display = 'inline-flex';

  currentChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: def.label + ' (' + unit + ')',
        data: data,
        borderColor: def.color,
        backgroundColor: def.color + '1a',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => 'Time: ' + items[0].label,
            label: item => def.label + ': ' + item.formattedValue + ' ' + unit
          }
        },
        zoom: {
          zoom: {
            drag: { enabled: true, backgroundColor: 'rgba(37,99,235,0.1)', borderColor: 'rgba(37,99,235,0.4)', borderWidth: 1 },
            mode: 'x',
          },
          limits: { x: { minRange: 5 } }
        }
      },
      scales: {
        x: {
          display: true,
          ticks: { maxTicksLimit: 8, font: { size: 11 } },
          grid: { display: false },
        },
        y: {
          display: true,
          title: { display: true, text: unit, font: { size: 11 } },
          ticks: { font: { size: 11 } },
          grid: { color: '#e2e8f022' },
        }
      },
      onHover: (event, elements, chart) => {
        if (!currentTimeseries || !currentTimeseries.lat) return;
        if (!elements || elements.length === 0) {
          if (cursorMarker) cursorMarker.setOpacity(0);
          return;
        }
        const idx = elements[0].index;
        const lat = currentTimeseries.lat[idx];
        const lon = currentTimeseries.lon[idx];
        if (lat && lon && cursorMarker && map) {
          cursorMarker.setLatLng([lat, lon]);
          cursorMarker.setOpacity(1);
        }
      }
    }
  });
}

function resetChartZoom() {
  if (currentChart) currentChart.resetZoom();
}

// ---- Multi-file overlay chart ----
let overlayTimeseries = {};  // { fileId: timeseries }

function renderOverlayChart(tabId) {
  activeTab = tabId;
  const def = CHART_DEFS[tabId];
  if (!def) return;
  if (typeof Chart === 'undefined') return;
  const c = conv();
  const unit = def.unitFn(c);

  if (currentChart) { currentChart.destroy(); currentChart = null; }
  const container = document.querySelector('.chart-container');
  container.innerHTML = '<canvas id="chartCanvas"></canvas>';
  const canvas = document.getElementById('chartCanvas');
  document.getElementById('resetZoomBtn').style.display = 'inline-flex';

  const datasets = [];
  let maxLabels = [];
  overlayFileIds.forEach((fid, i) => {
    const ts = overlayTimeseries[fid];
    if (!ts || !ts[def.key]) return;
    const rawData = ts[def.key];
    const data = rawData.map(v => Math.round(def.convFn(v, c) * 100) / 100);
    const elapsed = ts.elapsed || [];
    const labels = elapsed.map(s => fmtElapsed(s));
    if (labels.length > maxLabels.length) maxLabels = labels;
    const color = OVERLAY_COLORS[i % OVERLAY_COLORS.length];
    datasets.push({
      label: (files[fid] ? files[fid].filename : fid),
      data: data,
      borderColor: color,
      backgroundColor: color + '1a',
      fill: false,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    });
  });

  if (datasets.length === 0) return;

  currentChart = new Chart(canvas, {
    type: 'line',
    data: { labels: maxLabels, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: true, labels: { boxWidth: 12, font: { size: 11 } } },
        zoom: {
          zoom: {
            drag: { enabled: true, backgroundColor: 'rgba(37,99,235,0.1)', borderColor: 'rgba(37,99,235,0.4)', borderWidth: 1 },
            mode: 'x',
          },
          limits: { x: { minRange: 5 } }
        }
      },
      scales: {
        x: { display: true, ticks: { maxTicksLimit: 8, font: { size: 11 } }, grid: { display: false } },
        y: { display: true, title: { display: true, text: unit, font: { size: 11 } }, ticks: { font: { size: 11 } } }
      }
    }
  });
}

function toggleOverlay(fileId, checked) {
  if (checked && !overlayFileIds.includes(fileId)) overlayFileIds.push(fileId);
  if (!checked) overlayFileIds = overlayFileIds.filter(x => x !== fileId);
  if (overlayFileIds.length > 1) refreshOverlay();
  else {
    // Revert to single-file view
    document.getElementById('overlayLegend').style.display = 'none';
    hideSimilarity();
    overlayTimeseries = {};
    const keys = Object.keys(files);
    if (keys.length > 0) {
      const first = overlayFileIds.length === 1 ? overlayFileIds[0] : keys[0];
      clearMapLayers();
      if (files[first].gps_count > 0) loadRoute(first);
      if (files[first].has_stats) loadStats(first);
    }
  }
}

async function refreshOverlay() {
  if (overlayFileIds.length < 2) return;
  // Load multi timeseries
  const url = '/timeseries-multi?ids=' + overlayFileIds.join('&ids=');
  try {
    const r = await fetch(url);
    overlayTimeseries = await r.json();
  } catch (e) { console.warn('Multi timeseries error:', e); return; }

  // Find available tabs across all files
  const allKeys = new Set();
  for (const fid of overlayFileIds) {
    const ts = overlayTimeseries[fid];
    if (ts) for (const [id, def] of Object.entries(CHART_DEFS)) {
      if (ts[def.key]) allKeys.add(id);
    }
  }

  // Build tabs
  const tabsEl = document.getElementById('chartTabs');
  tabsEl.innerHTML = '';
  let first = null;
  for (const id of Object.keys(CHART_DEFS)) {
    if (!allKeys.has(id)) continue;
    if (!first) first = id;
    const btn = document.createElement('button');
    btn.className = 'chart-tab' + (id === first ? ' active' : '');
    btn.textContent = CHART_DEFS[id].label;
    btn.onclick = () => {
      tabsEl.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderOverlayChart(id);
    };
    tabsEl.appendChild(btn);
  }

  document.getElementById('charts-card').classList.add('active');

  // Build overlay legend
  const legendEl = document.getElementById('overlayLegend');
  legendEl.style.display = 'flex';
  legendEl.innerHTML = overlayFileIds.map((fid, i) => {
    const color = OVERLAY_COLORS[i % OVERLAY_COLORS.length];
    const name = files[fid] ? files[fid].filename : fid;
    return '<span class="overlay-legend-item"><span class="overlay-swatch" style="background:' + color + '"></span>' + name + '</span>';
  }).join('');

  // Overlay routes on map
  clearMapLayers();
  const mapCard = document.getElementById('map-card');
  let anyGps = false;
  for (let i = 0; i < overlayFileIds.length; i++) {
    const fid = overlayFileIds[i];
    if (files[fid] && files[fid].gps_count > 0) {
      anyGps = true;
      await loadRoute(fid, OVERLAY_COLORS[i % OVERLAY_COLORS.length], i === 0);
    }
  }
  if (!anyGps) hideMap();

  if (first) renderOverlayChart(first);

  // Compute pairwise similarity for overlay files with GPS
  loadSimilarity();
}

async function loadSimilarity() {
  const gpsIds = overlayFileIds.filter(fid => files[fid] && files[fid].gps_count > 0);
  if (gpsIds.length < 2) { hideSimilarity(); return; }

  // For now, compare all pairs (practical for 2-4 files)
  const pairs = [];
  for (let i = 0; i < gpsIds.length; i++) {
    for (let j = i + 1; j < gpsIds.length; j++) {
      pairs.push([gpsIds[i], gpsIds[j]]);
    }
  }

  const results = [];
  for (const [a, b] of pairs) {
    try {
      const r = await fetch('/similarity?a=' + a + '&b=' + b);
      const d = await r.json();
      if (!d.error) results.push(d);
    } catch (e) { console.warn('Similarity error:', e); }
  }

  if (results.length > 0) {
    document.getElementById('similarity-card').classList.add('active');
    renderSimilarity(results);
  } else {
    hideSimilarity();
  }
}

function scoreColor(pct) {
  if (pct >= 80) return '#16a34a';
  if (pct >= 50) return '#f59e0b';
  return '#dc2626';
}

function renderSimilarity(results) {
  const grid = document.getElementById('simGrid');
  let html = '';
  for (const r of results) {
    const nameA = r.file_a.replace(/\\.fit$/i, '');
    const nameB = r.file_b.replace(/\\.fit$/i, '');
    const fColor = scoreColor(r.frechet_score);
    const oColor = scoreColor(r.overlap_avg);
    html += '<div class="sim-item">' +
      '<div class="sim-label">Shape Similarity</div>' +
      '<div class="sim-score" style="color:' + fColor + '">' + r.frechet_score + '%</div>' +
      '<div class="sim-bar-bg"><div class="sim-bar-fill" style="width:' + r.frechet_score + '%;background:' + fColor + '"></div></div>' +
      '<div class="sim-detail">Fr\\u00e9chet: ' + (r.frechet_m < 1000 ? Math.round(r.frechet_m) + ' m' : (r.frechet_m/1000).toFixed(1) + ' km') + '</div>' +
      '</div>';
    html += '<div class="sim-item">' +
      '<div class="sim-label">Route Overlap</div>' +
      '<div class="sim-score" style="color:' + oColor + '">' + r.overlap_avg + '%</div>' +
      '<div class="sim-bar-bg"><div class="sim-bar-fill" style="width:' + r.overlap_avg + '%;background:' + oColor + '"></div></div>' +
      '<div class="sim-detail">' + nameA + ': ' + r.overlap_a + '% \\u00b7 ' + nameB + ': ' + r.overlap_b + '%</div>' +
      '</div>';
  }
  grid.innerHTML = html;
}

function hideSimilarity() {
  document.getElementById('similarity-card').classList.remove('active');
  document.getElementById('simGrid').innerHTML = '';
}

async function loadStats(fileId) {
  currentFileId = fileId;
  try {
    const [statsR, tsR] = await Promise.all([
      fetch('/stats/' + fileId),
      fetch('/timeseries/' + fileId)
    ]);
    currentStats = await statsR.json();
    currentTimeseries = await tsR.json();

    if (currentStats && Object.keys(currentStats).length > 0) {
      document.getElementById('stats-card').classList.add('active');
      document.getElementById('reportBtn').style.display = 'inline-flex';
      renderStats(currentStats);
      // Update max HR from stats if available
      if (currentStats.max_heart_rate) {
        document.getElementById('zoneMaxHR').value = currentStats.max_heart_rate;
      }
    }

    if (currentTimeseries && currentTimeseries.count > 0) {
      document.getElementById('charts-card').classList.add('active');
      document.getElementById('overlayLegend').style.display = 'none';
      const firstTab = buildChartTabs(currentTimeseries);
      if (firstTab) {
        document.querySelector('.chart-tab').classList.add('active');
        renderChart(firstTab);
      }
    }
    // Load zones
    loadZones(fileId);
  } catch (e) { console.warn('Stats load error:', e); }
}

function hideStats() {
  document.getElementById('stats-card').classList.remove('active');
  document.getElementById('charts-card').classList.remove('active');
  document.getElementById('resetZoomBtn').style.display = 'none';
  document.getElementById('reportBtn').style.display = 'none';
  if (currentChart) { currentChart.destroy(); currentChart = null; }
  currentStats = null;
  currentTimeseries = null;
  activeTab = null;
}

// ---- Laps ----
let currentLaps = null;

async function loadLaps(fileId) {
  try {
    const r = await fetch('/laps/' + fileId);
    currentLaps = await r.json();
    if (currentLaps && currentLaps.length > 0) {
      document.getElementById('laps-card').classList.add('active');
      renderLaps(currentLaps);
    }
  } catch (e) { console.warn('Laps load error:', e); }
}

function renderLaps(laps) {
  const c = conv();
  const tbl = document.getElementById('lapsTable');
  // Determine available columns
  const cols = [{ key: 'lap', label: '#' }];
  const colDefs = [
    { key: 'total_timer_time', label: 'Duration', fmt: v => fmtDuration(v) },
    { key: 'total_distance', label: 'Distance', fmt: v => c.dist(v).toFixed(2) + ' ' + c.distUnit },
    { key: 'avg_speed', label: 'Avg Speed', fmt: v => c.speed(v).toFixed(1) + ' ' + c.speedUnit },
    { key: 'avg_heart_rate', label: 'Avg HR', fmt: v => v + ' bpm' },
    { key: 'max_heart_rate', label: 'Max HR', fmt: v => v + ' bpm' },
    { key: 'avg_power', label: 'Avg Power', fmt: v => v + ' W' },
    { key: 'avg_cadence', label: 'Cadence', fmt: v => v + ' rpm' },
    { key: 'total_ascent', label: 'Ascent', fmt: v => Math.round(c.elev(v)) + ' ' + c.elevUnit },
    { key: 'total_calories', label: 'Cal', fmt: v => v },
  ];
  for (const cd of colDefs) {
    if (laps.some(l => l[cd.key] !== undefined)) cols.push(cd);
  }
  let html = '<thead><tr>' + cols.map(c => '<th>' + c.label + '</th>').join('') + '</tr></thead><tbody>';
  laps.forEach((lap, i) => {
    html += '<tr>';
    for (const col of cols) {
      if (col.key === 'lap') { html += '<td>' + (i + 1) + '</td>'; continue; }
      const v = lap[col.key];
      html += '<td>' + (v !== undefined ? col.fmt(v) : '\\u2014') + '</td>';
    }
    html += '</tr>';
  });
  html += '</tbody>';
  tbl.innerHTML = html;
}

function hideLaps() {
  document.getElementById('laps-card').classList.remove('active');
  currentLaps = null;
}

// ---- Zones ----
const HR_ZONE_COLORS = ['#3b82f6','#22c55e','#eab308','#f97316','#ef4444'];
const PW_ZONE_COLORS = ['#93c5fd','#60a5fa','#3b82f6','#2563eb','#1d4ed8','#1e3a8a'];

function updateMaxHRFromAge() {
  const age = parseInt(document.getElementById('zoneAge').value) || 30;
  document.getElementById('zoneMaxHR').value = 220 - age;
  reloadZones();
}

// Attach age → maxHR update
document.getElementById('zoneAge').addEventListener('change', updateMaxHRFromAge);

async function loadZones(fileId) {
  currentFileId = fileId;
  const maxHR = parseInt(document.getElementById('zoneMaxHR').value) || 190;
  const ftp = parseInt(document.getElementById('zoneFTP').value) || 200;
  try {
    const r = await fetch('/zones/' + fileId + '?max_hr=' + maxHR + '&ftp=' + ftp);
    const zones = await r.json();
    if (zones && (zones.hr || zones.power)) {
      document.getElementById('zones-card').classList.add('active');
      renderZones(zones);
    }
  } catch (e) { console.warn('Zones load error:', e); }
}

function reloadZones() {
  if (currentFileId) loadZones(currentFileId);
}

function renderZones(zones) {
  const content = document.getElementById('zoneContent');
  let html = '';
  if (zones.hr) {
    html += '<div class="zone-section"><div class="zone-section-title">Heart Rate Zones (Max HR: ' + zones.hr.max_hr + ' bpm)</div>';
    zones.hr.labels.forEach((label, i) => {
      const pct = zones.hr.pct[i];
      const secs = zones.hr.zones[i];
      html += '<div class="zone-bar-row"><span class="zone-label">' + label + '</span>' +
        '<div class="zone-bar-bg"><div class="zone-bar-fill" style="width:' + Math.max(pct, 1) + '%;background:' + HR_ZONE_COLORS[i] + '">' +
        '<span class="zone-bar-text">' + pct + '%</span></div></div>' +
        '<span class="zone-time">' + fmtDuration(secs) + '</span></div>';
    });
    html += '</div>';
  }
  if (zones.power) {
    html += '<div class="zone-section"><div class="zone-section-title">Power Zones (FTP: ' + zones.power.ftp + ' W)</div>';
    zones.power.labels.forEach((label, i) => {
      const pct = zones.power.pct[i];
      const secs = zones.power.zones[i];
      html += '<div class="zone-bar-row"><span class="zone-label">' + label + '</span>' +
        '<div class="zone-bar-bg"><div class="zone-bar-fill" style="width:' + Math.max(pct, 1) + '%;background:' + PW_ZONE_COLORS[i] + '">' +
        '<span class="zone-bar-text">' + pct + '%</span></div></div>' +
        '<span class="zone-time">' + fmtDuration(secs) + '</span></div>';
    });
    html += '</div>';
  }
  content.innerHTML = html;
}

function hideZones() {
  document.getElementById('zones-card').classList.remove('active');
  document.getElementById('zoneContent').innerHTML = '';
}

// ---- Moments & Group Analysis ----
let momentMarkerLayer = null;
let currentMomentsFileId = null;
let currentMomentsData = {};  // { fileId: { moments, achievements } }

const MOMENT_ICONS = { speed_surge: '\\ud83c\\udfce\\ufe0f', power_spike: '\\u26a1', sprint: '\\ud83c\\udfc3', climb: '\\u26f0\\ufe0f', speed_demon: '\\ud83c\\udfc6' };
const MOMENT_COLORS = { speed_surge: '#e6198a', power_spike: '#d97706', sprint: '#dc2626', climb: '#15803d' };
const MOMENT_LABELS = { speed_surge: 'Speed Surge', power_spike: 'Power Spike', sprint: 'Sprint', climb: 'Climb', speed_demon: 'Speed Demon' };

function getThresholdParams() {
  const ss = parseFloat(document.getElementById('thSpeedSurge').value) || 50;
  const pp = parseFloat(document.getElementById('thPowerSpike').value) || 400;
  const sp = parseFloat(document.getElementById('thSprintPower').value) || 400;
  const sa = parseFloat(document.getElementById('thSprintAccel').value) || 1.0;
  const sd = parseFloat(document.getElementById('thSprintMinDur').value) || 3;
  const cg = parseFloat(document.getElementById('thClimbGradient').value) || 5;
  const cd = parseFloat(document.getElementById('thClimbMinDur').value) || 30;
  const ce = parseFloat(document.getElementById('thClimbMinGain').value) || 10;
  const tw = parseFloat(document.getElementById('thTimeWindow').value) || 60;
  return 'speed_surge=' + (ss / 3.6) + '&power_spike=' + pp + '&sprint_power=' + sp +
    '&sprint_accel=' + sa + '&sprint_min_duration=' + sd +
    '&climb_gradient=' + (cg / 100) + '&climb_min_duration=' + cd +
    '&climb_min_elevation_gain=' + ce + '&time_window=' + tw;
}

function fmtGarminTs(ts, lon) {
  const d = new Date((ts + 631065600) * 1000);
  if (lon != null) {
    // Estimate local time from longitude (offset = lon / 15 hours)
    const offsetH = Math.round(lon / 15);
    const local = new Date(d.getTime() + offsetH * 3600000);
    const sign = offsetH >= 0 ? '+' : '';
    return local.toISOString().replace('T', ' ').replace(/\\.\\d+Z/, '') + ' UTC' + sign + offsetH;
  }
  return d.toISOString().replace('T', ' ').replace(/\\.\\d+Z/, ' UTC');
}

async function loadMoments(fileId) {
  currentMomentsFileId = fileId;
  document.getElementById('thresholds-card').classList.add('active');
  try {
    const r = await fetch('/moments/' + fileId + '?' + getThresholdParams());
    const d = await r.json();
    if (d.error) return;
    currentMomentsData[fileId] = d;
    renderMoments(d);
    addMomentMarkers(d.moments);
    // Check for group moments
    const fids = Object.keys(files);
    if (fids.length >= 2) loadGroupMoments();
  } catch (e) { console.warn('Moments load error:', e); }
}

function renderMoments(data) {
  const el = document.getElementById('momentsContent');
  const moments = data.moments || [];
  const achievements = data.achievements || [];
  if (moments.length === 0 && achievements.length === 0) {
    el.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">No moments detected with current thresholds.</div>';
    document.getElementById('moments-card').classList.add('active');
    return;
  }

  // Count by type
  const counts = {};
  for (const m of moments) { counts[m.type] = (counts[m.type] || 0) + 1; }

  let html = '<div class="moment-badges">';
  for (const [type, count] of Object.entries(counts)) {
    html += '<span class="moment-badge ' + type + '">' + (MOMENT_ICONS[type] || '') + ' ' + count + ' ' + (MOMENT_LABELS[type] || type) + '</span>';
  }
  if (achievements.length > 0) {
    for (const a of achievements) {
      html += '<span class="moment-badge speed_demon">' + MOMENT_ICONS[a.type] + ' ' + (MOMENT_LABELS[a.type] || a.type) + ': ' + a.value + ' km/h</span>';
    }
  }
  html += '</div>';

  // Group moments by type
  window._individualMoments = {};
  const byType = {};
  for (const m of moments) {
    if (!byType[m.type]) byType[m.type] = [];
    byType[m.type].push(m);
  }

  for (const [type, items] of Object.entries(byType)) {
    if (!window._individualMoments[type]) window._individualMoments[type] = [];
    html += '<div class="moment-type-section">';
    html += '<div class="moment-type-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\\'none\\'?\\'block\\':\\'none\\'">';
    html += (MOMENT_ICONS[type] || '') + ' ' + (MOMENT_LABELS[type] || type) + ' (' + items.length + ') \\u25be</div>';
    html += '<ul class="moment-list" style="display:none">';
    for (let mi = 0; mi < items.length; mi++) {
      const m = items[mi];
      const globalIdx = window._individualMoments[type].length;
      window._individualMoments[type].push(m);
      let detail = '';
      if (type === 'speed_surge') detail = (m.value * 3.6).toFixed(1) + ' km/h';
      else if (type === 'power_spike') detail = m.value + ' W';
      else if (type === 'sprint') detail = m.value + ' W peak, ' + m.duration + 's';
      else if (type === 'climb') detail = '+' + m.value + ' m, ' + fmtDuration(m.duration);
      html += '<li class="' + type + '" onclick="selectIndividualMoment(\\'' + type + '\\',' + globalIdx + ',this)"><span>' + detail + '</span><span style="color:var(--text-muted)">' + fmtGarminTs(m.timestamp, m.lon) + '</span>' +
        '<button class="copy-ai-btn moment-copy-btn" title="Copy AI Prompt" onclick="event.stopPropagation();toggleAiMenu(this)">\\ud83d\\udccb<div class="copy-ai-menu">' +
        '<div onclick="event.stopPropagation();copyIndividualMoment(\\'' + type + '\\',' + globalIdx + ',\\'video\\')">\\ud83c\\udfac Video</div>' +
        '<div onclick="event.stopPropagation();copyIndividualMoment(\\'' + type + '\\',' + globalIdx + ',\\'pov\\')">\\ud83c\\udfae POV</div>' +
        '<div onclick="event.stopPropagation();copyIndividualMoment(\\'' + type + '\\',' + globalIdx + ',\\'data\\')">\\ud83d\\udcca Data</div>' +
        '</div></button></li>';
    }
    html += '</ul></div>';
  }

  el.innerHTML = html;
  document.getElementById('moments-card').classList.add('active');
}

function makeMomentIcon(type, isGroup) {
  const color = MOMENT_COLORS[type] || '#666';
  const icon = MOMENT_ICONS[type] || '\\u2022';
  const size = isGroup ? 32 : 22;
  const fontSize = isGroup ? 16 : 12;
  const border = isGroup ? '3px solid #fff' : '2px solid #fff';
  const shadow = isGroup
    ? '0 2px 8px rgba(0,0,0,0.35), 0 0 0 2px ' + color + '40'
    : '0 1px 4px rgba(0,0,0,0.3)';
  return L.divIcon({
    className: '',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
    html: '<div style="width:' + size + 'px;height:' + size + 'px;border-radius:50%;' +
      'background:' + color + ';border:' + border + ';box-shadow:' + shadow + ';' +
      'display:flex;align-items:center;justify-content:center;font-size:' + fontSize + 'px;' +
      'line-height:1;cursor:pointer">' + icon + '</div>'
  });
}

function streetViewLink(lat, lon) {
  return '<div style="margin-top:6px"><a href="https://www.google.com/maps/@' + lat + ',' + lon +
    ',3a,75y,0h,90t/data=!3m6!1e1!3m4!1s!2e0!7i16384!8i8192" target="_blank" rel="noopener" ' +
    'style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:#1a73e8;' +
    'color:#fff;border-radius:4px;font-size:11px;font-weight:500;text-decoration:none">' +
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
    '<circle cx="12" cy="10" r="3"/><path d="M12 2a8 8 0 0 0-8 8c0 5.4 7 11.5 7.3 11.8a1 1 0 0 0 1.4 0C13 21.5 20 15.4 20 10a8 8 0 0 0-8-8z"/>' +
    '</svg>Street View</a>' +
    '<a href="https://www.google.com/maps?q=' + lat + ',' + lon + '" target="_blank" rel="noopener" ' +
    'style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:#fff;' +
    'color:#1a73e8;border:1px solid #dadce0;border-radius:4px;font-size:11px;font-weight:500;' +
    'text-decoration:none;margin-left:4px">' +
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
    '<polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>Maps</a></div>';
}

var highlightMarker = null;
function highlightMomentOnMap(lat, lon, type) {
  if (highlightMarker) { map.removeLayer(highlightMarker); highlightMarker = null; }
  if (!map || lat == null || lon == null) return;
  const color = MOMENT_COLORS[type] || '#2563eb';
  highlightMarker = L.marker([lat, lon], {
    icon: L.divIcon({
      className: '',
      iconSize: [40, 40],
      iconAnchor: [20, 20],
      html: '<div class="moment-highlight-ring" style="width:40px;height:40px;background:' + color + '30;"></div>'
    }),
    interactive: false,
    zIndexOffset: 1000,
  }).addTo(map);
  map.setView([lat, lon], Math.max(map.getZoom(), 15), { animate: true });
  // Open the matching moment marker popup
  if (momentMarkerLayer) {
    momentMarkerLayer.eachLayer(function(layer) {
      if (layer.getLatLng && Math.abs(layer.getLatLng().lat - lat) < 0.00001 && Math.abs(layer.getLatLng().lng - lon) < 0.00001) {
        layer.openPopup();
      }
    });
  }
}

function addMomentMarkers(moments) {
  if (!map) return;
  if (!momentMarkerLayer) {
    momentMarkerLayer = L.layerGroup().addTo(map);
  }
  momentMarkerLayer.clearLayers();
  const hasGeoMoments = moments.some(m => m.lat != null && m.lon != null);
  document.getElementById('momentLegend').style.display = hasGeoMoments ? 'flex' : 'none';
  for (const m of moments) {
    if (m.lat == null || m.lon == null) continue;
    let detail = '';
    if (m.type === 'speed_surge') detail = (m.value * 3.6).toFixed(1) + ' km/h';
    else if (m.type === 'power_spike') detail = m.value + ' W';
    else if (m.type === 'sprint') detail = m.value + ' W peak, ' + m.duration + 's';
    else if (m.type === 'climb') detail = '+' + m.value + ' m, ' + fmtDuration(m.duration);
    let shapeHtml = '';
    if (m.route_shape) shapeHtml = '<div class="route-shape-tag">' + m.route_shape.description + '</div>';
    L.marker([m.lat, m.lon], { icon: makeMomentIcon(m.type, false) })
      .bindPopup('<b>' + (MOMENT_ICONS[m.type] || '') + ' ' + (MOMENT_LABELS[m.type] || m.type) + '</b><br>' + detail + shapeHtml + streetViewLink(m.lat, m.lon),
        { maxWidth: 280 })
      .addTo(momentMarkerLayer);
  }
}

function addGroupMomentMarkers(groupMoments) {
  if (!map || !momentMarkerLayer) return;
  for (const gm of groupMoments) {
    if (gm.lat == null || gm.lon == null) continue;
    let detail = gm.member_count + ' riders';
    if (gm.type === 'speed_surge') detail += ', peak ' + (gm.value * 3.6).toFixed(1) + ' km/h';
    else if (gm.type === 'power_spike') detail += ', peak ' + gm.value + ' W';
    else if (gm.type === 'sprint') detail += ', peak ' + gm.value + ' W';
    else if (gm.type === 'climb') detail += ', +' + gm.value + ' m';
    let shapeHtml = '';
    if (gm.route_shape) shapeHtml = '<div class="route-shape-tag">' + gm.route_shape.description + '</div>';
    L.marker([gm.lat, gm.lon], { icon: makeMomentIcon(gm.type, true) })
      .bindPopup('<b>' + (MOMENT_ICONS[gm.type] || '') + ' Group ' + (MOMENT_LABELS[gm.type] || gm.type) + '</b><br>' + detail + shapeHtml + streetViewLink(gm.lat, gm.lon),
        { maxWidth: 280 })
      .addTo(momentMarkerLayer);
  }
}

async function loadGroupMoments() {
  const fids = Object.keys(files);
  if (fids.length < 2) {
    document.getElementById('group-moments-card').classList.remove('active');
    return;
  }
  const idsParam = fids.map(id => 'ids=' + id).join('&');
  try {
    const r = await fetch('/group-moments?' + idsParam + '&' + getThresholdParams());
    const d = await r.json();
    if (d.error) return;
    renderGroupMoments(d);
    if (d.group_moments && d.group_moments.length > 0) {
      addGroupMomentMarkers(d.group_moments);
    }
  } catch (e) { console.warn('Group moments error:', e); }
}

function renderGroupMoments(data) {
  const el = document.getElementById('groupMomentsContent');
  const gm = data.group_moments || [];
  const ga = data.group_achievements || [];
  if (gm.length === 0 && ga.length === 0) {
    el.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">No group moments found. Riders may not have overlapping events within the time window.</div>';
    document.getElementById('group-moments-card').classList.add('active');
    return;
  }

  window._groupMoments = gm;
  let html = '';
  for (let gi = 0; gi < gm.length; gi++) {
    const m = gm[gi];
    const icon = MOMENT_ICONS[m.type] || '';
    const label = MOMENT_LABELS[m.type] || m.type;
    let valStr = '';
    if (m.type === 'speed_surge') valStr = (m.value * 3.6).toFixed(1) + ' km/h';
    else if (m.type === 'power_spike' || m.type === 'sprint') valStr = m.value + ' W';
    else if (m.type === 'climb') valStr = '+' + m.value + ' m';
    html += '<div class="group-moment-item" onclick="selectGroupMoment(' + gi + ',this)">';
    html += '<div class="group-moment-header">';
    html += '<span class="moment-badge ' + m.type + '">' + icon + ' ' + label + '</span>';
    html += '<span style="font-weight:600">' + m.member_count + ' riders</span>';
    html += '<span style="font-size:0.82rem;color:var(--text-muted)">Peak: ' + valStr + '</span>';
    if (m.duration) html += '<span style="font-size:0.82rem;color:var(--text-muted)">' + fmtDuration(m.duration) + '</span>';
    if (m.timestamp) html += '<span style="font-size:0.75rem;color:var(--text-muted)">' + fmtGarminTs(m.timestamp, m.lon) + '</span>';
    html += '<button class="copy-ai-btn" onclick="event.stopPropagation();toggleAiMenu(this)">\\ud83d\\udccb AI Prompt<div class="copy-ai-menu">' +
      '<div onclick="event.stopPropagation();copyAiPrompt(' + gi + ',\\'video\\')">\\ud83c\\udfac Video Prompt</div>' +
      '<div onclick="event.stopPropagation();copyAiPrompt(' + gi + ',\\'pov\\')">\\ud83c\\udfae POV Prompt</div>' +
      '<div onclick="event.stopPropagation();copyAiPrompt(' + gi + ',\\'data\\')">\\ud83d\\udcca Data Prompt</div></div></button>';
    html += '</div>';
    html += '<div class="group-moment-members">';
    for (const mem of m.members) {
      const fname = files[mem.file_id] ? files[mem.file_id].filename : mem.file_id;
      let memVal = '';
      if (m.type === 'speed_surge') memVal = (mem.value * 3.6).toFixed(1) + ' km/h';
      else if (m.type === 'power_spike' || m.type === 'sprint') memVal = mem.value + ' W';
      else if (m.type === 'climb') memVal = '+' + mem.value + ' m';
      if (mem.timestamp) memVal += ' at ' + fmtGarminTs(mem.timestamp, m.lon || mem.lon);
      html += '<span>' + fname.replace(/\\.fit$/i, '') + ': ' + memVal + '</span>';
    }
    html += '</div></div>';
  }

  window._groupAchievements = ga;
  if (ga.length > 0) {
    html += '<div style="margin-top:12px"><div style="font-weight:600;font-size:0.85rem;margin-bottom:8px">Group Achievements</div>';
    for (let ai = 0; ai < ga.length; ai++) {
      const a = ga[ai];
      html += '<div class="group-moment-item" onclick="selectGroupAchievement(' + ai + ',this)"><div class="group-moment-header">';
      html += '<span class="moment-badge speed_demon">' + (MOMENT_ICONS[a.type] || '\\ud83c\\udfc6') + ' ' + (MOMENT_LABELS[a.type] || a.type) + '</span>';
      html += '<span style="font-weight:600">' + a.member_count + ' riders</span>';
      html += '<button class="copy-ai-btn" onclick="event.stopPropagation();toggleAiMenu(this)">\\ud83d\\udccb AI Prompt<div class="copy-ai-menu">' +
        '<div onclick="event.stopPropagation();copyAchievementPrompt(' + ai + ',\\'video\\')">\\ud83c\\udfac Video Prompt</div>' +
        '<div onclick="event.stopPropagation();copyAchievementPrompt(' + ai + ',\\'pov\\')">\\ud83c\\udfae POV Prompt</div>' +
        '<div onclick="event.stopPropagation();copyAchievementPrompt(' + ai + ',\\'data\\')">\\ud83d\\udcca Data Prompt</div></div></button>';
      html += '</div><div class="group-moment-members">';
      for (const mem of a.members) {
        const fname = files[mem.file_id] ? files[mem.file_id].filename : mem.file_id;
        let memExtra = mem.value + ' km/h';
        if (mem.timestamp) memExtra += ' at ' + fmtGarminTs(mem.timestamp, mem.lon);
        html += '<span>' + fname.replace(/\\.fit$/i, '') + ': ' + memExtra + '</span>';
      }
      html += '</div></div>';
    }
    html += '</div>';
  }

  el.innerHTML = html;
  document.getElementById('group-moments-card').classList.add('active');
}

function redetectMoments() {
  if (momentMarkerLayer) momentMarkerLayer.clearLayers();
  currentMomentsData = {};
  const fids = Object.keys(files);
  if (fids.length === 0) return;
  // Re-analyze each file
  const promises = fids.map(fid => loadMoments(fid));
}

function toggleAiMenu(btn) {
  const menu = btn.querySelector('.copy-ai-menu');
  const wasOpen = menu.classList.contains('show');
  document.querySelectorAll('.copy-ai-menu.show').forEach(m => m.classList.remove('show'));
  if (!wasOpen) menu.classList.add('show');
}
document.addEventListener('click', function() {
  document.querySelectorAll('.copy-ai-menu.show').forEach(m => m.classList.remove('show'));
});

function getTimeOfDay(ts, lon) {
  const d = new Date((ts + 631065600) * 1000);
  const offsetH = (lon != null) ? Math.round(lon / 15) : 0;
  const h = (d.getUTCHours() + offsetH + 24) % 24;
  if (h < 6) return 'pre-dawn';
  if (h < 10) return 'morning';
  if (h < 14) return 'midday';
  if (h < 17) return 'afternoon';
  if (h < 20) return 'evening';
  return 'night';
}

function streetViewUrl(lat, lon) {
  return 'https://www.google.com/maps/@' + lat + ',' + lon + ',3a,75y,0h,90t/data=!3m6!1e1!3m4!1s!2e0!7i16384!8i8192';
}

function mapsUrl(lat, lon) {
  return 'https://www.google.com/maps?q=' + lat + ',' + lon;
}

function momentValStr(type, value) {
  if (type === 'speed_surge') return (value * 3.6).toFixed(1) + ' km/h';
  if (type === 'power_spike' || type === 'sprint') return value + ' W';
  if (type === 'climb') return '+' + value + ' m';
  return String(value);
}

function buildVideoPrompt(m) {
  const label = MOMENT_LABELS[m.type] || m.type;
  const tod = m.timestamp ? getTimeOfDay(m.timestamp, m.lon) : 'daytime';
  const riderCount = m.members ? m.members.length : 1;
  const peak = momentValStr(m.type, m.value);
  let riders = '';
  if (m.members) {
    riders = m.members.map(function(mem) {
      const fname = files[mem.file_id] ? files[mem.file_id].filename.replace(/\\.fit$/i, '') : mem.file_id;
      return fname + ' at ' + momentValStr(m.type, mem.value);
    }).join(', ');
  }
  let prompt = 'Cinematic cycling scene: ' + riderCount + ' cyclist' + (riderCount > 1 ? 's' : '') + ' in a ' + label.toLowerCase() + ' at high intensity.\\n\\n';
  prompt += 'Setting: ' + tod + ' ride';
  if (m.route_shape) prompt += ', ' + m.route_shape.description;
  if (m.lat != null && m.lon != null) prompt += ', at GPS ' + m.lat.toFixed(4) + ', ' + m.lon.toFixed(4);
  prompt += '.\\n\\n';
  prompt += 'Key metrics:\\n';
  prompt += '- Event: ' + label + '\\n';
  prompt += '- Peak: ' + peak + '\\n';
  if (m.timestamp) prompt += '- Time: ' + fmtGarminTs(m.timestamp, m.lon) + '\\n';
  if (m.duration) prompt += '- Duration: ' + m.duration + 's\\n';
  if (riders) prompt += '- Riders: ' + riders + '\\n';
  if (m.lat != null && m.lon != null) {
    prompt += '\\nLocation reference (Street View):\\n' + streetViewUrl(m.lat, m.lon) + '\\n';
  }
  if (m.route_shape) {
    prompt += '\\nRoad: ' + m.route_shape.turn + '. Camera should follow the ' + (m.route_shape.turn === 'straight road' ? 'road ahead' : 'curve direction') + '.';
  }
  prompt += '\\nStyle: Realistic, dynamic camera angles, motion blur on wheels, road-level perspective. Dramatic lighting with ' + tod + ' atmosphere.';
  return prompt;
}

function buildDataPrompt(m) {
  const label = MOMENT_LABELS[m.type] || m.type;
  let prompt = '# Cycling Group Moment Data\\n\\n';
  prompt += '## Event: ' + label + '\\n';
  prompt += '- Type: ' + m.type + '\\n';
  prompt += '- Peak Value: ' + momentValStr(m.type, m.value) + '\\n';
  if (m.duration) prompt += '- Duration: ' + m.duration + 's (' + fmtDuration(m.duration) + ')\\n';
  if (m.member_count) prompt += '- Rider Count: ' + m.member_count + '\\n';
  if (m.timestamp) prompt += '- Timestamp: ' + fmtGarminTs(m.timestamp, m.lon) + '\\n';
  if (m.lat != null && m.lon != null) {
    prompt += '- GPS: ' + m.lat.toFixed(6) + ', ' + m.lon.toFixed(6) + '\\n';
    prompt += '- Google Maps: ' + mapsUrl(m.lat, m.lon) + '\\n';
    prompt += '- Street View: ' + streetViewUrl(m.lat, m.lon) + '\\n';
  }
  if (m.route_shape) {
    prompt += '\\n## Road Shape\\n';
    prompt += '- Road Shape: ' + m.route_shape.description + '\\n';
    prompt += '- Turn: ' + m.route_shape.turn + '\\n';
    prompt += '- Turn Angle: ' + m.route_shape.turn_angle + '°\\n';
    prompt += '- Gradient: ' + m.route_shape.gradient_pct + '%\\n';
    prompt += '- Gradient Type: ' + m.route_shape.gradient_desc + '\\n';
  }
  if (m.members && m.members.length > 0) {
    prompt += '\\n## Rider Breakdown\\n';
    for (const mem of m.members) {
      const fname = files[mem.file_id] ? files[mem.file_id].filename.replace(/\\.fit$/i, '') : mem.file_id;
      prompt += '- ' + fname + ': ' + momentValStr(m.type, mem.value);
      if (mem.timestamp) prompt += ' at ' + fmtGarminTs(mem.timestamp, m.lon || mem.lon);
      if (mem.duration) prompt += ' (' + mem.duration + 's)';
      prompt += '\\n';
    }
  }
  prompt += '\\n## Suggested Uses\\n';
  prompt += '- Video generation (Sora/Runway): Use the metrics and location for a cycling scene\\n';
  prompt += '- Image generation (Midjourney): Create a dramatic cycling moment illustration\\n';
  prompt += '- Analysis (ChatGPT/Claude): Analyze rider performance and group dynamics\\n';
  return prompt;
}

var povHudEnabled = true;

function todAtmosphere(tod) {
  var map = {'pre-dawn':'deep blue sky with soft horizon glow, neon-lit road markings, cool ambient lighting','morning':'warm golden light, vibrant green foliage, bright clean sky with soft clouds','midday':'bright vivid lighting, saturated colors, sharp clean shadows on smooth road','afternoon':'warm amber sunlight, rich golden tones on buildings and trees','evening':'deep orange and purple sky, glowing road markings, warm dramatic lighting','night':'dark sky with stylized stars, glowing street lights, neon road markings illuminated'};
  return map[tod] || 'bright vivid daylight';
}

function describeRoadAhead(rs) {
  if (!rs) return 'smooth open road stretching ahead with bold yellow center line';
  var t = rs.turn || 'straight road';
  var g = rs.gradient_desc || 'flat';
  if (t === 'straight road' && g === 'flat') return 'straight smooth road stretching ahead, bold lane markings, flat terrain';
  if (t === 'straight road') {
    if (g === 'downhill') return 'smooth road descending ahead, bold lane markings flowing downhill';
    if (g === 'slight uphill' || g === 'uphill') return 'road rising ahead, gradient visible on the clean asphalt surface';
    if (g === 'steep climb' || g === 'very steep') return 'steep road climbing sharply upward, bold lane markings ascending';
  }
  var dir = t.indexOf('right') >= 0 ? 'right' : 'left';
  if (t.indexOf('gentle') >= 0) return 'smooth road sweeping gently to the ' + dir + ', bold lane markings curving ahead';
  if (t.indexOf('sharp') >= 0) return 'sharp ' + dir + ' turn ahead, road curving with visible lane markings';
  if (t.indexOf('hairpin') >= 0) return 'tight hairpin ' + dir + ' bend ahead, road folding back on itself';
  if (t === 'U-turn') return 'tight switchback ahead, road doubling back sharply';
  return 'road curving to the ' + dir + ' ahead, bold lane markings guiding the way';
}

function describeGroupRiders(m) {
  if (!m.members || m.members.length <= 1) return '';
  var lines = [];
  var sorted = m.members.slice().sort(function(a, b) { return b.value - a.value; });
  for (var ri = 0; ri < sorted.length; ri++) {
    var mem = sorted[ri];
    var fname = files[mem.file_id] ? files[mem.file_id].filename.replace(/\\.fit$/i, '') : mem.file_id;
    var pos;
    if (ri === 0) pos = 'leading the group ahead';
    else if (ri === sorted.length - 1) pos = 'visible behind, chasing';
    else if (mem.value >= m.value * 0.95) pos = 'riding alongside, matching pace';
    else pos = 'slightly behind, drafting';
    lines.push(fname + ': ' + pos + ' (' + momentValStr(m.type, mem.value) + ')');
  }
  return lines.join('\\n');
}

function buildPovPrompt(m) {
  var label = MOMENT_LABELS[m.type] || m.type;
  var tod = m.timestamp ? getTimeOfDay(m.timestamp, m.lon) : 'daytime';
  var riderCount = m.members ? m.members.length : 1;
  var peak = momentValStr(m.type, m.value);
  var road = describeRoadAhead(m.route_shape);

  var prompt = 'Zwift-style 3D cycling game scene during a ' + label.toLowerCase() + '.\\n\\n';
  prompt += 'Camera: First-person rider\\'s eye view looking forward at the road. The stylized 3D road and world fill the upper 70% of the frame. In the lower portion, the handlebars and the rider\\'s gloved hands gripping the drops are clearly visible';
  if (povHudEnabled) prompt += ', bike computer mounted on stem showing ' + peak;
  prompt += '.\\n\\n';
  prompt += 'World: Stylized 3D game environment — smooth clean roads with bold yellow/white lane markings, vibrant low-poly trees and buildings lining the road, saturated colors, polished game-engine look (similar to Zwift). Not photorealistic.\\n\\n';
  prompt += 'Road ahead: ' + road + '.\\n\\n';
  prompt += 'Atmosphere: ' + todAtmosphere(tod) + '.\\n\\n';

  if (riderCount > 1 && m.members) {
    prompt += 'Other riders on the road:\\n';
    prompt += describeGroupRiders(m) + '\\n\\n';
  }

  prompt += 'Key metrics:\\n';
  prompt += '- Event: ' + label + '\\n';
  prompt += '- Peak: ' + peak + '\\n';
  if (m.timestamp) prompt += '- Time: ' + fmtGarminTs(m.timestamp, m.lon) + '\\n';
  if (m.duration) prompt += '- Duration: ' + m.duration + 's\\n';

  if (povHudEnabled) {
    prompt += '\\nHUD overlay (game-style):\\n';
    prompt += '- Speed: ' + (m.type === 'speed_surge' ? (m.value * 3.6).toFixed(1) + ' km/h' : 'visible on computer') + '\\n';
    if (m.type === 'power_spike' || m.type === 'sprint') prompt += '- Power: ' + m.value + ' W\\n';
    if (m.route_shape) prompt += '- Gradient: ' + m.route_shape.gradient_pct + '%\\n';
    if (m.duration) prompt += '- Timer: ' + fmtDuration(m.duration) + '\\n';
  }

  if (m.lat != null && m.lon != null) {
    prompt += '\\nLocation reference (Street View):\\n' + streetViewUrl(m.lat, m.lon) + '\\n';
  }
  prompt += '\\nStyle: Zwift-style 3D game graphics. Vibrant saturated colors, smooth clean geometry, polished stylized world. Road and scenery fill upper frame, handlebars and gloved hands visible in lower frame. Slight motion blur on road. ' + todAtmosphere(tod) + '.';
  return prompt;
}

function buildIndividualPovPrompt(m, type) {
  var label = MOMENT_LABELS[type] || type;
  var tod = m.timestamp ? getTimeOfDay(m.timestamp, m.lon) : 'daytime';
  var val = momentValStr(type, m.value);
  var fname = currentMomentsFileId && files[currentMomentsFileId] ? files[currentMomentsFileId].filename.replace(/\\.fit$/i, '') : 'Cyclist';
  var road = describeRoadAhead(m.route_shape);

  var prompt = '# POV Cycling Moment: ' + label + '\\n\\n';
  prompt += '- Rider: ' + fname + '\\n';
  prompt += '- Value: ' + val + '\\n';
  if (m.duration) prompt += '- Duration: ' + m.duration + 's (' + fmtDuration(m.duration) + ')\\n';
  if (m.timestamp) prompt += '- Time: ' + fmtGarminTs(m.timestamp, m.lon) + ' (' + tod + ')\\n';
  if (m.lat != null && m.lon != null) {
    prompt += '- GPS: ' + m.lat.toFixed(6) + ', ' + m.lon.toFixed(6) + '\\n';
    prompt += '- Street View: ' + streetViewUrl(m.lat, m.lon) + '\\n';
  }
  if (m.route_shape) prompt += '- Road: ' + m.route_shape.description + '\\n';

  prompt += '\\nZwift-style 3D game scene: First-person view looking forward at a stylized road. Handlebars and gloved hands visible in the lower portion of the frame. ' + road + '. Vibrant low-poly trees and buildings lining the route, bold lane markings on smooth clean road';
  if (povHudEnabled) prompt += ', bike computer reads ' + val;
  prompt += '. ';
  prompt += (type === 'climb' ? 'Road tilting upward, terrain rising around the rider, gradient visible' : type === 'sprint' ? 'Road surface streaking with speed, bold lane markings blurring past' : 'Road stretching ahead into the stylized world, speed building');
  prompt += '. ' + todAtmosphere(tod) + '.';
  if (povHudEnabled && m.route_shape) prompt += ' HUD shows gradient ' + m.route_shape.gradient_pct + '%.';
  return prompt;
}

function buildAchievementPovPrompt(a) {
  var label = MOMENT_LABELS[a.type] || a.type;
  var riderCount = a.members ? a.members.length : 1;
  var earliestTs = null;
  var lat = null, lon = null;
  var riders = '';
  if (a.members) {
    riders = a.members.map(function(mem) {
      var fname = files[mem.file_id] ? files[mem.file_id].filename.replace(/\\.fit$/i, '') : mem.file_id;
      if (mem.timestamp && (earliestTs === null || mem.timestamp < earliestTs)) earliestTs = mem.timestamp;
      if (mem.lat != null) { lat = mem.lat; lon = mem.lon; }
      return fname + ' at ' + mem.value + ' km/h';
    }).join(', ');
  }
  var tod = earliestTs ? getTimeOfDay(earliestTs, lon) : 'daytime';

  var prompt = 'Zwift-style 3D cycling game scene — ' + label.toLowerCase() + ' achieved!\\n\\n';
  prompt += 'Camera: First-person rider\\'s eye view at extreme speed. Stylized 3D road fills the upper frame — bold lane markings streaking past, vibrant game world blurring on the sides. Handlebars and gloved hands in aero tuck visible in the lower portion of the frame';
  if (povHudEnabled) prompt += ', bike computer flashing ' + a.max_value + ' km/h';
  prompt += '.\\n\\n';
  prompt += 'World: Stylized 3D game environment — smooth clean roads, vibrant low-poly scenery, saturated colors, polished game-engine look (similar to Zwift).\\n\\n';
  prompt += 'Atmosphere: ' + todAtmosphere(tod) + '.\\n\\n';

  if (riderCount > 1) {
    prompt += 'Other riders visible: ' + (riderCount - 1) + ' cyclist' + (riderCount > 2 ? 's' : '') + ' on the road — some behind drafting, others being overtaken.\\n';
    if (riders) prompt += 'Riders: ' + riders + '\\n';
    prompt += '\\n';
  }

  prompt += 'Key metrics:\\n';
  prompt += '- Achievement: ' + label + '\\n';
  prompt += '- Max Speed: ' + a.max_value + ' km/h\\n';
  if (earliestTs) prompt += '- Time: ' + fmtGarminTs(earliestTs, lon) + '\\n';
  if (lat != null) prompt += '\\nLocation reference (Street View):\\n' + streetViewUrl(lat, lon) + '\\n';

  if (povHudEnabled) {
    prompt += '\\nHUD overlay: Speed ' + a.max_value + ' km/h (flashing/highlighted as achievement unlocked).\\n';
  }
  prompt += '\\nStyle: Zwift-style 3D game graphics. Vibrant saturated colors, smooth clean geometry, polished stylized world. Road and scenery fill upper frame, handlebars and gloved hands visible in lower frame. Bold markings blurring at speed. ' + todAtmosphere(tod) + '.';
  return prompt;
}

function copyAiPrompt(idx, variant) {
  document.querySelectorAll('.copy-ai-menu.show').forEach(m => m.classList.remove('show'));
  const gm = window._groupMoments || [];
  if (idx < 0 || idx >= gm.length) return;
  const m = gm[idx];
  const text = variant === 'pov' ? buildPovPrompt(m) : variant === 'data' ? buildDataPrompt(m) : buildVideoPrompt(m);
  copyToClipboard(text);
}

function buildIndividualPrompt(m, type) {
  const label = MOMENT_LABELS[type] || type;
  const tod = m.timestamp ? getTimeOfDay(m.timestamp, m.lon) : 'daytime';
  const val = momentValStr(type, m.value);
  const fname = currentMomentsFileId && files[currentMomentsFileId] ? files[currentMomentsFileId].filename.replace(/\\.fit$/i, '') : 'Cyclist';
  let prompt = '# Cycling Moment: ' + label + '\\n\\n';
  prompt += '- Rider: ' + fname + '\\n';
  prompt += '- Value: ' + val + '\\n';
  if (m.duration) prompt += '- Duration: ' + m.duration + 's (' + fmtDuration(m.duration) + ')\\n';
  if (m.timestamp) prompt += '- Time: ' + fmtGarminTs(m.timestamp, m.lon) + ' (' + tod + ')\\n';
  if (m.lat != null && m.lon != null) {
    prompt += '- GPS: ' + m.lat.toFixed(6) + ', ' + m.lon.toFixed(6) + '\\n';
    prompt += '- Street View: ' + streetViewUrl(m.lat, m.lon) + '\\n';
    prompt += '- Maps: ' + mapsUrl(m.lat, m.lon) + '\\n';
  }
  if (m.route_shape) {
    prompt += '- Road: ' + m.route_shape.description + '\\n';
  }
  prompt += '\\nCinematic prompt: A cyclist ' + (type === 'climb' ? 'climbing' : type === 'sprint' ? 'sprinting' : 'surging') + ' at ' + val;
  prompt += ' during a ' + tod + ' ride. Realistic, dynamic angles, motion blur on wheels.';
  return prompt;
}

function buildAchievementVideoPrompt(a) {
  const label = MOMENT_LABELS[a.type] || a.type;
  const riderCount = a.members ? a.members.length : 1;
  let riders = '';
  let earliestTs = null;
  let lat = null, lon = null;
  if (a.members) {
    riders = a.members.map(function(mem) {
      const fname = files[mem.file_id] ? files[mem.file_id].filename.replace(/\\.fit$/i, '') : mem.file_id;
      if (mem.timestamp && (earliestTs === null || mem.timestamp < earliestTs)) earliestTs = mem.timestamp;
      if (mem.lat != null) { lat = mem.lat; lon = mem.lon; }
      return fname + ' at ' + mem.value + ' km/h';
    }).join(', ');
  }
  const tod = earliestTs ? getTimeOfDay(earliestTs, lon) : 'daytime';
  let prompt = 'Cinematic cycling scene: ' + riderCount + ' cyclist' + (riderCount > 1 ? 's' : '') + ' achieving ' + label.toLowerCase() + ' status at extreme speed.\\n\\n';
  prompt += 'Setting: ' + tod + ' ride';
  if (lat != null) prompt += ' at GPS ' + lat.toFixed(4) + ', ' + lon.toFixed(4);
  prompt += '.\\n\\n';
  prompt += 'Key metrics:\\n';
  prompt += '- Achievement: ' + label + '\\n';
  prompt += '- Max Speed: ' + a.max_value + ' km/h\\n';
  if (earliestTs) prompt += '- Time: ' + fmtGarminTs(earliestTs, lon) + '\\n';
  if (riders) prompt += '- Riders: ' + riders + '\\n';
  if (lat != null) prompt += '\\nLocation reference (Street View):\\n' + streetViewUrl(lat, lon) + '\\n';
  prompt += '\\nStyle: Realistic, dynamic camera angles, extreme speed effect, motion blur on wheels and road, road-level perspective. Dramatic ' + tod + ' lighting.';
  return prompt;
}

function buildAchievementDataPrompt(a) {
  const label = MOMENT_LABELS[a.type] || a.type;
  let prompt = '# Cycling Group Achievement Data\\n\\n';
  prompt += '## Achievement: ' + label + '\\n';
  prompt += '- Type: ' + a.type + '\\n';
  prompt += '- Max Value: ' + a.max_value + ' km/h\\n';
  if (a.member_count) prompt += '- Rider Count: ' + a.member_count + '\\n';
  if (a.members && a.members.length > 0) {
    prompt += '\\n## Rider Breakdown\\n';
    for (const mem of a.members) {
      const fname = files[mem.file_id] ? files[mem.file_id].filename.replace(/\\.fit$/i, '') : mem.file_id;
      prompt += '- ' + fname + ': ' + mem.value + ' km/h';
      if (mem.timestamp) prompt += ' at ' + fmtGarminTs(mem.timestamp, mem.lon);
      if (mem.lat != null) {
        prompt += '\\n  - Street View: ' + streetViewUrl(mem.lat, mem.lon);
        prompt += '\\n  - Maps: ' + mapsUrl(mem.lat, mem.lon);
      }
      prompt += '\\n';
    }
  }
  prompt += '\\n## Suggested Uses\\n';
  prompt += '- Video generation (Sora/Runway): Capture the extreme speed moment with dramatic visuals\\n';
  prompt += '- Image generation (Midjourney): Create a speed demon cycling illustration\\n';
  prompt += '- Analysis (ChatGPT/Claude): Compare rider peak speeds and conditions\\n';
  return prompt;
}

function copyAchievementPrompt(idx, variant) {
  document.querySelectorAll('.copy-ai-menu.show').forEach(m => m.classList.remove('show'));
  const ga = window._groupAchievements || [];
  if (idx < 0 || idx >= ga.length) return;
  const a = ga[idx];
  const text = variant === 'pov' ? buildAchievementPovPrompt(a) : variant === 'data' ? buildAchievementDataPrompt(a) : buildAchievementVideoPrompt(a);
  copyToClipboard(text);
}

function clearMomentSelection() {
  document.querySelectorAll('.moment-list li.moment-active').forEach(function(el) { el.classList.remove('moment-active'); });
  document.querySelectorAll('.group-moment-item.moment-active').forEach(function(el) { el.classList.remove('moment-active'); });
  if (highlightMarker && map) { map.removeLayer(highlightMarker); highlightMarker = null; }
}

function selectIndividualMoment(type, idx, el) {
  const moments = (window._individualMoments || {})[type] || [];
  if (idx < 0 || idx >= moments.length) return;
  const m = moments[idx];
  const wasActive = el.classList.contains('moment-active');
  clearMomentSelection();
  if (wasActive) return;
  el.classList.add('moment-active');
  if (m.lat != null && m.lon != null) highlightMomentOnMap(m.lat, m.lon, m.type || type);
}

function selectGroupMoment(idx, el) {
  const gm = window._groupMoments || [];
  if (idx < 0 || idx >= gm.length) return;
  const m = gm[idx];
  const wasActive = el.classList.contains('moment-active');
  clearMomentSelection();
  if (wasActive) return;
  el.classList.add('moment-active');
  if (m.lat != null && m.lon != null) highlightMomentOnMap(m.lat, m.lon, m.type);
}

function selectGroupAchievement(idx, el) {
  const ga = window._groupAchievements || [];
  if (idx < 0 || idx >= ga.length) return;
  const a = ga[idx];
  const wasActive = el.classList.contains('moment-active');
  clearMomentSelection();
  if (wasActive) return;
  el.classList.add('moment-active');
  // Find first member with GPS
  var lat = null, lon = null;
  if (a.members) {
    for (var k = 0; k < a.members.length; k++) {
      if (a.members[k].lat != null && a.members[k].lon != null) {
        lat = a.members[k].lat;
        lon = a.members[k].lon;
        break;
      }
    }
  }
  if (lat != null && lon != null) highlightMomentOnMap(lat, lon, a.type);
}

function copyIndividualMoment(type, idx, variant) {
  document.querySelectorAll('.copy-ai-menu.show').forEach(function(m) { m.classList.remove('show'); });
  const moments = (window._individualMoments || {})[type] || [];
  if (idx < 0 || idx >= moments.length) return;
  const m = moments[idx];
  const text = variant === 'pov' ? buildIndividualPovPrompt(m, type) : variant === 'data' ? buildIndividualPrompt(m, type) : buildIndividualPrompt(m, type);
  copyToClipboard(text);
}

function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function() { showCopyToast('Copied to clipboard!'); }).catch(function() { fallbackCopy(text); });
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); showCopyToast('Copied to clipboard!'); } catch (e) { showCopyToast('Copy failed'); }
  document.body.removeChild(ta);
}

function showCopyToast(msg) {
  const t = document.createElement('div');
  t.className = 'copy-toast'; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(function() { t.remove(); }, 2300);
}

function exportReport() {
  if (currentFileId) {
    window.open('/report/' + currentFileId + '?' + getThresholdParams(), '_blank');
  }
}

function exportGroupReport() {
  const fids = Object.keys(files);
  if (fids.length < 2) return;
  window.open('/report-group?' + fids.map(id => 'ids=' + id).join('&') + '&' + getThresholdParams(), '_blank');
}

function hideMoments() {
  document.getElementById('moments-card').classList.remove('active');
  document.getElementById('group-moments-card').classList.remove('active');
  document.getElementById('thresholds-card').classList.remove('active');
  document.getElementById('momentsContent').innerHTML = '';
  document.getElementById('groupMomentsContent').innerHTML = '';
  if (momentMarkerLayer) momentMarkerLayer.clearLayers();
  document.getElementById('momentLegend').style.display = 'none';
  currentMomentsData = {};
}
</script>
</body>
</html>"""



# ==============================================================================
# File Watcher for Hot Reload
# ==============================================================================

def _file_watcher(filepath, interval=1.0):
    """Watch a file for changes and restart the process when modified."""
    last_mtime = os.stat(filepath).st_mtime
    while True:
        time.sleep(interval)
        try:
            mtime = os.stat(filepath).st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                print(f"\n  [reload] Detected change in {os.path.basename(filepath)}, restarting...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except (OSError, IOError):
            pass


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    # Bind to 0.0.0.0 so it works inside Docker; still accessible via localhost
    host = os.environ.get('HOST', '0.0.0.0')

    reload_enabled = os.environ.get('RELOAD', '1') != '0'

    server = HTTPServer((host, port), FITHandler)

    print(f"\n  FIT Toolkit")
    print(f"  Open in your browser: http://localhost:{port}")
    if reload_enabled:
        print(f"  Hot reload: ON (watching for file changes)")
    print(f"  Press Ctrl+C to stop.\n")

    # Auto-open browser (skip inside Docker where DISPLAY is not set)
    if os.environ.get('DISPLAY') or os.environ.get('BROWSER') or not os.environ.get('container'):
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()

    # Start file watcher for hot reload
    if reload_enabled:
        script_path = os.path.abspath(__file__)
        watcher = threading.Thread(target=_file_watcher, args=(script_path,), daemon=True)
        watcher.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.server_close()
