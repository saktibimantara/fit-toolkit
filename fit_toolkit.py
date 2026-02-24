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
  #stats-card, #charts-card, #laps-card, #zones-card { display: none; }
  #stats-card.active, #charts-card.active, #laps-card.active, #zones-card.active { display: block; }
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

  @media (max-width: 600px) {
    body { padding: 12px; }
    .time-row { flex-direction: column; align-items: flex-start; }
    .time-row label { min-width: auto; }
    #map { height: 300px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
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
      <div class="unit-toggle">
        <button class="active" onclick="setUnits('metric')">Metric</button>
        <button onclick="setUnits('imperial')">Imperial</button>
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
    if (d.gps_count > 0) loadRoute(d.id);
    if (d.has_stats) loadStats(d.id);
    if (d.has_laps) loadLaps(d.id);
    updateBtn();
  } catch (e) { log('Upload failed: ' + e.message, 'error'); }
}

function removeFile(id) {
  fetch('/remove/' + id, { method: 'DELETE' });
  delete files[id];
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
  } else { hideMap(); hideStats(); hideLaps(); hideZones(); hideSimilarity(); }
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
