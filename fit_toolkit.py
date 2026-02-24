#!/usr/bin/env python3
"""
FIT Toolkit
A web-based toolkit for working with Garmin FIT files.
Features: timestamp adjustment, route map visualization, batch processing.

Zero dependencies — uses only the Python standard library.

Usage:
    python3 fit_toolkit.py

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
RECORD_MSG_NUM = 20  # Global message number for GPS record messages


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
                gps_fields = {}  # field_def_num -> offset
                for _ in range(nf):
                    fdn, fsz = self.data[pos], self.data[pos+1]
                    pos += 3
                    if fsz == 4 and self._is_timestamp_field(fdn, gmn):
                        ts_fields.append((total, fsz, fdn))
                    # GPS: field 0 = lat, field 1 = lon (sint32, 4 bytes) in record msg
                    if gmn == RECORD_MSG_NUM and fdn in (0, 1) and fsz == 4:
                        gps_fields[fdn] = total
                    total += fsz

                dev_total = 0
                if has_dev:
                    ndf = self.data[pos]; pos += 1
                    for _ in range(ndf):
                        dev_total += self.data[pos+1]; pos += 3

                self.definitions[lmt] = {
                    'endian': endian, 'total_size': total + dev_total,
                    'timestamp_fields': ts_fields,
                    'gps_fields': gps_fields,  # {0: lat_offset, 1: lon_offset}
                }
            else:
                lmt = rh & 0x0F
                if lmt not in self.definitions:
                    raise ValueError(f"Data msg refs undefined type {lmt} at 0x{pos-1:X}")
                defn = self.definitions[lmt]
                self._collect_timestamps(pos, defn)
                self._collect_gps(pos, defn)
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
        self.gps_points.append((round(lat, 6), round(lon, 6)))

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
                self._send_json({'points': pts})
            else:
                self._send_json({'error': 'File not found'}, 404)

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

            uploaded_files[file_id] = {
                'filename': filename,
                'bytes': file_bytes,
                'original_start': start_str,
                'gps_points': fit.gps_points,
            }

            self._send_json({
                'id': file_id,
                'filename': filename,
                'original_start': start_str,
                'size_str': size_str,
                'gps_count': gps_count,
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
  .file-remove {
    background: none; border: none; color: var(--danger); cursor: pointer;
    font-size: 1.2rem; padding: 4px 8px; border-radius: 4px; flex-shrink: 0;
  }
  .file-remove:hover { background: #fef2f2; }
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
  /* Ensure Leaflet tile images render correctly */
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
  @media (max-width: 600px) {
    body { padding: 12px; }
    .time-row { flex-direction: column; align-items: flex-start; }
    .time-row label { min-width: auto; }
    #map { height: 300px; }
  }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js" crossorigin="anonymous"></script>
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
    <div class="map-legend">
      <span><span class="legend-dot" style="background:#22c55e"></span>Start</span>
      <span><span class="legend-dot" style="background:#ef4444"></span>Finish</span>
    </div>
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
}

async function loadRoute(fileId) {
  try {
    const r = await fetch('/gps/' + fileId);
    const d = await r.json();
    if (d.error || !d.points || d.points.length < 2) return;

    // IMPORTANT: make the card visible BEFORE initializing the map
    // Leaflet needs a visible container with real dimensions to render tiles
    const mapCard = document.getElementById('map-card');
    mapCard.classList.add('active');

    // Small delay to let the browser layout the visible container
    await new Promise(resolve => setTimeout(resolve, 50));

    initMap();
    routeLayer.clearLayers();
    markerLayer.clearLayers();

    const latlngs = d.points.map(p => [p[0], p[1]]);

    // Route line
    L.polyline(latlngs, { color: '#2563eb', weight: 3.5, opacity: 0.85 }).addTo(routeLayer);

    // Start marker (green)
    L.circleMarker(latlngs[0], {
      radius: 8, fillColor: '#22c55e', color: '#fff', weight: 2, fillOpacity: 1
    }).bindPopup('Start').addTo(markerLayer);

    // Finish marker (red)
    L.circleMarker(latlngs[latlngs.length - 1], {
      radius: 8, fillColor: '#ef4444', color: '#fff', weight: 2, fillOpacity: 1
    }).bindPopup('Finish').addTo(markerLayer);

    // Fit bounds
    const bounds = L.latLngBounds(latlngs).pad(0.05);
    map.fitBounds(bounds);

    // Info
    document.getElementById('mapInfo').textContent = d.points.length + ' GPS points';

    // Force Leaflet to recalculate container size and reload tiles
    setTimeout(() => { map.invalidateSize(); map.fitBounds(bounds); }, 200);
  } catch (e) {
    console.warn('Map load error:', e);
  }
}

function hideMap() {
  document.getElementById('map-card').classList.remove('active');
  if (routeLayer) routeLayer.clearLayers();
  if (markerLayer) markerLayer.clearLayers();
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
    log('Added: ' + d.filename + ' (start: ' + (d.original_start || 'N/A') + gpsNote + ')', 'info');
    if (Object.keys(files).length === 1 && d.original_start) {
      document.getElementById('originalTime').textContent = d.original_start + ' UTC';
    }
    // Load route map for the first file with GPS data
    if (d.gps_count > 0) loadRoute(d.id);
    updateBtn();
  } catch (e) { log('Upload failed: ' + e.message, 'error'); }
}

function removeFile(id) {
  fetch('/remove/' + id, { method: 'DELETE' });
  delete files[id];
  renderFileList(); updateBtn();
  const keys = Object.keys(files);
  document.getElementById('originalTime').textContent =
    keys.length > 0 ? (files[keys[0]].original_start || '\\u2014') + ' UTC' : '\\u2014';
  // Update map: show first remaining file's route, or hide map
  if (keys.length > 0) {
    const first = files[keys[0]];
    if (first.gps_count > 0) loadRoute(keys[0]); else hideMap();
  } else { hideMap(); }
}

function renderFileList() {
  fileList.innerHTML = '';
  for (const [id, f] of Object.entries(files)) {
    const li = document.createElement('li'); li.className = 'file-item';
    li.innerHTML = '<div class="file-info"><span class="file-name">' + f.filename +
      '</span><span class="file-meta">Start: ' + (f.original_start || 'N/A') +
      ' UTC &middot; ' + f.size_str + '</span></div>' +
      '<button class="file-remove" onclick="removeFile(\\'' + id + '\\')" title="Remove">&times;</button>';
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

    // Cascade carry from seconds up to year
    for (let i = 5; i >= 1; i--) {
      const mx = (i === 2) ? daysInMonth(vals[0], vals[1]) : maxes[i];
      const mn = mins[i];
      if (vals[i] > mx) { vals[i] = mn; if (i > 0) vals[i-1]++; }
      else if (vals[i] < mn) { vals[i] = mx; if (i > 0) vals[i-1]--; }
    }
    // Clamp year
    vals[0] = Math.max(mins[0], Math.min(maxes[0], vals[0]));
    // Clamp day to valid range for the resulting month
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
