# FIT Toolkit

A self-hosted web toolkit for working with Garmin FIT files. Upload `.fit` files, visualize routes on a map, adjust timestamps, and download the results — all from your browser.

Inspired by [fitfiletools.com](https://www.fitfiletools.com), but runs entirely on your machine with zero external dependencies.

## Features

### Analysis
- **Activity stats** — summary statistics (distance, speed, heart rate, power, cadence, elevation, calories, temperature) with metric/imperial toggle
- **Lap splits** — per-lap breakdown table with duration, distance, speed, heart rate, power, cadence, ascent, and calories (parsed from FIT lap messages)
- **Zone analysis** — heart rate and power zone distribution with configurable thresholds (age-based max HR via 220−age, adjustable FTP) shown as color-coded horizontal bar charts with time-in-zone
- **Time-series charts** — interactive Chart.js graphs for elevation, speed, heart rate, cadence, power, and temperature

### Visualization
- **Route map** — GPS tracks on an interactive OpenStreetMap via Leaflet, with start/finish markers
- **Linked map-chart cursor** — hovering over a chart data point highlights the corresponding GPS position on the map in real time
- **Drag-to-zoom** — click and drag on any chart to zoom into a time range; reset with one click
- **Multi-file overlay** — select multiple files to overlay their routes on the map and compare charts side by side with color-coded legends

### Tools
- **Time adjuster** — set a new start date/time and shift all timestamps in the file
- **Batch processing** — upload and process multiple FIT files at once
- **Drag & drop** — drop `.fit` files directly into the browser
- **Arrow key time controls** — increment/decrement date and time fields with keyboard arrows, with automatic carry-over (seconds → minutes → hours, etc.)
- **Smart defaults** — time fields pre-fill with current UTC time; one-click buttons to switch between current time and the original file's start time

### Development
- **Hot reload** — edit the source and the server restarts automatically (no manual restart needed)
- **Zero dependencies** — built entirely with Python's standard library, no `pip install` required
- **Docker ready** — includes Dockerfile and docker-compose for one-command setup

## Quick Start

### Option 1: Docker (recommended)

```bash
docker compose up --build -d
```

Open [http://localhost:5050](http://localhost:5050) in your browser.

### Option 2: Run directly with Python

```bash
python3 fit_toolkit.py
```

Requires Python 3.8+. No external packages needed.

## Usage

1. **Upload** — drag and drop `.fit` files or click to browse. The route map, stats, lap splits, zone analysis, and charts appear automatically.
2. **Explore** — hover over charts to see the cursor move on the map. Drag to zoom into time ranges. Toggle metric/imperial units. Adjust zone thresholds (age, max HR, FTP).
3. **Compare** — upload multiple files and tick the overlay checkboxes to compare routes and charts side by side.
4. **Adjust time** — set a new start date/time using the input fields or arrow keys. Use the quick buttons to fill in the current time or the original file's time.
5. **Download** — click "Adjust Files" to process. Download the result as a single file or a zip archive for batch jobs.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `5050` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `RELOAD` | `1` | Hot reload (`1` = on, `0` = off) |

## Hot Reload (Development)

The docker-compose file mounts `fit_toolkit.py` as a volume. When you edit and save the file, the server detects the change and restarts automatically — just refresh your browser.

To disable hot reload, set `RELOAD=0` in docker-compose.yml or the environment.

## Project Structure

```
fit-toolkit/
├── fit_toolkit.py        # Single-file application (server + parser + UI)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## How It Works

FIT (Flexible and Interoperable Data Transfer) is a binary format used by Garmin devices to store activity data. The toolkit parses the binary FIT structure directly — no third-party FIT libraries needed.

- **Timestamps** are stored as `uint32` seconds since the Garmin epoch (1989-12-31 00:00:00 UTC). The adjuster locates all timestamp fields (`timestamp`, `time_created`, `start_time`, `local_timestamp`), shifts them by a calculated offset, and recalculates the file's CRC checksums.
- **GPS coordinates** are stored as signed 32-bit integers in semicircle format, converted to decimal degrees (`value × 180 / 2³¹`), and rendered on the map.

## Technical Notes

- FIT binary parsing is done in-place — timestamps are patched directly in the byte array, then CRCs are recalculated
- The map uses Leaflet.js with OpenStreetMap tiles (no API key required)
- Charts use Chart.js 4.5.0 with the chartjs-plugin-zoom (drag-to-zoom) loaded from CDN
- Session stats are extracted from FIT session messages (global message 18); lap splits from lap messages (global message 19); time-series data from record messages (global message 20)
- Large GPS tracks are downsampled to 2,000 points and time-series to 1,000 points to keep the browser responsive
- Header CRC of `0x0000` is treated as optional per the FIT SDK specification

## License

MIT
