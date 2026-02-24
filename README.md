# FIT Toolkit

A self-hosted web toolkit for working with Garmin FIT files. Upload `.fit` files, visualize routes on a map, adjust timestamps, and download the results — all from your browser.

Inspired by [fitfiletools.com](https://www.fitfiletools.com), but runs entirely on your machine with zero external dependencies.

## Features

- **Route map** — visualizes GPS tracks on an interactive OpenStreetMap using Leaflet
- **Time adjuster** — set a new start date/time and shift all timestamps in the file
- **Batch processing** — upload and process multiple FIT files at once
- **Drag & drop** — drop `.fit` files directly into the browser
- **Arrow key time controls** — increment/decrement date and time fields with keyboard arrows, with automatic carry-over (seconds → minutes → hours, etc.)
- **Smart defaults** — time fields pre-fill with current UTC time; one-click buttons to switch between current time and the original file's start time
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

1. **Upload** — drag and drop `.fit` files or click to browse. The route map appears automatically if the file contains GPS data.
2. **Set time** — adjust the "New start" date/time using the input fields or arrow keys. Use the quick buttons to fill in the current time or the original file's time.
3. **Download** — click "Adjust Files" to process. Download the result as a single file or a zip archive for batch jobs.

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
- Large GPS tracks are downsampled to 2,000 points to keep the browser responsive
- Header CRC of `0x0000` is treated as optional per the FIT SDK specification

## License

MIT
