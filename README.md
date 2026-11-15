# FireBridge: Harmonizing MODIS and VIIRS Active-Fire Hotspots

**Team QuasarAI** · 
NASA Space Apps Challenge 2026 · Challenge: Harmonizing MODIS and VIIRS hotspots

MODIS (Terra/Aqua, 1 km) has produced a global fire record since 2000, but the satellites are at the end of their lives. VIIRS (Suomi-NPP / NOAA-20 / NOAA-21, 375 m) is the successor, and it sees fire very differently: in our data it reports **~8× more fire detections** than MODIS for the same week. If the two records are stitched together without harmonization, every long-term fire trend shows a fake "jump" when MODIS ends.

FireBridge matches both sensors detection by detection, learns how they differ, and converts VIIRS into a **MODIS-equivalent** product. That lets the long-term fire record continue after MODIS.

👉 **Interactive website:** `docs/index.html` (published with GitHub Pages). It includes an animated day-by-day fire map, a detection simulator, interactive charts and a downloadable harmonized grid.

## Results (global, 8–15 Nov 2026, NASA FIRMS NRT)

| | Raw VIIRS | Harmonized VIIRS |
|---|---|---|
| Count bias vs real MODIS Aqua (held-out days) | **+695 %** | **−13 %** |
| Mean absolute error per 0.25° cell-day | 4.6 pixels | 0.8 pixels |
| Correlation at 1° (held-out days) | 0.84 | 0.85 |

- 472,840 detections analyzed (77,206 MODIS + 395,634 VIIRS NOAA-21), with 84,212 matched MODIS–VIIRS pairs.
- **Finding:** Aqua's orbit has drifted. It now passes a median of **143 min after** NOAA-21, so the "same overpass" assumption used in older studies no longer holds.
- Only 42 % of MODIS fire pixels are confirmed by a VIIRS hotspot, and only 25 % of VIIRS hotspots are seen by MODIS. The difference is driven mostly by small fires (< 1 MW).
- The detectability model has a test AUC of 0.68. It predicts the total number of matched detections almost exactly: 10,187 predicted vs 10,075 observed.
- At night MODIS detects 2–3× more small fires than by day at the same fire power, because there is no solar contamination.

## Seen from space (satellite imagery)

The website also includes a **before/after swipe comparison** of four fire-affected landscapes and a **40-year time-lapse (1984–2026)** of the Amazon frontier in Rondônia, Brazil. All imagery is streamed live from **NASA GIBS**:

| Site | Before | After |
|---|---|---|
| Amazon · Rondônia | Landsat WELD annual composite (30 m), 2001 | HLS Sentinel-2 (30 m), Jul 2025 |
| Camp Fire · Paradise, CA | HLS Sentinel-2, 27 Oct 2018 | HLS Sentinel-2, 6 Dec 2018 |
| Black Summer · SE Australia | MODIS Terra bands 7-2-1 (250 m), 28 Oct 2019 | MODIS Terra 7-2-1, 30 Jan 2020 |
| Pantanal · Brazil | MODIS Terra 7-2-1, 10 Jul 2019 | MODIS Terra 7-2-1, 8 Oct 2020 |

`gibs_pick.py` chooses the clearest date for each site and year by downloading real GIBS tiles and scoring cloud cover and missing data. Colours differ between the old and new sensors (the same harmonization problem FireBridge solves for fire), so these images are used for visual comparison, not to measure area.

## Method

1. **Common schema.** Location, UTC time, FRP, MIR/TIR brightness temperatures, confidence (MODIS 0–100 → low/nominal/high), day/night.
2. **Space–time matching.** We run a KD-tree on 3-D Earth coordinates. For each MODIS pixel we look for VIIRS hotspots inside its footprint (half-diagonal of scan × track + half a VIIRS pixel) within a 4 h window. The window is based on the measured time-offset distribution.
3. **Sensor relationship.**
   - FRP transfer: robust (Huber) log-log regression of MODIS pixel FRP vs the summed VIIRS FRP inside it.
   - Detectability: a gradient-boosting classifier gives P(MODIS also detects this VIIRS hotspot).
4. **Harmonized grid.** We aggregate VIIRS to a 0.25° daily grid. A Poisson gradient-boosting model then maps the cell statistics to MODIS-equivalent pixel counts and FRP.
5. **Validation.** We train on the first 6 days and test on the last 2, which the model never saw, against real MODIS Aqua.

## Run it

```bash
pip install -r requirements.txt
python harmonize.py --modis MODIS_C6_1_Global_7d.csv.gz --viirs J2_VIIRS_C2_Global_7d.csv.gz
```

Data comes from NASA FIRMS: <https://firms.modaps.eosdis.nasa.gov/active_fire/>. The 7-day global CSVs need no login.

The pipeline writes these outputs:
- `outputs/summary.json`: all metrics.
- `outputs/harmonized_grid_daily.csv.gz`: the harmonized product (day, cell, MODIS count/FRP, VIIRS count/FRP, MODIS-equivalent count/FRP).
- `outputs/fig_*.png`: figures.
- `docs/`: the interactive website (`docs/firebridge_harmonized_grid.csv.gz` is the downloadable product).

## Limitations & next steps

- One week of NRT data. Running on the multi-year FIRMS archive (2012 onwards) would make the transfer models much stronger and allow per-biome models.
- Aqua and NOAA-21 no longer overpass at the same time, so part of the disagreement is real fire change during the day.
- Next steps: add Suomi-NPP and NOAA-20, add land-cover features, and use Terra's morning passes.

## Data, tools and AI use

- **NASA data:** FIRMS MODIS Collection 6.1 NRT (MCD14DL) and FIRMS VIIRS 375 m Collection 2 NRT (VJ214IMGTDL, NOAA-21).
- **Tools:** Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, Leaflet, Esri basemap, Inter and Instrument Serif fonts.
- **AI use:** Parts of the code, documentation and text were drafted with the help of an AI assistant (Claude) and reviewed and run by the team. The results come from running the code on real NASA data.
