# Project form answers (copy & paste — all in English)

All links below already point to the martinaravenap/firebridge repository.

---

## Project Name
FireBridge: Harmonizing MODIS and VIIRS Fire Hotspots (Team QuasarAI)

## High-Level Project Summary
FireBridge, by team QuasarAI, is an open-source pipeline and interactive website that harmonizes NASA's MODIS and VIIRS active-fire hotspots. MODIS has built a global fire record since 2000 but is being retired. Its successor, VIIRS, sees about 8× more fires, so stitching them together naively creates fake jumps in fire trends. We matched 472,840 NASA FIRMS detections in space and time, used machine learning to learn how the two sensors differ, and converted VIIRS into a MODIS-equivalent fire product. On days the model never saw, the count bias against real MODIS dropped from +695% to −13%. This keeps the long-term fire record usable for climate, air-quality and disaster-response science.

## Link to Project Demo
https://martinaravenap.github.io/firebridge/FireBridge_slides.pdf

## Link to Final Project
https://martinaravenap.github.io/firebridge/
(Code: https://github.com/martinaravenap/firebridge)

## Detailed Project Description
**What it does:** FireBridge takes the MODIS (1 km) and VIIRS (375 m) active-fire hotspot records from NASA FIRMS and produces a harmonized, MODIS-equivalent 0.25° daily fire grid. An interactive website lets anyone compare the raw sensors with the harmonized product on an animated day-by-day fire map, test a detection simulator (fire power, day/night, viewing angle), explore interactive charts and download the harmonized grid.

**How it works:**
1. Common schema: both datasets are put on the same fields (location, UTC time, fire radiative power, brightness temperatures, confidence class, day/night).
2. Space–time matching: a KD-tree on 3-D Earth coordinates finds the VIIRS hotspots inside each MODIS pixel footprint from the same overpass window. We measured that Aqua's orbit has drifted to a median of 143 minutes after NOAA-21, so we use a 4-hour window. The result is 84,212 matched pairs.
3. Learning the sensor difference: a gradient-boosting classifier estimates the probability that MODIS also detects a given VIIRS hotspot (test AUC 0.68, and it predicts the total number of matches almost exactly). A robust log-log regression compares the fire radiative power of matched pixels.
4. Harmonized grid: a Poisson gradient-boosting model converts VIIRS cell statistics into MODIS-equivalent fire-pixel counts and FRP.
5. Honest validation: we trained on 6 days and tested on 2 held-out days against real MODIS Aqua. The count bias went from +695% to −13% and the mean error per cell-day from 4.6 to 0.8 fire pixels. The correlation at 1° is 0.85.

**Key findings:** Aqua's orbit drift breaks the "simultaneous overpass" assumption. Only 25% of VIIRS hotspots are seen by MODIS, and most of the difference comes from small fires. At night MODIS detects 2–3× more small fires than by day at the same fire power. Agreement also varies strongly by region, from 63% in Australia to 14% in South & SE Asia.

**Benefits:** Scientists and agencies can keep using 25 years of fire trends after MODIS ends, without an artificial jump in the data. The code is open and re-runs with free NASA data.

**What we hope to achieve:** Next we want to run the method on the full FIRMS archive (2012 onwards), add NOAA-20 and Suomi-NPP, and train per-biome models to produce a public long-term harmonized fire record.

**Tools:** Python, pandas, NumPy, SciPy (cKDTree), scikit-learn (HistGradientBoosting, Huber regression), matplotlib, HTML/CSS/JavaScript with Leaflet, Esri basemap, GitHub Pages.

## Use of Super Intelligence (SI)
Yes. We used an AI assistant (Claude) to help write and debug parts of the Python code, the web dashboard and the project text. The team defined the idea and the approach, ran all the code on real NASA FIRMS data, and reviewed the results. All metrics come from actually running the code. No AI-generated images, video or audio were used. The figures are produced by our code from NASA data. The AI assistance is acknowledged in the README, the dashboard footer and the slides.

## NASA Data
- NASA FIRMS (Fire Information for Resource Management System): MODIS Collection 6.1 near-real-time active fire data (Terra and Aqua, MCD14DL), global, 7 days (8–15 Nov 2026). https://firms.modaps.eosdis.nasa.gov/active_fire/
- NASA FIRMS VIIRS 375 m Collection 2 near-real-time active fire data, NOAA-21 (VJ214IMGTDL_NRT), global, same period.

## Space Agency Partner Data and Other References
- No partner-agency data were used in this version.
- Python libraries: pandas, NumPy, SciPy, scikit-learn, matplotlib (open source).
- Leaflet 1.9.4 (open-source map library), Esri World Light Gray basemap.
- Giglio, L. et al. (2016), "The collection 6 MODIS active fire detection algorithm and fire products", Remote Sensing of Environment.
- Schroeder, W. et al. (2014), "The New VIIRS 375 m active fire detection data product", Remote Sensing of Environment.
- AI assistant: Claude (Anthropic), used for code and text assistance.
