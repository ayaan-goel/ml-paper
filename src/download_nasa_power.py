"""Download daily NASA POWER weather for Gujarat districts (1997-2023).

One request per district at the district headquarters' approximate
coordinates. District names match the crop-statistics file (all 33
current districts, including those created in 2013). Raw responses are saved
unchanged in data/raw/nasa_power/, with a log of the request parameters.
"""
import json, time, datetime, pathlib, urllib.request, urllib.parse

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa_power"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN,WS2M"
START, END = "19970101", "20231231"

# District headquarters, approximate (lat, lon). NASA POWER meteorology is
# on a ~0.5 degree grid, so this precision is sufficient.
DISTRICTS = {
    "Ahmedabad": (23.03, 72.58), "Amreli": (21.60, 71.22), "Anand": (22.56, 72.95),
    "Banas Kantha": (24.17, 72.43), "Bharuch": (21.71, 72.98), "Bhavnagar": (21.76, 72.15),
    "Dahod": (22.84, 74.25), "Dangs": (20.75, 73.69), "Gandhinagar": (23.22, 72.65),
    "Jamnagar": (22.47, 70.06), "Junagadh": (21.52, 70.46), "Kheda": (22.75, 72.68),
    "Kachchh": (23.25, 69.67), "Mahesana": (23.60, 72.40), "Narmada": (21.87, 73.50),
    "Navsari": (20.95, 72.92), "Panch Mahals": (22.77, 73.61), "Patan": (23.85, 72.13),
    "Porbandar": (21.64, 69.60), "Rajkot": (22.30, 70.80), "Sabar Kantha": (23.60, 72.97),
    "Surat": (21.17, 72.83), "Surendranagar": (22.73, 71.64), "Tapi": (21.12, 73.40),
    "Vadodara": (22.31, 73.18), "Valsad": (20.61, 72.93),
    # Districts created in 2013
    "Arvalli": (23.47, 73.30), "Botad": (22.17, 71.67), "Chhotaudepur": (22.31, 74.01),
    "Devbhumi Dwarka": (22.20, 69.65), "Gir Somnath": (20.91, 70.37),
    "Mahisagar": (23.13, 73.61), "Morbi": (22.82, 70.84),
}

log = []
for name, (lat, lon) in DISTRICTS.items():
    path = OUT / f"{name}.csv"
    if path.exists() and END[:4] in path.read_text(encoding="utf-8")[:300]:
        print("skip", name); continue
    q = urllib.parse.urlencode({"parameters": PARAMS, "community": "AG", "latitude": lat,
                                "longitude": lon, "start": START, "end": END, "format": "CSV"})
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + q
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(url, timeout=180).read()
            break
        except Exception as e:
            print(name, "retry", attempt, e); time.sleep(5)
    else:
        raise SystemExit(f"failed: {name}")
    path.write_bytes(data)
    log.append({"district": name, "lat": lat, "lon": lon, "url": url,
                "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "bytes": len(data)})
    print("ok", name, len(data))
    time.sleep(1)

if log:
    with open(OUT / "_download_log.json", "a", encoding="utf-8") as f:
        for r in log: f.write(json.dumps(r) + "\n")
