"""Download district-wise, season-wise crop Area, Production and Yield (APY) statistics.

The statistics are published by the Directorate of Economics and Statistics (DES),
Ministry of Agriculture and Farmers Welfare (https://data.desagri.gov.in). This script
fetches the full 1997-98 to 2022-23 extract as a single CSV from the India Data Portal.
"""
import pathlib, urllib.request

URL = ("https://ckandev.indiadataportal.com/dataset/f2bbc28c-6c7c-462b-9064-ea4c4213d466/"
       "resource/f980409d-49a2-42ae-9eb0-182365005c04/download/crop-wise-area-production-yield.csv")
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "crop" / "crop-wise-area-production-yield.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, OUT)
    print("saved", OUT, OUT.stat().st_size, "bytes")
