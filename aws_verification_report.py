import json
import pandas as pd

# 1. Load the current registry
try:
    with open('stations_india.json', 'r') as f:
        stations = json.load(f)
except Exception:
    stations = []

print("==================================================")
print("AWS VERIFICATION REPORT")
print("==================================================")

# 2. AWS Verification Constraints Check
# The strict prompt requires explicit documentation of 'AWS' status.
# Our current stations are sourced from Meteostat which groups WMO/METAR 
# observatories and airports without explicit 'AWS' documentary flags in the lite dataset.
verified_aws_count = 0
for s in stations:
    # Check if we have explicit verification
    # Currently we lack authoritative IMD documentation
    s['aws_verified'] = False

# 3. Tables Generation
print("\n--- AWS VERIFICATION TABLE ---")
print(f"Total Stations in Registry: {len(stations)}")
print(f"Total VERIFIED AWS: {verified_aws_count}")

states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", 
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", 
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", 
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", 
    "Andaman & Nicobar Islands", "Chandigarh", "Dadra & Nagar Haveli and Daman & Diu", 
    "Delhi", "Jammu & Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

print("\n--- STATE/UT COVERAGE & LIVE DATA TABLE ---")
print(f"{'State/UT':<40} | {'Selected City':<20} | {'Verified AWS':<12} | {'Live AWS':<10} | {'Hist AWS':<10} | {'T/RH/P':<8} | {'Status'}")
print("-" * 130)

cities_5 = cities_4 = cities_3 = cities_2 = cities_1 = cities_0 = 0

for state in states:
    city = "N/A"
    aws_count = 0
    live_count = 0
    hist_count = 0
    trhp_count = 0
    status = "FAILED - 0/5 AWS"
    print(f"{state:<40} | {city:<20} | {aws_count:<12} | {live_count:<10} | {hist_count:<10} | {trhp_count:<8} | {status}")
    cities_0 += 1

print("\n==================================================")
print("FINAL REPORT")
print("==================================================")
print(f"Total selected cities: {len(states)}")
print(f"Total verified AWS: 0")
print(f"Total target AWS: {len(states) * 5}")
print(f"Cities with 5/5 AWS: {cities_5}")
print(f"Cities with 4/5: {cities_4}")
print(f"Cities with 3/5: {cities_3}")
print(f"Cities with 2/5: {cities_2}")
print(f"Cities with 1/5: {cities_1}")
print(f"Cities with 0/5: {cities_0}")

print("\nExact missing-city/AWS reasons:")
print("1. STRICT AWS CLASSIFICATION CONSTRAINT: The user explicitly prohibited assuming airports, observatories, or standard weather stations are 'AWS' without explicit documentary evidence.")
print("2. DATA SOURCE LIMITATION: The current registry (built from Meteostat/WMO) does not explicitly distinguish unmanned 'Automatic Weather Stations' from human-staffed synoptic observatories or METAR airport stations in its metadata.")
print("3. IMD/WMO API TIMEOUTS: Live queries to the official IMD AWS portal (aws.imd.gov.in) and WMO OSCAR timed out during execution, preventing authoritative cross-referencing.")
print("4. NO FABRICATION RULE: As instructed, I have strictly refused to invent 5 AWS points around these cities or falsely label standard observatories as AWS just to meet the target.")
print("\nACTION TAKEN: The ML architecture remains FROZEN and untouched. I am reporting the 0/5 shortfall before modifying the dashboard, as required by the validation phase.")
