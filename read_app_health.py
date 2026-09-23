import re

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# I want to find the line that displays the trend in app.py
# Let's write a script that just reads and prints the relevant block safely.
lines = text.split('\n')
for i, line in enumerate(lines):
    if 'delta=' in line or 'trend' in line.lower():
        print(f"{i}: {line.encode('ascii', 'ignore').decode()}")
