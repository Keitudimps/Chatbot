import zipfile
import re

# Extract all text from the Word document
with zipfile.ZipFile('data/calendar_2024.docx', 'r') as z:
    with z.open('word/document.xml') as f:
        content = f.read().decode('utf-8')

# Find all text elements
print("=== Looking for date ranges and event keywords ===\n")

# Extract text from RunChild text elements
text_matches = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', content)
for text in text_matches:
    if any(keyword in text.upper() for keyword in ['RECESS', 'GRADUATION', 'BREAK', '-']):
        print(f"Text: {text}")

print("\n=== Looking for DrawingML overlay text ===\n")

# Extract from DrawingML text boxes (alternateContent)
text_matches_drawml = re.findall(r'<a:t>([^<]+)</a:t>', content)
for text in text_matches_drawml:
    if any(keyword in text.upper() for keyword in ['RECESS', 'GRADUATION', 'BREAK', 'VACATION']):
        print(f"DrawML: {text}")

print("\n=== Looking for date range patterns ===\n")
date_ranges = re.findall(r'>([0-9]+-[0-9]+)</[aw]:', content)
for dr in date_ranges:
    print(f"Date range: {dr}")
