import urllib.request
import re

bib_file = r"d:\multi\spatio\Spatio_Precursor_Project\manuscript\sn-bibliography.bib"

with open(bib_file, 'r', encoding='utf-8') as f:
    text = f.read()

dois = re.findall(r'doi=\{([^}]+)\}', text)

print(f"Found {len(dois)} DOIs. Verifying...")

for doi in dois:
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
    try:
        with urllib.request.urlopen(req) as response:
            data = response.read().decode('utf-8')
            if 'message' in data:
                print(f"VALID: {doi}")
            else:
                print(f"INVALID?: {doi}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"NOT FOUND: {doi}")
        else:
            print(f"HTTP ERROR {e.code}: {doi}")
    except Exception as e:
        print(f"ERROR: {doi} - {str(e)}")
