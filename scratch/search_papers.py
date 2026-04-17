import urllib.request
import urllib.parse
import json
import time

queries = [
    "earthquake precursor deep learning",
    "geomagnetic anomaly earthquake",
    "graph neural network earthquake",
    "physics-informed neural network seismic",
    "spatio-temporal deep learning earthquake",
    "ULF geomagnetic precursor",
    "solar cycle geomagnetic storm"
]

results_springer = []
results_other = []
dois_seen = set()

def search_crossref(query, rows=15):
    q = urllib.parse.quote(query)
    # Get more rows and we'll filter
    url = f"https://api.crossref.org/works?query={q}&select=DOI,title,publisher,container-title,author,published&rows={rows}&sort=relevance"
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data['message']['items']
    except Exception as e:
        print(f"Error querying {query}: {e}")
        return []

for q in queries:
    items = search_crossref(q, rows=20)
    for item in items:
        doi = item.get('DOI')
        if not doi or doi in dois_seen: continue
        title = item.get('title', [''])[0].lower()
        if 'erratum' in title or 'correction' in title: continue
        
        publisher = item.get('publisher', '').lower()
        container = item.get('container-title', [''])[0]
        
        # We need 10 Springer Nature / Earth Science Informatics
        if 'springer' in publisher or 'nature' in publisher or 'earth science informatics' in container.lower():
            if len(results_springer) < 10:
                results_springer.append(item)
                dois_seen.add(doi)
        else:
            if len(results_other) < 15:
                # Assuming others are high quality publishers (Elsevier, IEEE, AGU, etc)
                if any(x in publisher for x in ['elsevier', 'ieee', 'american geophysical union', 'wiley', 'oxford']):
                    results_other.append(item)
                    dois_seen.add(doi)
    time.sleep(1)

print(f"Found {len(results_springer)} Springer references.")
for i, r in enumerate(results_springer):
    print(f"S{i+1}: {r.get('title',[''])[0]} - {r.get('container-title',[''])[0]} ({r.get('publisher')})")

print(f"\nFound {len(results_other)} Other Q1 references.")
for i, r in enumerate(results_other):
    print(f"O{i+1}: {r.get('title',[''])[0]} - {r.get('container-title',[''])[0]} ({r.get('publisher')})")

with open('found_refs.json', 'w') as f:
    json.dump({'springer': results_springer, 'other': results_other}, f)
