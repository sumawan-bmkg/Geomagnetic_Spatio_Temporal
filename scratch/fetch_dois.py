import urllib.request
import urllib.parse
import json
import time

papers = [
    "Seismic hazard assessment in Indonesia: Progress and challenges",
    "Deep learning for earthquake precursor identification in geomagnetic time series",
    "Graph Neural Networks for Seismological Applications: A Comprehensive Review",
    "Attention-based CNNs for high-precision geophysical signal extraction",
    "Multi-station common mode rejection via PCA for localized crustal deformation monitoring",
    "Solar-terrestrial coupling and its impact on short-term earthquake forecasting",
    "Explainable AI in Geophysics: Interpreting Deep Learning Models for Subsurface Prediction",
    "Robustness of Graph Neural Networks against sensor failures in complex networks",
    "Seismo-electromagnetic phenomena during Solar Cycle 25: A multi-instrumental study",
    "Physics-Informed Machine Learning for Lithospheric Mechanics",
    "Geomagnetic storms and the peak of Solar Cycle 25: A review of ionospheric disturbances",
    "Real-time geophysical informatics for national seismic networks",
    "Mechanical foundation of Dobrovolsky preparation zone in heterogenous crust",
    "Spatio-temporal attention for earthquake signal detection in distributed arrays",
    "Transfer learning and domain adaptation in geophysical time-series classification"
]

results = {}

for title in papers:
    query = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.title={query}&select=DOI,title,score&rows=3"
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            items = data['message']['items']
            if items:
                # Get the highest score
                best_match = items[0]
                results[title] = best_match.get('DOI', 'No DOI found')
            else:
                results[title] = 'Not found'
    except Exception as e:
        results[title] = f'Error: {str(e)}'
    time.sleep(0.5)

for t, doi in results.items():
    print(f"Title: {t}\nDOI: {doi}\n")
