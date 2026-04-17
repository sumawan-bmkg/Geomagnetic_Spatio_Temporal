import json
import re

with open('found_refs.json', 'r') as f:
    data = json.load(f)

bibtex_entries = []
keys = []

def make_bibtex(item, type_ref="article"):
    try:
        authors = item.get('author', [])
        if not authors: return None
        first_author = authors[0].get('family', 'Unknown').replace(' ', '')
        first_author = re.sub(r'[^A-Za-z]', '', first_author)
        
        # Get year
        published = item.get('published', item.get('created', {}))
        date_parts = published.get('date-parts', [[2024]])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
        else:
            year = "2024"
            
        title = item.get('title', [''])[0]
        title_word = ""
        for w in title.split():
            clean_w = re.sub(r'[^A-Za-z]', '', w)
            if len(clean_w) > 4:
                title_word = clean_w.lower()
                break
        
        key = f"{first_author.lower()}{year}{title_word}"
        # Make key unique
        orig_key = key
        counter = 1
        while key in keys:
            key = f"{orig_key}{counter}"
            counter += 1
            
        keys.append(key)
        
        author_str = " and ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in authors])
        journal = item.get('container-title', [''])[0]
        publisher = item.get('publisher', '')
        doi = item.get('DOI', '')
        
        bib = f"@{type_ref}{{{key},\n"
        bib += f"  title={{{title}}},\n"
        bib += f"  author={{{author_str}}},\n"
        if journal: bib += f"  journal={{{journal}}},\n"
        bib += f"  year={{{year}}},\n"
        if publisher: bib += f"  publisher={{{publisher}}},\n"
        if doi: bib += f"  doi={{{doi}}}\n"
        bib += "}\n"
        return bib
    except Exception as e:
        return None

for item in data['springer']:
    b = make_bibtex(item)
    if b: bibtex_entries.append(b)

for item in data['other']:
    b = make_bibtex(item)
    if b: bibtex_entries.append(b)

bib_content = "\n".join(bibtex_entries)
with open('../manuscript/sn-bibliography.bib', 'w', encoding='utf-8') as f:
    f.write(bib_content)

print(f"Generated {len(bibtex_entries)} bibtex entries.")
print("KEYS:")
for k in keys:
    print(k)
