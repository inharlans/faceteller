"""Turn the "부위별 문구" sheet of the team's output workbook into frontend/src/regionComments.json.

    python3 scripts/region-comments-from-xlsx.py <workbook.xlsx> [src/regionComments.json]

The sheet has one row for each combination of confirmed shapes of a region: region, id, "name: class / name: (보류) / ...",
reading direction, sentence. The JSON keeps, for each region, the feature names in the sheet's order and the sentence
under the classes joined by "|" (a held feature is "(보류)"). Standard library only.
"""
import json, re, sys, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
SHEET = '부위별 문구'


def rows_of(path, sheet):
    z = zipfile.ZipFile(path)
    strings = [''.join(t.text or '' for t in si.iter(M + 't')) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(M + 'si')] if 'xl/sharedStrings.xml' in z.namelist() else []
    names = [s.get('name') for s in ET.fromstring(z.read('xl/workbook.xml')).find(M + 'sheets')]
    out = []
    for r in ET.fromstring(z.read(f'xl/worksheets/sheet{names.index(sheet) + 1}.xml')).find(M + 'sheetData').findall(M + 'row'):
        cells = {}
        for c in r.findall(M + 'c'):
            n = 0
            for ch in re.match(r'[A-Z]+', c.get('r')).group(0): n = n * 26 + ord(ch) - 64
            v = c.find(M + 'v')
            cells[n] = strings[int(v.text)] if c.get('t') == 's' and v is not None else ''.join(x.text or '' for x in c.iter(M + 't')) if c.get('t') == 'inlineStr' else (v.text if v is not None else '')
        out.append([cells.get(k, '') for k in range(1, 9)])
    return out


def main():
    source = Path(sys.argv[1]); target = Path(sys.argv[2] if len(sys.argv) > 2 else 'src/regionComments.json')
    head, *rows = rows_of(source, SHEET)
    assert head[:5] == ['부위', '조합ID', '확정된 형태 조합', '풀이 방향', '문구'], head
    regions = {}
    for region, _, combo, tone, text, *_ in rows:
        pairs = [p.split(': ', 1) for p in combo.split(' / ')]
        assert all(len(p) == 2 for p in pairs) and text.strip() and tone in ('좋음', '조심', '섞임'), (region, combo)
        entry = regions.setdefault(region, {'features': [n for n, _ in pairs], 'comments': {}})
        assert entry['features'] == [n for n, _ in pairs], (region, combo)
        key = '|'.join(c for _, c in pairs)
        assert key not in entry['comments'], (region, key)
        entry['comments'][key] = [tone, text.strip()]
    target.write_text(json.dumps({'source': f'{source.name} · {SHEET}', 'regions': regions}, ensure_ascii=False, indent=0) + '\n', encoding='utf8')
    print(target, {r: len(e['comments']) for r, e in regions.items()})


if __name__ == '__main__':
    main()
