"""Import only explicit label/reviewer edits, keyed by immutable annotation_id."""
import argparse,json,sys,urllib.request
from pathlib import Path
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1]

def collect_edits(path):
    book=load_workbook(path,read_only=True,data_only=False)
    try:
        sheet=book['레이블 검수'];iterator=sheet.iter_rows(values_only=True);headers=next(iterator)
        required=['annotation_id','수정 레이블','검수자','수정 메모']
        if any(k not in headers for k in required):raise ValueError('Required review columns missing')
        schema=json.loads((ROOT/'data/label_schema.json').read_text(encoding='utf-8'))
        original={r['annotation_id']:r for r in json.loads((ROOT/'data/derived/annotations.json').read_text(encoding='utf-8'))}
        edits=[];seen=set()
        for row in iterator:
            d=dict(zip(headers,row));label=d['수정 레이블']
            if label is None or label=='':continue
            key=d['annotation_id']
            if key in seen:raise ValueError(f'Duplicate annotation ID: {key}')
            seen.add(key)
            if key not in original:raise ValueError(f'Unknown annotation ID: {key}')
            if not str(d['검수자'] or '').strip():raise ValueError(f'Reviewer missing: {key}')
            if label not in ['unknown',*schema['tasks'][original[key]['task']]['classes']]:raise ValueError(f'Invalid label {label}: {key}')
            edits.append({'annotation_id':key,'label':label,'notes':f"[{d['검수자']}] {d['수정 메모'] or ''}"})
        return edits
    finally:book.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('workbook');p.add_argument('--apply',action='store_true');p.add_argument('--server',default='http://127.0.0.1:8765');a=p.parse_args()
    edits=collect_edits(a.workbook);print(f'Validated explicit edits: {len(edits)}')
    if not a.apply:
        print('Dry run only. Add --apply to save via the running local workbench.');return
    # Apply through the same optimistic-concurrency/audit path as the browser.
    for edit in edits:
        image_id=edit['annotation_id'].split(':')[0]
        with urllib.request.urlopen(a.server+'/api/image/'+image_id) as r:details=json.load(r)
        prior=next(r for r in details['annotations'] if r['annotation_id']==edit['annotation_id'])
        edit['revision']=prior['revision']
        req=urllib.request.Request(a.server+'/api/review',data=json.dumps(edit).encode(),headers={'Content-Type':'application/json','X-Face-Workbench':'1'},method='POST')
        with urllib.request.urlopen(req) as r:assert json.load(r)['saved']
    print('Saved. Rebuild the workbook to create a new snapshot; retrain models to reflect changes.')
if __name__=='__main__':main()
