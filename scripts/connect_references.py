"""Small idempotent source edit kept for reproducibility of the built workbench."""
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'web/index.html'
text=p.read_text(encoding='utf-8')
if 'function referenceHtml(' not in text:
    helper='''function referenceHtml(task,label){
 const links=(project.literature_links||[]).filter(r=>r.task===task&&r.label===label);
 if(!links.length)return '';
 return '<div class="small" style="margin-top:10px">관련 형태 표현: '+links.map(r=>{const entry=project.literature.find(x=>x.name_zh===r.source_name);return entry?`<a href="${esc(entry.source_url)}" target="_blank" rel="noreferrer" title="${esc(r.note)}">${esc(entry.name_ko)} · ${esc(r.evidence)}</a>`:''}).join(' / ')+'<div class="muted">문헌의 일부 형태 표현만 참고합니다. 전통 유형 확정이 아닙니다.</div></div>';
}
'''
    text=text.replace('function toast(t)',helper+'function toast(t)')
    text=text.replace('${esc(spec.limitations)}</details></div>', '${esc(spec.limitations)}</details>${referenceHtml(r.task,r.manual_label)}</div>')
    text=text.replace('${esc(defs[p.task].limitations)}</p></div>', '${esc(defs[p.task].limitations)}</p>${p.label!=="uncertain"?referenceHtml(p.task,p.label):""}</div>')
    p.write_text(text,encoding='utf-8')
print('Literature reference links connected')
