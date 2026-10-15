"""Self-contained source dictionary; no network or image upload needed."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/knowledge-v2'
def main():
 data={k:json.loads((ROOT/p).read_text(encoding='utf8')) for k,p in [('interpretations','data/interpretation_catalog_v1.json'),('type_requirements','data/type_requirements.json'),('measurements','data/measurement_definitions.json')]}
 topics=['재물','지위·명예','가족·관계','수명·건강','복록','인격·행동','능력에 대한 주장','생활·성취','형벌·재난']
 html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>관상 문헌과 형태 측정 사전</title><style>
 *{box-sizing:border-box}body{font:16px/1.7 'Malgun Gothic',system-ui,sans-serif;color:#1e3041;background:#f3f5f7;margin:0}main{max-width:1080px;margin:auto;padding:36px 24px}h1{font-size:30px;letter-spacing:-1px}h2{font-size:22px;margin:8px 0}h3{font-size:16px;margin-bottom:6px}.muted,.small{color:#566776}.small{font-size:13px}.source-card{padding:24px;background:white;border:1px solid #d8e0e7;border-radius:12px;margin:16px 0}.row,.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.badge{font-size:12px;border:1px solid #cdd7e0;border-radius:20px;padding:3px 10px;background:#f6f8fa}.note{padding:16px;border-left:3px solid #bd9543;background:#fff9ec}.toolbar{position:sticky;top:0;background:#f3f5f7;padding:12px 0;z-index:1}input,select,button{font:inherit;padding:9px 12px;border:1px solid #bbc8d4;border-radius:7px;background:white}input{max-width:100%;flex:1;min-width:180px}button,summary{cursor:pointer}button.active{background:#203e58;color:white}a{color:#1e56b8}blockquote{background:#f2f4f6;margin:10px 0;padding:15px}details{margin:12px 0}code{overflow-wrap:anywhere}#definitions{display:none}@media(max-width:600px){main{padding:20px 14px}.toolbar{position:static}h1{font-size:25px}.source-card{padding:18px}}
 </style><main><p class="muted">문헌 지식 · 측정 기준 / 2026.09.22</p><h1>형태의 정의와 문헌의 해석</h1><p>103개 유형의 원문·해석 요약과 17개 측정 정의를 검색할 수 있습니다. 인터넷 없이도 열립니다.</p><p class="note">103건 모두 AI 초벌 요약이며 독립 문헌 검수 전입니다. 역사 문헌이 주장한 내용과 사진에서 관찰하는 형태를 구분합니다. 전통 유형을 판정하는 CNN이나 개인의 운세를 예측하는 기능은 포함하지 않습니다.</p><div class="row"><button id="dictionary-tab" class="active">문헌 사전 103</button><button id="definitions-tab">측정 정의 17</button></div><section id="dictionary"><div class="toolbar"><input id="query" aria-label="유형 또는 해석 검색" placeholder="귀미, 복록, 코끝…"><select id="part" aria-label="부위"><option value="">모든 부위</option><option value="brow">눈썹</option><option value="eye">눈</option><option value="nose">코</option><option value="mouth">입</option></select><select id="topic" aria-label="해석 주제"><option value="">모든 주제</option>'''+''.join(f'<option>{x}</option>' for x in topics)+'''</select><select id="polarity" aria-label="문헌 평가"><option value="">모든 평가</option><option>길</option><option>흉</option><option>혼합</option><option>조건부</option><option>해석보류</option></select></div><p id="count"></p><div id="records"></div></section><section id="definitions"></section><footer class="small">고전 원저작: 공공영역. Wikisource 전사·편집: 원문별 출처와 재사용 안내 참조. 한국어 요약: 프로젝트 초벌. 참고한 범위는 『고금도서집성』 633·634권의 103항목입니다.</footer></main>'''
 html+='<script>'+ (ROOT/'web/knowledge-ui.js').read_text(encoding='utf8')+'</script>\n'
 html+='<script>const knowledge='+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+''';
 const $=id=>document.getElementById(id);
 function render(){const rows=KnowledgeView.select(knowledge.interpretations.records,{query:$('query').value,part:$('part').value,topic:$('topic').value,polarity:$('polarity').value});$('count').textContent=rows.length+' / 103항목 · 독립 검수 0건';$('records').innerHTML=KnowledgeView.cards(rows,knowledge.type_requirements.records)}
 for(const id of ['query','part','topic','polarity'])$(id).addEventListener(id==='query'?'input':'change',render);
 for(const tab of ['dictionary','definitions'])$(tab+'-tab').onclick=()=>{for(const k of ['dictionary','definitions']){$(k).style.display=k===tab?'block':'none';$(k+'-tab').classList.toggle('active',k===tab)}};
 $('definitions').innerHTML=KnowledgeView.measurements(knowledge.measurements);render();
 </script></html>'''
 (OUT/'관상문헌_측정사전.html').write_text(html,encoding='utf8')
 print(OUT/'관상문헌_측정사전.html')
if __name__=='__main__':main()
