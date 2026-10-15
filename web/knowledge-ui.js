/* Shared by the local workbench and the standalone, offline dictionary. */
window.KnowledgeView=(()=>{
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const names={eye:'눈',brow:'눈썹',nose:'코',mouth:'입'};
 function select(records,{query='',part='',topic='',polarity=''}={}){
  const q=query.trim().toLowerCase();
  return records.filter(r=>(!part||r.part===part)&&(!q||JSON.stringify([r.name_ko,r.name_zh,r.morphology_ko,r.interpretation_ko,r.original_section]).toLowerCase().includes(q))&&(!topic||topic in r.historical_topic_assessment)&&(!polarity||(topic?r.historical_topic_assessment[topic]===polarity:Object.values(r.historical_topic_assessment).includes(polarity))));
 }
 function cards(records,requirements){
  const lookup=new Map(requirements.map(r=>[r.record_id,r]));
  return records.map(r=>{
   const req=lookup.get(r.record_id);
   return `<article class="source-card"><div class="row"><h2>${esc(r.name_ko)} <span class="muted">${esc(r.name_zh)}</span></h2><span class="badge">${esc(names[r.part])} · 초벌 해석</span></div><h3>문헌의 형태 묘사</h3><p>${esc(r.morphology_ko)}</p><h3>문헌에서는 어떻게 해석했나</h3><p>${esc(r.interpretation_ko)}</p><div class="row">${Object.entries(r.historical_topic_assessment).map(([k,v])=>`<span class="badge">${esc(k)} · ${esc(v)}</span>`).join('')}</div><p class="small muted">주제별 표시는 문헌의 주장입니다. 개인의 특성·운세를 평가한 결과가 아닙니다.</p><p><strong>근거 구절:</strong> ${r.interpretation_evidence.map(esc).join(' / ')}</p><details><summary>전체 원문과 검토 사항</summary><blockquote>${esc(r.original_section)}</blockquote><p>${esc(r.review_note)}</p><p>형태 판정 한계: ${esc(r.limitation)}</p></details><details><summary>사진 레이블링에 필요한 조건</summary><p>${esc(req?.candidate_morphology_conditions)}</p><ul>${(req?.required_inputs||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><p>${(req?.blockers||[]).map(esc).join(' / ')}</p><p>현재 전통 유형 정답 및 사진→해석 자동 연결: 미확정</p></details><p class="small"><a href="${esc(r.source_url)}" target="_blank" rel="noreferrer">원문 출처 ↗</a>　${esc(r.record_id)}　· 독립 문헌 검수 전</p></article>`;
  }).join('')||'<p>검색 결과가 없습니다.</p>';
 }
 function measurements(spec){return `<p class="note">화면 가이드만으로 실제 얼굴 크기는 측정할 수 없습니다. 비율의 크다/작다 기준도 아직 확정되지 않았습니다.</p><div class="source-list">${spec.features.map(f=>`<article class="source-card"><h3>${esc(f.label_ko)}</h3><span class="badge">${f.measurement_status==='available_proxy'?'계산 가능한 대리 지표 · 구간 미확정':'추가 측정·구현 필요'}</span><p><code>${esc(f.formula)}</code></p><p>${esc(f.limitation)}</p></article>`).join('')}</div>`;}
 return {select,cards,measurements};
})();
