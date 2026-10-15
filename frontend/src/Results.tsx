import {useEffect, useLayoutEffect, useRef, useState} from 'react';
import {PhotoOverlay, type OverlayPreset} from './PhotoOverlay';
import {FEATURES, NOTICE, REGIONS, featureName, percent, points, safeUrl, sourceText, sourceUrl, valenceLabel, type AnalysisResult, type Item, type NamedType, type Passage, type Valence} from './types';

const REASONS: Record<string,string> = {LOW_CONFIDENCE:'형태가 어느 한쪽으로 뚜렷하지 않음', MODEL_NOT_TRAINED:'해당 항목의 모델 준비 전', DEPTH_VIEW_DIRECTIONS_INVALID:'사선 촬영 방향을 확인할 수 없음', REGRESSION_HEAD_HAS_NO_CLASS_CONFIDENCE:'분류 신뢰도가 없는 측정 항목', DEPTH_MEASUREMENT_FAILED:'깊이 측정 실패', QUALITY_FAILED:'입력 품질 기준 미충족', BOUNDARIES_NOT_VALIDATED:'측정 경계 검증 전', DEPTH_SCORE_MISSING:'측정 적합도 없음'};
Object.assign(REASONS, {
  LOW_OR_INVALID_CONFIDENCE:'형태가 어느 한쪽으로 뚜렷하지 않음',CONFIDENCE_NOT_CALIBRATED:'이 항목은 아직 판정 기준 자료가 부족함',
  DEPTH_BOUNDARIES_UNVALIDATED:'깊이 분류 기준을 확인할 수 없음',DEPTH_QUALITY_FAILED:'깊이 측정 품질이 충분하지 않음',
  DEPTH_SCORE_UNAVAILABLE:'깊이 적합도를 계산할 수 없음',MEASUREMENT_UNAVAILABLE:'측정값을 얻지 못함',
  LABEL_UNAVAILABLE:'형태를 확정할 수 없음',BASIS_UNAVAILABLE:'판정 근거가 부족함',
  DEPTH_CONFIDENCE_LOW:'깊이 출력이 불안정함',DEPTH_EXPRESSION_CHANGED:'세 사진에서 중립적인 표정을 유지해야 함',
  DEPTH_VIEWS_INCONSISTENT:'사진 간 얼굴 형상이 일치하지 않음',DEPTH_ESTIMATION_FAILED:'깊이 분석 중 오류가 발생함',
  CNN_HEAD_FAILED:'이 항목의 형태 분류에 실패함',CNN_PART_FAILED:'이 부위의 형태 분류에 실패함',
  CROP_UNAVAILABLE:'이 부위가 충분히 보이지 않음',EAR_VIEW_DIRECTIONS_INVALID:'귀가 보이는 양쪽 사선 사진이 필요함',
  EXPRESSION_BLINK:'눈을 감았거나 가늘게 뜬 상태임',EXPRESSION_FROWN:'눈썹을 찌푸린 표정임',EXPRESSION_SMILE:'웃는 표정이 형태에 영향을 줌',
  FACE_TOO_SMALL:'얼굴이 너무 작게 찍힘',FRONTAL_POSE_INVALID:'정면 사진의 방향을 확인해야 함',
  ILLUMINATION_TOO_DARK:'얼굴이 너무 어둡게 찍힘',ILLUMINATION_TOO_BRIGHT:'얼굴이 너무 밝게 찍힘',LANDMARKS_INVALID:'얼굴 위치를 안정적으로 찾지 못함',MOUTH_OPEN:'입술이 벌어진 상태임',
  PARSING_FAILED:'부위의 경계를 확인할 수 없음',PREDICTED_CLASS_NOT_TRAINED:'이 형태는 학습 자료가 없어 판정하지 않음',
  CLASS_CALIBRATION_INSUFFICIENT:'이 형태는 검증 자료가 부족해 판정하지 않음',
  BROW_MEASUREMENT_UNAVAILABLE:'눈썹 털의 양쪽 끝을 확인하기 어려움',
  FOREHEAD_VISIBILITY_UNAVAILABLE:'이마와 머리선이 충분히 보이지 않음',HAIRLINE_NOT_SUFFICIENTLY_VISIBLE:'머리선이 충분히 보이지 않음',
  HAIRLINE_IRREGULAR_OR_OCCLUDED:'머리선이 가려져 있거나 경계가 불명확함',LOW_HAIR_EDGE_OR_BANGS_UNCERTAIN:'앞머리와 머리선을 구분하기 어려움',
  HAT_OCCLUSION:'모자가 이마를 가리고 있음',FOREHEAD_HEIGHT_IMPLAUSIBLE:'이마 높이를 안정적으로 측정할 수 없음',
  FOREHEAD_WIDTH_NOT_VISIBLE:'이마의 양쪽 경계가 보이지 않음',FOREHEAD_SKIN_NOT_VISIBLE:'이마 피부가 충분히 보이지 않음',
  EAR_NOT_VISIBLE:'귀가 충분히 보이지 않음',EAR_VIEW_OUTSIDE_OBLIQUE_RANGE:'귀를 확인할 사선 각도가 맞지 않음',EAR_GEOMETRY_IMPLAUSIBLE:'귀의 경계를 안정적으로 확인할 수 없음',
  SKIN_MARK_VISIBILITY_INSUFFICIENT:'피부 표시를 확인할 해상도나 가시성이 부족함',SKIN_MARK_VISIBILITY_UNAVAILABLE:'피부 표시를 확인할 수 없음',
  INSUFFICIENT_CLEAR_SKIN_COVERAGE:'가리지 않은 피부 영역이 부족함',LIGHTING_OR_RESOLUTION_UNSUITABLE:'피부 표시를 확인할 조명이나 해상도가 부족함',
  TEXTURE_OR_SHADOW_UNCERTAIN:'피부 표시와 그림자를 구분하기 어려움',LIGHTING_UNSTABLE:'조명에 따라 결과가 달라짐',
});
// Retaking a photo only helps with 'photo' holds; never ask for a retake when the model lacks support.
const UNSUPPORTED = new Set(['CONFIDENCE_NOT_CALIBRATED','CLASS_CALIBRATION_INSUFFICIENT','PREDICTED_CLASS_NOT_TRAINED','MODEL_NOT_TRAINED','BOUNDARIES_NOT_VALIDATED','DEPTH_BOUNDARIES_UNVALIDATED','REGRESSION_HEAD_HAS_NO_CLASS_CONFIDENCE']);
const UNCERTAIN = new Set(['LOW_CONFIDENCE','LOW_OR_INVALID_CONFIDENCE','LABEL_UNAVAILABLE','BASIS_UNAVAILABLE','MEASUREMENT_UNAVAILABLE','DEPTH_SCORE_UNAVAILABLE','DEPTH_SCORE_MISSING','CNN_HEAD_FAILED','CNN_PART_FAILED','DEPTH_ESTIMATION_FAILED']);
export type HoldKind = 'photo' | 'uncertain' | 'unsupported';
export function holdKind(i: Item): HoldKind {
  if (i.hold_reasons.some(r => UNSUPPORTED.has(r))) return 'unsupported';
  return i.hold_reasons.some(r => REASONS[r] && !UNCERTAIN.has(r)) ? 'photo' : 'uncertain';
}
const HOLD_KIND_LABEL: Record<HoldKind,string> = {photo: '사진에서 확인 어려움', uncertain: '형태가 뚜렷하지 않음', unsupported: '아직 판정 자료 부족'};
export function heldSummary(held: Item[]) {
  const count = (k: HoldKind) => held.filter(i => holdKind(i) === k).length;
  return [
    count('photo') && `${count('photo')}개는 사진에서 잘 보이지 않아 보류했습니다. 밝기나 각도를 바꿔 다시 찍으면 확인될 수 있습니다.`,
    count('uncertain') && `${count('uncertain')}개는 형태가 어느 한쪽으로 뚜렷하지 않아 보류했습니다.`,
    count('unsupported') && `${count('unsupported')}개는 아직 판정할 자료가 부족한 항목이라 다시 찍어도 결과가 같습니다.`,
  ].filter((v): v is string => Boolean(v));
}
/** Photo-free snapshot of the in-memory result, saved only when the user asks. */
export function diagnosticPayload(result: AnalysisResult, savedAt: Date) {
  const runtime = (result as unknown as {runtime?: unknown}).runtime;
  return {format: 'faceteller-result-diagnostic-v1', saved_at: savedAt.toISOString(), contains_photos: false,
    status: result.status, items: result.items,
    passages: result.passages.map(p => ({passage_id: p.passage_id, rank: p.rank, G: p.G, evidence_item_ids: p.evidence_item_ids, score_components: p.score_components, reading_title: p.reading?.title, valence: p.reading?.valence})),
    named_types: (result.named_types || []).map(t => ({id: t.id, name: t.name, evidence_item_ids: t.evidence_item_ids})),
    ...(runtime === undefined ? {} : {runtime})};
}
function saveDiagnostic(result: AnalysisResult) {
  const now = new Date(), stamp = now.toISOString().slice(0, 16).replace(/[-:T]/g, '');
  const url = URL.createObjectURL(new Blob([JSON.stringify(diagnosticPayload(result, now), null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `faceteller-result-${stamp}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
export function itemSummary(i: Item) {
  const base = `${featureName(i)} · ${i.status==='HELD' ? '보류' : i.label_name || '확정'}`;
  if(i.status==='HELD') return `${base} — ${[...new Set(i.hold_reasons.map(r=>REASONS[r]||(/^[A-Z_]+$/.test(r)?'형태를 안정적으로 확인할 수 없음':r)))].join(' / ') || '확인할 수 없음'}`;
  const measured=i.measurement?.value??i.measurement?.ratio;
  const ratio=typeof measured==='number' ? measured.toFixed(4) : measured??'없음';
  if(i.kind==='depth') return `${base}${i.measurement ? ` · 상대 비율 ${ratio}` : ''} · 경계점수 ${points(i.measurement_score)}`;
  return `${base}${i.confidence == null ? '' : ` · 분류 신뢰도 ${percent(i.confidence)}`}`;
}
const reviewText = (p: Passage) => {const status=p.source.annotation_status||p.source.review_status;return status ? /auto|ai|unreviewed/i.test(status) ? '번역·풀이는 자동 작성 후 사람 검수 전' : `자료 상태: ${status}` : '자료 상태 정보 없음';};
const evidenceOf = (result: AnalysisResult, ids: string[]) => result.items.filter(i=>i.status==='CONFIRMED' && ids.includes(i.item_id));
const regionsOf = (items: Item[]) => [...new Set(items.map(i=>REGIONS[i.region]||i.region))];
/** Title from the server's reading; otherwise the matched features themselves, never an invented type name. */
export const passageTitle = (p: Passage, evidence: Item[]) => p.reading?.title || evidence.map(i=>`${featureName(i)} ${i.label_name||''}`.trim()).join(' · ') || '문헌 기록';
const DIAGRAM_PINS: [string,number,number,number,number][]=[
  ['forehead',200,125,200,72],['brow',150,185,45,166],['glabella',200,185,355,166],
  ['eye',245,215,355,205],['under_eye',150,240,45,240],['ear',310,235,370,242],
  ['cheekbone',255,270,355,280],['nose',200,270,45,290],['philtrum',200,305,355,320],
  ['mouth',200,340,45,340],['mole',140,295,45,385],['face',200,390,200,455],
];
export function FaceDiagram({result,active}: {result: AnalysisResult; active: string[]}) {
  const confirmed=new Set(result.items.filter(i=>i.status==='CONFIRMED').map(i=>i.region));
  return <div className="facebox"><svg viewBox="0 0 400 490" role="img" aria-label="12개 판정 부위 도해. 실제 사용자 사진이 아닙니다."><g className="face-line"><ellipse cx="200" cy="240" rx="108" ry="155"/><path d="M95 165 Q200 65 305 165 M125 185 Q150 173 177 185 M223 185 Q250 173 275 185 M92 214 Q65 240 96 268 M308 214 Q335 240 304 268"/><ellipse cx="155" cy="215" rx="24" ry="10"/><ellipse cx="245" cy="215" rx="24" ry="10"/><path d="M200 220 L192 274 Q200 282 210 274 M184 278 Q200 292 217 278 M169 332 Q200 319 231 332 Q200 350 169 332Z M170 367 Q200 380 230 367"/></g>{DIAGRAM_PINS.map(([region,x,y,tx,ty])=><g key={region} data-region={region} className={`pin ${confirmed.has(region)?'':'held'} ${active.includes(region)?'active':''}`}><line x1={x} y1={y} x2={tx} y2={ty-8} stroke="currentColor" strokeWidth=".7" opacity=".4"/><circle cx={x} cy={y} r="12"/><text x={tx} y={ty} textAnchor="middle">{REGIONS[region]}</text></g>)}</svg><p className="note center">실선은 판정한 부위, 점선은 보류한 부위입니다.</p></div>;
}
function ValenceTag({value}: {value?: {valence?: Valence; valence_label?: string} | null}) {
  const label=value ? valenceLabel(value) : '';
  return label ? <span className="valence" data-valence={value?.valence||'neutral'}>{label}</span> : null;
}
function Traits({items}: {items: Item[]}) {
  return items.length ? <ul className="traits" aria-label="내 얼굴에서 확인한 생김새">{items.map(i=><li key={i.item_id}><span>{featureName(i)}</span><b>{i.label_name||'확정'}</b></li>)}</ul> : null;
}
function TypeCard({type:t,result,onActive}: {type: NamedType; result: AnalysisResult; onActive:(regions:string[])=>void}) {
  const evidence=evidenceOf(result,t.evidence_item_ids), regions=evidence.map(i=>i.region), url=safeUrl(t.source_url);
  return <article className="type-card" data-valence={t.valence||'neutral'} tabIndex={0} onMouseEnter={()=>onActive(regions)} onMouseLeave={()=>onActive([])} onFocus={()=>onActive(regions)} onBlur={()=>onActive([])}>
    <div className="card-top"><span className="kicker">관상 유형</span><ValenceTag value={t}/></div>
    <h3 className="type-name">{t.name}{t.name_hanja && <span className="hanja" lang="zh">{t.name_hanja}</span>}</h3>
    <Traits items={evidence}/>
    {t.description && <p className="type-desc">{t.description}</p>}
    {t.meaning && <p className="meaning">{t.meaning}</p>}
    {(t.original_text || url) && <details className="source-box"><summary>원문 보기</summary>{t.original_text && <p className="original" lang="zh">{t.original_text}</p>}{url && <p className="source"><a href={url} target="_blank" rel="noopener noreferrer">출처 보기</a></p>}</details>}
  </article>;
}
function PassageCard({passage:p,index,result,onActive}: {passage: Passage;index:number;result:AnalysisResult;onActive:(regions:string[])=>void}) {
  const evidence=evidenceOf(result,p.evidence_item_ids), regions=evidence.map(i=>i.region), url=sourceUrl(p);
  const reading=p.reading||undefined;
  return <article className="passage-card" data-valence={reading?.valence||'none'} tabIndex={0} onMouseEnter={()=>onActive(regions)} onMouseLeave={()=>onActive([])} onFocus={()=>onActive(regions)} onBlur={()=>onActive([])}>
    <div className="card-top"><span className="rank" aria-label={`${p.rank||index+1}번째 기록`}>{p.rank||index+1}</span><ValenceTag value={reading}/>{regionsOf(evidence).length>0 && <span className="partchip">{regionsOf(evidence).join(' · ')}</span>}</div>
    <h3 className="card-title">{passageTitle(p,evidence)}</h3>
    <Traits items={evidence}/>
    <p className="meaning">{reading ? reading.meaning : p.translation}</p>
    <details className="source-box"><summary>원문과 출처</summary>
      <p className="original" lang="zh">{p.original_text}</p>
      {reading && <p className="translation"><span className="label">풀이</span>{p.translation}</p>}
      {p.commentary.trim() && <p className="commentary">{p.commentary}</p>}
      <p className="source">{url?<a href={url} target="_blank" rel="noopener noreferrer">{sourceText(p)}</a>:sourceText(p)}</p>
      <p className="fine">{reviewText(p)}</p>
      <p className="fine score-line">일치 정도 {points(p.G)} · 형태 판정의 확실한 정도를 합친 값이며 정답 확률이 아닙니다.</p>
      {(evidence.length>0 || (p.score_components?.length??0)>0) && <ul className="evidence-list">{p.score_components?.length ? p.score_components.map((c,j)=><li key={`${c.item_id}-${j}`}>{FEATURES[c.item_id]||c.item_id} · {c.label_name} — {c.kind==='depth'?'깊이 경계점수':'분류 신뢰도'} {points(c.score)}</li>) : evidence.map(i=><li key={i.item_id}>{itemSummary(i)}</li>)}</ul>}
    </details>
  </article>;
}
function LicenseNote({passages}: {passages:Passage[]}) {
  return passages.some(p=>typeof p.source.license==='string' && p.source.license.includes('CC BY-SA'))
    ? <span><br/>위키문헌 전사: <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener noreferrer">CC BY-SA 4.0</a> · 판본은 각 기록의 출처 링크에서 확인할 수 있습니다.</span> : null;
}
/** Opt-in view of the in-memory frontal photo; url exists only while show is on. */
export interface PhotoView {available: boolean; show: boolean; url: string | null; hidden: string[]; setShow: (show: boolean) => void; toggle: (id: string) => void; preset: (preset: OverlayPreset) => void}
function PhotoToggle({photo,hint}: {photo: PhotoView; hint?: string}) {
  return <label className="photo-toggle"><input type="checkbox" checked={photo.show} onChange={e=>photo.setShow(e.target.checked)}/><span>사진과 측정선 표시</span>{hint && <small>{hint}</small>}</label>;
}
export function Results({result,photo,onPreview,onReset}: {result:AnalysisResult;photo?:PhotoView;onPreview:()=>void;onReset:()=>void}) {
  const [active,setActive]=useState<string[]>([]);
  const confirmed=result.items.filter(i=>i.status==='CONFIRMED'), held=result.items.filter(i=>i.status==='HELD'), types=result.named_types||[];
  return <section aria-labelledby="result-title"><div className="result-layout"><aside><FaceDiagram result={result} active={active}/><div className="result-counts"><span>판정 <strong>{confirmed.length}</strong></span><span>보류 <strong>{held.length}</strong></span></div><details className="judgments"><summary>항목별 판정 전체 보기</summary>{[...confirmed,...held].map(i=><div className={`judgment ${i.status==='HELD'?'held':''}`} key={i.item_id}>{i.status==='HELD' && <span className="hold-kind" data-kind={holdKind(i)}>{HOLD_KIND_LABEL[holdKind(i)]}</span>}<strong>{REGIONS[i.region]||i.region} · {itemSummary(i)}</strong><p>{i.basis}</p></div>)}<div className="diagnostic"><button className="text-button" onClick={()=>saveDiagnostic(result)}>판정 정보 파일로 저장(JSON)</button><p className="note">사진은 들어가지 않으며 어디에도 자동으로 보내지 않습니다.</p></div></details></aside>
    <div><div className="res-head"><h1 id="result-title">생김새로 찾은 옛 관상 기록</h1><span className="partchip">{result.status==='PARTIAL'?'일부 항목 보류':'모든 항목 판정'}</span></div><p className="res-sub">사진에서 확인한 생김새와 같은 내용을 담은 옛 문헌을 골랐습니다.</p>
      {photo?.available && <section className="photo-panel" aria-label="내 사진과 측정선"><PhotoToggle photo={photo} hint="처음으로 돌아가거나 창을 닫으면 사진이 지워집니다."/>{photo.show && photo.url && <PhotoOverlay url={photo.url} overlay={result.face_overlay} items={result.items} hidden={photo.hidden} onToggle={photo.toggle} onPreset={photo.preset}/>}</section>}
      {types.length>0 && <section className="type-section" aria-labelledby="types-title"><h2 id="types-title" className="section-title">해당하는 관상 유형</h2><div className="cards">{types.map(t=><TypeCard key={t.id} type={t} result={result} onActive={setActive}/>)}</div></section>}
      <section aria-labelledby="passages-title"><h2 id="passages-title" className="section-title">문헌에 남은 기록</h2><div className="cards">{result.passages.length ? result.passages.map((p,i)=><PassageCard key={p.passage_id} passage={p} index={i} result={result} onActive={setActive}/>) : <div className="empty-result"><h3>맞는 문헌 기록이 없습니다</h3><p>생김새 판정은 끝났지만 그 조건을 담은 문헌 조항을 찾지 못했습니다.</p></div>}</div></section>
      {held.length>0 && <div className="held-notice">{heldSummary(held).map(line=><p key={line}>{line}</p>)}<p className="note">항목별 사유는 ‘항목별 판정 전체 보기’에서 볼 수 있습니다.</p></div>}<p className="disclaimer">{result.notice||NOTICE}<LicenseNote passages={result.passages}/></p><div className="actions"><button className="button" onClick={onPreview}>결과지 미리보기</button><button className="button ghost" onClick={onReset}>다시 시작</button></div></div></div></section>;
}

export function majorItems(result:AnalysisResult) {
  const evidence=new Set(result.passages.flatMap(p=>p.evidence_item_ids));
  return evidence.size ? result.items.filter(i=>i.status==='CONFIRMED' && evidence.has(i.item_id)) : result.items.filter(i=>i.status==='CONFIRMED').slice(0,5);
}
const typeLine = (t: NamedType) => `${t.name}${t.name_hanja?`(${t.name_hanja})`:''}${valenceLabel(t)?` · ${valenceLabel(t)}`:''}`;
export function Printable({result,photo,preview,onBack}: {result:AnalysisResult;photo?:PhotoView;preview:boolean;onBack:()=>void}) {
  const content=useRef<HTMLDivElement>(null),sheet=useRef<HTMLDivElement>(null), host=useRef<HTMLDivElement>(null);
  const [fits,setFits]=useState(true), [font,setFont]=useState(11), [scale,setScale]=useState(1);
  const fit = () => {
    if(!sheet.current || !content.current) return false;
    let ok=false, size=11;
    for(;size>=8;size-=.25) {sheet.current.style.fontSize=`${size}pt`; if(content.current.offsetHeight<=277*96/25.4-4 && content.current.scrollWidth<=190*96/25.4+1) {ok=true;break;}}
    setFits(ok); setFont(Math.max(8,size)); sheet.current.style.fontSize=`${Math.max(8,size)}pt`;
    sheet.current.dataset.fits=String(ok); return ok;
  };
  const printPhoto=photo?.show && photo.url ? photo.url : null;
  useLayoutEffect(()=>{fit(); let active=true; void document.fonts.ready.then(()=>{if(active) fit();}); return()=>{active=false;};},[result,printPhoto,photo?.hidden]);
  useEffect(()=>{if(!preview || !host.current) return; const size=()=>setScale(Math.min(1,host.current!.clientWidth/(210*96/25.4))); size(); const observer=new ResizeObserver(size); observer.observe(host.current); return()=>observer.disconnect();},[preview]);
  useEffect(()=>{const before=()=>{fit();};window.addEventListener('beforeprint',before);return()=>window.removeEventListener('beforeprint',before);},[result]);
  const print = (allowOverflow=false) => {const ok=fit();if(ok || allowOverflow) window.print();};
  const types=result.named_types||[];
  return <div className={preview?'print-view':'print-offscreen'}>
    {preview && <div className="preview-controls screen-only"><h1>결과지 미리보기</h1><p role="status" className={!fits?'warning':''}>{fits?`A4 한 장 · ${font}pt · 원문·풀이·출처 전체 수록`:'내용이 A4 한 장을 넘습니다. 글을 줄이지 않고 여러 장으로 인쇄합니다.'}</p><p className="note">해설은 결과 화면에서 볼 수 있습니다. 인쇄할 때 A4 · 배율 100% · 머리글/바닥글 끄기를 권장합니다.</p>{photo?.available && <PhotoToggle photo={photo} hint="선택했을 때만 결과지에 사진이 들어갑니다."/>}<div className="actions"><button className="button" disabled={!fits} onClick={()=>print()}>A4 한 장 인쇄</button>{!fits && <button className="button" onClick={()=>print(true)}>본문 보존 · 여러 장 인쇄</button>}<button className="button ghost" onClick={onBack}>결과로 돌아가기</button></div></div>}
    <div className="sheet-container" ref={host} style={preview?{height:Math.max(1123,content.current?.offsetHeight||0)*scale+30}:undefined}><div id="print-sheet" ref={sheet} style={preview?{transform:`scale(${scale})`,transformOrigin:'top left'}:undefined}><div ref={content} className="print-content">
      <header><h2>페이스텔러 · 옛 관상 기록 결과지</h2>{import.meta.env.MODE==='test' && window.__FACETELLER_TEST__?.fixture && <p>합성 UI 검증 · 실제 얼굴 분석 결과 아님</p>}<p>{result.notice||NOTICE}</p></header>
      <p className="print-overflow">A4 한 장을 넘어 본문을 줄이지 않고 여러 장으로 인쇄한 결과지입니다.</p>
      {printPhoto && <section className="print-photo"><h3>정면 사진과 측정선</h3><PhotoOverlay url={printPhoto} overlay={result.face_overlay} items={result.items} hidden={photo!.hidden} print onLoadSettled={()=>fit()}/></section>}
      <section className="print-summary"><h3>확인한 생김새</h3><p>판정 {result.items.filter(i=>i.status==='CONFIRMED').length}개 · 보류 {result.items.filter(i=>i.status==='HELD').length}개 / 보류 사유는 결과 화면에서 확인</p>{types.length>0 && <p><strong>관상 유형</strong> · {types.map(typeLine).join(' / ')}</p>}{majorItems(result).map(i=><p key={i.item_id}><strong>{itemSummary(i)}</strong> · 근거: {i.basis}</p>)}</section>
      <section><h3>문헌에 남은 기록</h3>{result.passages.length===0 && <p>맞는 문헌 기록이 없습니다.</p>}{result.passages.map((p,i)=><article className="print-passage" key={p.passage_id}><h4>{p.rank||i+1}. {passageTitle(p,evidenceOf(result,p.evidence_item_ids))}{p.reading && valenceLabel(p.reading)?` · ${valenceLabel(p.reading)}`:''}</h4><p className="print-original" lang="zh">{p.original_text}</p><p>{p.translation}</p><p className="print-source">{sourceUrl(p) ? <a href={sourceUrl(p)}>{sourceText(p)}</a> : sourceText(p)} · 일치 정도 {points(p.G)} · {reviewText(p)}</p></article>)}</section>
      <footer>일치 정도와 깊이 경계점수는 정답 확률이 아닙니다. 옛 문헌의 기록을 보여 줄 뿐 개인의 성격·건강·운명을 판단하지 않습니다.<LicenseNote passages={result.passages}/></footer>
    </div></div></div>
  </div>;
}
