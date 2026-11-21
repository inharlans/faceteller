import {DIRECTIONS, VALENCES, type AnalysisResult, type Direction, type InputMode, type InputSet, type FaceOverlay, type NamedType, type OverlayFeature, type Reading, type ResearchReport} from './types';
export class AnalysisError extends Error {constructor(message: string, public direction?: Direction, public code?: string) {super(message);}}
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
export function validateResult(v: unknown): AnalysisResult {
  if (!record(v) || !['COMPLETE','PARTIAL','REINPUT_REQUIRED'].includes(String(v.status)) || !Array.isArray(v.items) || !Array.isArray(v.passages) || v.passages.length > 5) throw new AnalysisError('분석 결과 형식을 확인할 수 없습니다.');
  for (const i of v.items) if (!record(i) || typeof i.item_id !== 'string' || typeof i.region !== 'string' || typeof i.basis !== 'string' || !['CONFIRMED','HELD'].includes(String(i.status)) || !Array.isArray(i.hold_reasons)) throw new AnalysisError('항목 결과가 올바르지 않습니다.');
  for (const p of v.passages) {
    if (!record(p) || !['original_text','translation','commentary'].every(k => typeof p[k] === 'string') || !record(p.source) || !Array.isArray(p.evidence_item_ids)) throw new AnalysisError('문헌 필수 내용이 누락되었습니다.');
    const s=p.source;
    if (![s.title || s.literature_title,s.chapter,s.locator || s.passage_no].every(value=>typeof value==='string' && value.trim())) throw new AnalysisError('문헌 출처가 누락되었습니다.');
    // Optional editorial enrichment: a malformed reading is dropped, never shown or repaired.
    if (p.reading !== undefined && p.reading !== null && !validReading(p.reading)) delete p.reading;
  }
  if (v.named_types !== undefined) v.named_types = Array.isArray(v.named_types) ? v.named_types.filter(validNamedType) : [];
  if (v.face_overlay !== undefined && v.face_overlay !== null) v.face_overlay = validOverlay(v.face_overlay);
  if (v.research_report !== undefined) v.research_report = validReport(v.research_report);
  return v as unknown as AnalysisResult;
}
const text = (value: unknown) => typeof value === 'string' && value.trim().length > 0;
const optionalText = (value: unknown) => value === undefined || value === null || typeof value === 'string';
const valence = (value: unknown) => VALENCES.includes(value as Reading['valence']);
function validReading(r: unknown): r is Reading {
  return record(r) && text(r.title) && text(r.meaning) && valence(r.valence) && optionalText(r.valence_label)
    && (r.meaning_basis === undefined || r.meaning_basis === 'representative' || r.meaning_basis === 'group')
    && (r.topics === undefined || (Array.isArray(r.topics) && r.topics.every(t => typeof t === 'string')));
}
/** Temporary research report (see types.ts): shown only when every table is well formed; otherwise dropped, never repaired. */
export function validReport(r: unknown): ResearchReport | null {
  if (!record(r) || r.schema !== 'faceteller-research-report-v1' || !record(r.judgements) || !record(r.cnn_check)) return null;
  const num = (n: unknown) => typeof n === 'number' && Number.isFinite(n), numOrNull = (n: unknown) => n === null || n === undefined || num(n);
  const passage = (p: unknown) => record(p) && text(p.representative) && Array.isArray(p.requirement) && p.requirement.length > 0 && p.requirement.every(text) && text(p.valence) && num(p.G)
    && Array.isArray(p.regions) && p.regions.every(text) && record(p.features) && Object.values(p.features).every(f => record(f) && text(f.class) && num(f.confidence) && typeof f.held === 'boolean')
    && optionalText(p.translation) && optionalText(p.source);
  // Optional detail (texts, source place, measured values): a malformed piece is dropped, the rest of the report stays.
  const part = (c: unknown) => record(c) && text(c.label) && num(c.value) && text(c.unit);
  const measure = (m: unknown) => record(m) && text(m.label) && numOrNull(m.value) && text(m.unit) && (m.cuts === undefined || m.cuts === null || (Array.isArray(m.cuts) && m.cuts.every(num)))
    && (m.cut_classes === undefined || m.cut_classes === null || (Array.isArray(m.cut_classes) && m.cut_classes.every(text))) && (m.components === undefined || (Array.isArray(m.components) && m.components.every(part)));
  const sourceDetail = (s: unknown) => record(s) && ['title','title_ko','chapter','section','scan_page','scan_page_end','attributed_to','url','license','text_witness'].every(k => optionalText(s[k]))
    && (s.char_span === undefined || s.char_span === null || (Array.isArray(s.char_span) && s.char_span.length === 2 && s.char_span.every(num)));
  const trimTexts = (p: Record<string, unknown>) => {for (const k of ['original_text','commentary','annotation_status','translation_kind','split_of']) if (!optionalText(p[k])) delete p[k]; if (p.source_detail !== undefined && p.source_detail !== null && !sourceDetail(p.source_detail)) delete p.source_detail;};
  const entry = (e: unknown) => record(e) && text(e.id) && text(e.valence) && Array.isArray(e.topics) && e.topics.every(text) && optionalText(e.translation) && optionalText(e.source);
  const line = (l: unknown) => record(l) && text(l.topic) && text(l.valence) && optionalText(l.translation) && Array.isArray(l.sources) && l.sources.every(text) && Array.isArray(l.entries) && l.entries.length > 0 && l.entries.length <= 100 && l.entries.every(entry);
  const trimPassage = (p: Record<string, unknown>) => {trimTexts(p);
    if (p.lines !== undefined) {if (Array.isArray(p.lines) && p.lines.length <= 50 && p.lines.every(line)) (p.lines as {entries: Record<string, unknown>[]}[]).forEach(l => l.entries.forEach(trimTexts)); else delete p.lines;}
    for (const k of ['entry_count', 'hidden_unfavourable', 'omitted_entries']) if (p[k] !== undefined && !num(p[k])) delete p[k];};
  const trimJudgement = (j: Record<string, unknown>) => {for (const k of ['measure','reference']) if (j[k] !== undefined && j[k] !== null && !measure(j[k])) delete j[k];
    if (j.probabilities !== undefined && j.probabilities !== null && !(record(j.probabilities) && Object.values(j.probabilities).every(num))) delete j.probabilities;};
  const lists = [r.plan_C_selected, r.all_matching, r.blocked_by_hold];
  if (!lists.every(l => Array.isArray(l) && l.length <= 200 && l.every(passage)) || (r.plan_C_selected as unknown[]).length > 5) return null;
  if (!Object.values(r.judgements).every(j => record(j) && text(j.name) && text(j.region) && text(j.status) && optionalText(j.method) && optionalText(j.class) && numOrNull(j.value) && numOrNull(j.confidence) && numOrNull(j.threshold))) return null;
  if (!Array.isArray(r.held_judgements) || !r.held_judgements.every(h => record(h) && text(h.feature) && text(h.name) && text(h.class) && num(h.confidence) && num(h.threshold) && text(h.reason))) return null;
  if (!Object.values(r.cnn_check).every(c => record(c) && Object.values(c).every(e => record(e) && text(e.class) && num(e.confidence)))) return null;
  if (r.notes !== undefined && !(Array.isArray(r.notes) && r.notes.every(text))) return null;
  if (!optionalText(r.summary)) delete r.summary;
  // Sentences from the server: all well formed or none are used (the page then looks them up itself).
  if (r.callouts !== undefined && r.callouts !== null && !(Array.isArray(r.callouts) && r.callouts.length <= 20 && r.callouts.every(c => record(c) && text(c.region) && text(c.text) && optionalText(c.tone)))) delete r.callouts;
  for (const l of lists) for (const p of l as Record<string, unknown>[]) trimPassage(p);
  for (const j of Object.values(r.judgements) as Record<string, unknown>[]) trimJudgement(j);
  return r as unknown as ResearchReport;
}
const MAX_OVERLAY_POINTS = 20000;
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
/** Keep only well-formed measured paths inside the photo; drop the whole overlay if its frame is invalid. */
export function validOverlay(o: unknown): FaceOverlay | null {
  if (!record(o) || !finite(o.width) || !finite(o.height) || o.width <= 0 || o.height <= 0 || !Array.isArray(o.features)) return null;
  const width = o.width, height = o.height, inside = (p: unknown) => Array.isArray(p) && p.length === 2 && finite(p[0]) && finite(p[1])
    && p[0] >= 0 && p[0] <= width && p[1] >= 0 && p[1] <= height;
  let budget = MAX_OVERLAY_POINTS;
  const features = o.features.filter((f): f is OverlayFeature => {
    if (!record(f) || !text(f.item_id) || !text(f.label) || !['landmark', 'segmentation'].includes(String(f.source)) || !Array.isArray(f.paths) || !f.paths.length) return false;
    if (!f.paths.every(path => Array.isArray(path) && path.length >= 2 && path.every(inside))) return false;
    if ((f.value !== undefined && f.value !== null && !finite(f.value)) || !optionalText(f.unit)) return false;
    budget -= f.paths.reduce((n: number, path: unknown[]) => n + path.length, 0);
    return budget >= 0;
  });
  return {width, height, features};
}
function validNamedType(t: unknown): t is NamedType {
  return record(t) && text(t.id) && text(t.name) && Array.isArray(t.evidence_item_ids) && t.evidence_item_ids.every(i => typeof i === 'string')
    && (t.valence === undefined || t.valence === null || valence(t.valence))
    && ['name_hanja','description','meaning','valence_label','source_url','original_text'].every(k => optionalText(t[k]));
}
export async function readNDJSON(response: Response, progress: (stage: string) => void, signal?: AbortSignal): Promise<AnalysisResult> {
  if (!response.body) throw new AnalysisError('응답 본문이 없습니다.');
  if (!response.headers.get('content-type')?.includes('application/x-ndjson')) {
    let error: Record<string, unknown> = {};
    if (!response.ok) {try {const v = await response.json(); error = v.detail || v;} catch { /* no body data logged */ }}
    throw new AnalysisError(String(error.message || `서버 응답을 확인할 수 없습니다. (HTTP ${response.status})`), error.direction as Direction, error.code as string);
  }
  const reader = response.body.getReader(), decoder = new TextDecoder('utf-8', {fatal: true});
  let buffer = '', total = 0, result: AnalysisResult | undefined;
  const onAbort = () => {void reader.cancel();};
  signal?.addEventListener('abort', onAbort, {once: true});
  const consume = (line: string) => {
    if (!line.trim()) return;
    if (result) throw new AnalysisError('최종 결과 뒤에 잘못된 데이터가 도착했습니다.');
    let e: Record<string, unknown>;
    try {e = JSON.parse(line);} catch {throw new AnalysisError('분석 응답을 읽지 못했습니다.');}
    if (!record(e)) throw new AnalysisError('분석 응답 형식 오류');
    if (e.type === 'error') throw new AnalysisError(String(e.message || '분석 중 오류가 발생했습니다.'), e.direction as Direction, e.code as string);
    if (!response.ok) throw new AnalysisError(`서버 요청 실패 (HTTP ${response.status})`);
    if (e.type === 'progress' && typeof e.stage === 'string') progress(e.stage);
    else if (e.type === 'result') result = validateResult(e.data);
    else throw new AnalysisError('알 수 없는 분석 이벤트입니다.');
  };
  try {
    while (true) {
      signal?.throwIfAborted();
      const {value, done} = await reader.read();
      signal?.throwIfAborted();
      total += value?.byteLength || 0;
      if (total > 4_000_000) throw new AnalysisError('응답 크기 제한을 초과했습니다.');
      buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
      let at: number;
      while ((at = buffer.indexOf('\n')) >= 0) {consume(buffer.slice(0, at)); buffer = buffer.slice(at + 1);}
      if (done) {consume(buffer); break;}
    }
    if (!result) throw new AnalysisError('최종 결과 전에 연결이 끊어졌습니다. 사진을 다시 입력해 주세요.');
    return result;
  } finally {signal?.removeEventListener('abort', onAbort); await reader.cancel().catch(() => {}); reader.releaseLock();}
}
export async function submitAnalysis(inputs: InputSet, mode: InputMode, signal: AbortSignal, progress: (stage: string) => void) {
  const body = new FormData();
  for (const d of DIRECTIONS) body.append(d, inputs[d], `${d}.${inputs[d].type === 'image/png' ? 'png' : inputs[d].type === 'image/webp' ? 'webp' : 'jpg'}`);
  body.append('input_mode', mode);
  return readNDJSON(await fetch('/api/v1/analysis', {method: 'POST', body, signal, cache: 'no-store', credentials: 'same-origin'}), progress, signal);
}
