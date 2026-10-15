import {DIRECTIONS, VALENCES, type AnalysisResult, type Direction, type InputMode, type InputSet, type FaceOverlay, type NamedType, type OverlayFeature, type Reading} from './types';
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
  return v as unknown as AnalysisResult;
}
const text = (value: unknown) => typeof value === 'string' && value.trim().length > 0;
const optionalText = (value: unknown) => value === undefined || value === null || typeof value === 'string';
const valence = (value: unknown) => VALENCES.includes(value as Reading['valence']);
function validReading(r: unknown): r is Reading {
  return record(r) && text(r.title) && text(r.meaning) && valence(r.valence) && optionalText(r.valence_label)
    && (r.topics === undefined || (Array.isArray(r.topics) && r.topics.every(t => typeof t === 'string')));
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
