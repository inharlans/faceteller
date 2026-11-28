import type {Anchors} from './FaceReport';

// Character portrait for the report sheet (2026-11-21). The frontal photo goes to the lab GPU image server through
// EC2 nginx (/api/v1/stylize/ -> SSH tunnel -> FLUX.1 Kontext) and comes back as a PNG drawing. It is display only:
// every measurement is made on the original photo. Nothing is stored; the drawing lives in an object URL.
export const PORTRAIT_TIMEOUT_MS = 300_000;   // about 60 s per drawing on the test GPU; long enough to wait for a few drawings queued before ours
// The progress screen stays up while the drawing is made: the five stages step on every STAGE_STEP_MS, the last one
// stays active until the drawing is in, and after PROGRESS_WAIT_MS the results are shown anyway (the drawing joins later).
export const STAGE_STEP_MS = 11_000, PROGRESS_WAIT_MS = 100_000, FINISH_MS = 800;   // FINISH_MS: gauges run up to 100 before the results

/** Where the drawing of one request stands (GET /progress/<id> of the image server, 2026-11-21): queued while another
 *  drawing is made (the request is then retried with the same id), drawing with step of steps, done once the PNG is ready. */
export interface DrawProgress {state: 'queued' | 'drawing' | 'done' | 'unknown'; step: number; steps: number}
export const PROGRESS_POLL_MS = 500;
export function newRequestId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
/** null when the server cannot say (older server, network): the page then falls back to the time estimate. */
export async function portraitProgress(id: string, signal?: AbortSignal): Promise<DrawProgress | null> {
  try {
    const r = await fetch(`/api/v1/stylize/progress/${encodeURIComponent(id)}`, {cache: 'no-store', credentials: 'same-origin', signal});
    const j = r.ok ? await r.json() : null;
    return j && typeof j.state === 'string' ? {state: j.state, step: Number(j.step) || 0, steps: Number(j.steps) || 0} : null;
  } catch { return null; }
}
/** Gauge of stage i when the whole drawing (0..1) is spread evenly over the stages; shown caps how far the analysis is. */
export function drawPercent(i: number, shown: number, fraction: number, stages: number): number {
  if (i < shown) return 100;
  if (i > shown) return 0;
  return Math.max(0, Math.min(99, Math.floor((fraction * stages - i) * 100)));
}

/** Length of one progress stage, from how long the image server took for its last drawing (spread over the stages),
 *  so the gauges finish about when the drawing does. STAGE_STEP_MS when the server has no figure yet. */
export async function stageStepMs(stages: number): Promise<number> {
  try {
    const r = await fetch('/api/v1/stylize/healthz', {cache: 'no-store', credentials: 'same-origin'});
    const s = r.ok ? (await r.json())?.last_metrics?.seconds : null;
    return typeof s === 'number' && s > 0 ? Math.min(14_000, Math.max(5_000, s * 1000 * .9 / stages)) : STAGE_STEP_MS;
  } catch { return STAGE_STEP_MS; }
}

/** Gauge of one stage in percent: done stages 100, the running one counts up over its step; the last one, while the
 *  drawing is still being made, goes to 90 in its step and then creeps towards 99 (100 only when the drawing is in). */
export function stagePercent(i: number, shown: number, last: number, elapsed: number, step: number, waiting: boolean): number {
  if (i < shown) return 100;
  if (i > shown) return 0;
  const t = Math.max(0, (elapsed - i * step) / step);
  if (i === last && waiting) return Math.floor(t < 1 ? 90 * t : 90 + 9 * (1 - Math.exp(-(t - 1) / 1.5)));
  return Math.min(99, Math.floor(100 * t));
}

export interface Portrait {url: string; anchors: Anchors | null}

const BUSY_RETRY_MS = 3_000;
const wait = (ms: number, signal?: AbortSignal) => new Promise<void>((done, fail) => {
  const t = setTimeout(done, ms); signal?.addEventListener('abort', () => {clearTimeout(t); fail(new Error('aborted'));}, {once: true});
});
/** POST the frontal photo (multipart field "image" only) and get the drawing back as a PNG Blob. The server draws one
 *  picture at a time and answers 429 while busy (also with a drawing from a request the page has already given up on),
 *  so a busy answer is retried every few seconds until the signal is aborted. */
export async function requestPortrait(front: Blob, signal?: AbortSignal, requestId?: string): Promise<Blob> {
  const body = new FormData();
  body.append('image', front, front.type === 'image/png' ? 'front.png' : 'front.jpg');
  let res: Response;
  for (;;) {
    res = await fetch('/api/v1/stylize/stylize', {method: 'POST', body, signal, cache: 'no-store', credentials: 'same-origin',
      headers: requestId ? {'X-Request-Id': requestId} : undefined});   // the same id on every retry
    if (res.status !== 429) break;
    await wait(BUSY_RETRY_MS, signal);
  }
  if (!res.ok) throw new Error(`portrait ${res.status}`);
  const blob = await res.blob();
  if (blob.type !== 'image/png') throw new Error('portrait: not a PNG');
  return blob;
}

// Face mesh points (anatomical sides: the person's right eye, 33, is on the left of the picture) for each region's arrow.
// 'L' is used when the box sits in the left column of the sheet, so it points at the feature on the left of the picture.
type Ids = number[];
const POINTS: Record<string, {L: Ids; R: Ids}> = {
  눈썹: {L: [105, 107], R: [334, 336]}, 눈: {L: [33, 133], R: [362, 263]}, 미간: {L: [9], R: [9]}, 코: {L: [4, 48], R: [4, 278]},
  광대: {L: [123], R: [352]}, 인중: {L: [2, 0], R: [2, 0]}, 입: {L: [61], R: [291]}, 얼굴형: {L: [152, 175], R: [152, 175]},   // 턱: the lower part of the chin, just above its tip (the line has no dot, so it ends inside the chin)
};

let landmarker: Promise<import('@mediapipe/tasks-vision').FaceLandmarker> | undefined;
async function loadLandmarker() {
  return landmarker ||= import('@mediapipe/tasks-vision').then(async ({FaceLandmarker, FilesetResolver}) =>
    FaceLandmarker.createFromOptions(await FilesetResolver.forVisionTasks('/assets/mediapipe/wasm'), {
      baseOptions: {modelAssetPath: '/assets/mediapipe/face_landmarker.task', delegate: 'CPU'}, runningMode: 'IMAGE', numFaces: 1}));
}

/** Arrow points of the drawing from its own face landmarks (fractions of the picture). null when no face is found;
 *  the sheet then uses its default points. */
export async function portraitAnchors(png: Blob): Promise<Anchors | null> {
  let bitmap: ImageBitmap | undefined;
  try {
    bitmap = await createImageBitmap(png);
    const face = (await loadLandmarker()).detect(bitmap).faceLandmarks[0];
    if (!face || face.length < 400) return null;
    const mean = (ids: Ids): [number, number] => [ids.reduce((s, i) => s + face[i].x, 0) / ids.length, ids.reduce((s, i) => s + face[i].y, 0) / ids.length];
    return Object.fromEntries(Object.entries(POINTS).map(([k, v]) => [k, {L: mean(v.L), R: mean(v.R)}])) as Anchors;
  } catch {
    return null;
  } finally {
    bitmap?.close();
  }
}
