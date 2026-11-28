import {withUnit} from './ResearchReport';
import type {ReportJudgement, ResearchReport} from './types';

// What the progress screen writes out while the visitor waits (2026-11-28): the steps the server really runs, in its
// order (deploy/v1/run_person.py, structure v10), in plain sentences, with the measured value of every feature and a
// few other figures of this very run once the answer is in. The lines share the whole wait: each gets a part of it in
// proportion to its length, and the gauges of the stages follow the same clock, so no single step fills the screen.
// Every number in the sentences mirrors deploy/v1 (478 landmarks, 5 models per part, 8 CNN features, 50 condition
// groups, 596 combinations, floor 0.6): change it there, change it here.

/** features: one line per feature with its measured value. detail: with the cuts and class probabilities, or the class
 *  and confidence only. figures: the other figures of this run. drawing: the block about the portrait. pace: how the
 *  wait is shared between the lines (by length, evenly per line, or evenly per stage). */
export interface LogOptions {features: boolean; detail: 'full' | 'short'; figures: boolean; drawing: boolean; pace: 'chars' | 'lines' | 'stages'}
export const LOG_DEFAULTS: LogOptions = {features: true, detail: 'full', figures: true, drawing: true, pace: 'chars'};

type Group = 'cnn' | 'numeric' | 'measured' | 'depth';
export interface FeatureLine {group: Group; full: string; short: string}
/** Figures of this run: counts from the typed tables, the rest from fields the report carries beside them (photos, summary_info). */
export interface RunFacts {yaw: number[] | null; faceWidth: number | null; features: number; confirmed: number; held: number; unjudged: number;
  matching: number; selected: number; regionSentences: number | null; summarySource: string | null; summaryModel: string | null; lines: FeatureLine[]}

const rec = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null;
const two = (n: number) => n.toFixed(2), minus = (s: string) => s.replace(/^-/, '−'), solid = (s: string) => s.replace(/-/g, '‑');   // ‑: a hyphen the line does not break at

/** One feature in a line: what was measured (or the class probabilities), the cuts it was compared with, and the class. */
export function featureLine(j: ReportJudgement): FeatureLine {
  const method = j.method ?? '', group: Group = method === 'CNN' ? 'cnn' : method.startsWith('VGGT') ? 'depth' : method === '수치' ? 'numeric' : 'measured', name = solid(j.name);
  if (j.class === null || j.confidence === null || (j.status !== '확정' && j.status !== '보류')) {const text = `${name}: ${j.status}`; return {group, full: text, short: text};}
  const held = j.status === '보류', short = held ? `${name}: 보류 (${j.class}, 신뢰도 ${two(j.confidence)})` : `${name}: ${j.class} (신뢰도 ${two(j.confidence)})`;
  const p = j.probabilities, m = j.measure;
  if (group === 'cnn' && p && Object.keys(p).length) return {group, short, full: `${name}: ${Object.entries(p).map(([c, v]) => `${c} ${two(v)}`).join(', ')} → ${held ? `보류 (신뢰도 ${two(j.confidence)})` : j.class}`};
  if (m && typeof m.value === 'number' && m.cuts?.length) return {group, short,
    full: `${name}: ${minus(withUnit(m.value, m.unit))} (경계 ${m.cuts.map(c => minus(withUnit(c, m.unit))).join(', ')}) → ${held ? `보류 (${j.class}, 신뢰도 ${two(j.confidence)})` : `${j.class}, 신뢰도 ${two(j.confidence)}`}`};
  return {group, full: short, short};
}

export function runFacts(report: ResearchReport | null | undefined): RunFacts | null {
  if (!report) return null;
  const extra = report as unknown as Record<string, unknown>, keys = Object.keys(report.judgements).sort(), js = keys.map(k => report.judgements[k]);
  const photos = Array.isArray(extra.photos) ? extra.photos.map(p => rec(p) ? num(p.yaw) : null) : [], info = rec(extra.summary_info) ? extra.summary_info : {};
  const confirmed = js.filter(j => j.status === '확정').length, held = js.filter(j => j.status === '보류').length;
  return {yaw: photos.length === 3 && photos.every(y => y !== null) ? photos as number[] : null, faceWidth: num(extra.face_width_px),
    features: js.length, confirmed, held, unjudged: js.length - confirmed - held, matching: report.all_matching.length, selected: report.plan_C_selected.length,
    regionSentences: Array.isArray(report.callouts) ? report.callouts.length : null,
    summarySource: typeof info.source === 'string' ? info.source : null, summaryModel: typeof info.model === 'string' ? info.model : null, lines: js.map(featureLine)};
}

type Fact = (f: RunFacts) => string | null;
type Entry = string | Fact | {features: Group};
const deg = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}°`;

/** Before the request: the checks the page made itself. Always shown as done. */
export const BEFORE = {title: '사진 확인', lines: [
  '보내기 전에 브라우저에서 사진 세 장의 얼굴, 방향, 크기, 밝기, 표정을 확인했습니다.',
  '사진은 서버 메모리에서만 처리하고 저장하지 않습니다.',
]};

/** One entry per server stage (types.ts STAGES, same order). A function is a figure of this run; {features} stands for
 *  the lines of the features judged that way. */
export const STEPS: Entry[][] = [
  [ // 얼굴 정렬
    '사진 세 장에서 각각 얼굴 점 478개를 찾습니다. (MediaPipe Face Landmarker)',
    '얼굴이 없거나 둘 이상 보이는 사진은 여기서 돌려보냅니다.',
    '얼굴 점으로 고개의 좌우, 상하, 기울기 각도와 눈 감음, 웃음 정도를 계산합니다.',
    f => f.yaw ? `이번 사진의 좌우 각도: 정면 ${deg(f.yaw[0])}, 사선 ${deg(f.yaw[1])}와 ${deg(f.yaw[2])}` : null,
  ],
  [ // 부위 추출
    '정면 사진의 얼굴 점을 표준 얼굴 틀에 3차원으로 맞춰, 고개가 조금 돌아가 있어도 같은 자세가 되도록 사진을 폅니다.',
    f => f.faceWidth ? `이번 정면 사진의 얼굴 폭: ${Math.round(f.faceWidth)}픽셀` : null,
    '편 사진에서 눈(좌, 우), 눈썹(좌, 우), 코, 입 조각을 잘라 냅니다.',
    '조각마다 얼굴 전체 그림을 옆에 붙여, 모델이 부위와 얼굴의 비율을 함께 보게 합니다.',
  ],
  [ // 형태 분류
    '부위마다 신경망(ResNet18) 다섯 개가 따로 판정하고, 그 확률을 평균합니다. 이렇게 판정하는 특징은 8개입니다.',
    '확률은 학습에 쓰지 않은 사람들의 자료로 보정합니다.',
    {features: 'cnn'},
    '미간과 입 폭은 얼굴 분할 모델로 눈썹과 입술 영역을 구한 뒤 거리로 잽니다.',
    {features: 'numeric'},
    '눈 크기, 코 길이, 턱 폭, 눈썹‑눈 간격, 입술 두께, 인중 길이는 얼굴 점 사이의 거리와 각도로 재고, K‑FACE 자료에서 정한 경계값과 비교합니다.',
    {features: 'measured'},
  ],
  [ // 깊이 분석
    '사진 세 장을 3차원 복원 모델(VGGT)에 함께 넣어, 얼굴 점마다 3차원 위치를 얻습니다.',
    '코 높이, 광대 돌출, 입 돌출은 이 3차원 좌표로 계산합니다. 정면 사진 한 장으로는 잴 수 없는 값입니다.',
    {features: 'depth'},
  ],
  [ // 문헌 선정
    '신뢰도가 기준(0.6)에 못 미치는 판정은 보류하고, 확정된 형태만 문헌과 대조합니다.',
    f => f.features ? `이번 결과: 특징 ${f.features}개 중 ${f.confirmed}개 확정, ${f.held}개 보류${f.unjudged ? `, ${f.unjudged}개 판정 불가` : ''}` : null,
    '확정된 형태를 문헌 조건 50묶음과 대조합니다. 조건의 형태가 모두 맞아야 그 조항이 해당됩니다.',
    '해당하는 조항 가운데 부위를 돌아가며 5개까지 고르고, 흉하게 본 기록은 2개까지만 넣습니다.',
    f => f.features ? `이번 결과: 조건이 맞는 조항 ${f.matching}개 중 ${f.selected}개 선정` : null,
    '부위별 설명은 형태 조합 596가지에 미리 연결해 둔 문구에서 찾습니다.',
    f => f.regionSentences !== null ? `이번 결과: 설명이 붙는 부위 ${f.regionSentences}곳` : null,
  ],
  [ // 총평 작성
    '총평은 서버 안에서 도는 언어 모델이 앞의 두 문장을 쓰고, 조심할 점은 정해 둔 표현으로 붙입니다. 외부 서비스로는 보내지 않습니다.',
    '모델을 쓸 수 없거나 답이 기준에 맞지 않으면 정해 둔 문형으로 대신합니다.',
    f => f.summarySource === 'model' ? `이번 총평: 언어 모델${f.summaryModel ? `(${f.summaryModel})` : ''}이 작성` : f.summarySource === 'template' ? '이번 총평: 정해 둔 문형으로 작성' : null,
  ],
];
/** The portrait for the report sheet (portrait.ts): the last block, on the clock of the last stage. */
export const DRAWING = {title: '인물화 그리기', lines: [
  '정면 사진을 이미지 생성 모델(FLUX.1 Kontext)에 보내 결과지에 넣을 인물화를 그립니다.',
  '그림은 보여 주기 위한 것이고, 모든 측정은 원본 사진으로 합니다.',
]};

/** block: index into Plan.blocks. pending: a figure whose answer is not in yet (never shown; it only keeps its time). */
export interface Slot {stage: number; block: number; text: string; live: boolean; pending: boolean}
/** starts[j]: the part of the wait (0..1) at which slot j opens. bounds[i]: where stage i starts; bounds[stages] = 1.
 *  factsFrom: where the first figure of this run is due; without the answer the account waits just before it. */
export interface Plan {blocks: string[]; slots: Slot[]; starts: number[]; bounds: number[]; factsFrom: number}
// Until the answer is in, the lines of the features keep the time of the 19 features of the structure.
const EXPECTED: Record<Group, number> = {cnn: 8, numeric: 2, measured: 6, depth: 3}, NOMINAL = 34;

export function logPlan(titles: readonly string[], facts: RunFacts | null, o: LogOptions = LOG_DEFAULTS): Plan {
  const slots: Slot[] = [], n = titles.length, wait = (stage: number) => slots.push({stage, block: stage, text: '', live: true, pending: true});
  titles.forEach((_, stage) => {
    for (const e of STEPS[stage] ?? []) {
      if (typeof e === 'string') slots.push({stage, block: stage, text: e, live: false, pending: false});
      else if (typeof e === 'function') {
        if (!o.figures) continue;
        if (!facts) {wait(stage); continue;}
        const text = e(facts); if (text) slots.push({stage, block: stage, text, live: true, pending: false});
      } else if (o.features) {
        if (!facts) {for (let i = 0; i < EXPECTED[e.features]; i++) wait(stage); continue;}
        for (const l of facts.lines) if (l.group === e.features) slots.push({stage, block: stage, text: o.detail === 'full' ? l.full : l.short, live: true, pending: false});
      }
    }
  });
  if (o.drawing && n) for (const text of DRAWING.lines) slots.push({stage: n - 1, block: n, text, live: false, pending: false});
  const inStage = titles.map((_, i) => slots.filter(s => s.stage === i).length);
  const weight = slots.map(s => o.pace === 'lines' ? 1 : o.pace === 'stages' ? 1 / inStage[s.stage] : Math.max(16, Math.min(80, s.pending ? NOMINAL : s.text.length)));
  const total = weight.reduce((a, b) => a + b, 0) || 1, starts: number[] = []; let sum = 0;
  for (const w of weight) {starts.push(sum / total); sum += w;}
  const bounds = titles.map((_, i) => {const j = slots.findIndex(s => s.stage >= i); return j < 0 ? 1 : starts[j];}); bounds.push(1);
  const first = slots.findIndex(s => s.pending);
  return {blocks: [...titles, ...(o.drawing && n ? [DRAWING.title] : [])], slots, starts, bounds, factsFrom: first < 0 ? 1 : starts[first]};
}

/** The stage the clock is in (-1 before the first, never past the last) and the gauge of every stage in percent; the
 *  running one stops at 99 (100 only when everything is in). */
export function stageView(bounds: readonly number[], f: number): {paced: number; percents: number[]} {
  const n = bounds.length - 1; let paced = -1;
  for (let i = 0; i < n; i++) if (bounds[i] <= f) paced = i;
  return {paced, percents: Array.from({length: n}, (_, i) => i < paced ? 100 : i > paced ? 0 : Math.max(0, Math.min(99, Math.floor((f - bounds[i]) / Math.max(1e-9, bounds[i + 1] - bounds[i]) * 100))))};
}
/** Part of the wait that has passed when only a time estimate is known: straight to 0.9, then creeping towards 1. */
export function estimateFraction(elapsed: number, total: number): number {
  const x = Math.max(0, elapsed) / Math.max(1, total);
  return x < .9 ? x : .9 + .099 * (1 - Math.exp(-(x - .9) / .3));
}

export interface LogLine {text: string; live: boolean}
export interface LogBlock {title: string; state: 'done' | 'active' | 'waiting'; lines: LogLine[]}
/** The blocks as they stand at f, and how far the newest line is through its own time (0..1; 1 when nothing is open). */
export function logView(plan: Plan, f: number): {blocks: LogBlock[]; newest: number} {
  const blocks: LogBlock[] = plan.blocks.map(title => ({title, state: 'waiting', lines: []})); let last = -1;
  plan.slots.forEach((s, j) => {if (plan.starts[j] <= f && !s.pending) {blocks[s.block].lines.push({text: s.text, live: s.live}); last = j;}});
  blocks.forEach((b, i) => {
    if (!b.lines.length) return;
    const end = plan.slots.reduce((e, s, j) => s.block === i ? Math.max(e, plan.starts[j + 1] ?? 1) : e, 0);
    b.state = f >= end ? 'done' : 'active';
  });
  return {blocks, newest: last < 0 || f >= 1 ? 1 : Math.max(0, Math.min(1, (f - plan.starts[last]) / Math.max(1e-9, (plan.starts[last + 1] ?? 1) - plan.starts[last])))};
}
