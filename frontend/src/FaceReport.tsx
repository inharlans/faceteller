import {useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties} from 'react';
import {createPortal} from 'react-dom';
import {ResearchReportView, conditionText, cutText, withUnit} from './ResearchReport';
import TABLE from './regionComments.json';
import {NOTICE, type ReportJudgement, type ReportPassage, type ResearchReport} from './types';

// One-page report sheet (A4, printed at the demo). Lengths below are millimetres of the page;
// the stylesheet scales one unit to 1/210 of the sheet width, so screen and paper share one layout.
const ZONE = {w: 186, h: 100}, FRAME = {x: 60, y: 5, w: 66, h: 88}, BOX_W = 56, PLACEHOLDER = {w: 300, h: 400};
// Callout boxes hold one or two sentences of 15 to 100 characters, so their heights differ. The sheet on screen measures
// them; when a column cannot hold its boxes even after the regions are moved between the columns, the type is set smaller.
const BOX_SCALES = [1, .93, .86, .8], BOX_GAP_MIN = 1.2, BOX_GAP_MAX = 9;
export interface Boxes {scale: number; heights: Record<string, number>}
const BOXES: Boxes = {scale: 1, heights: {}};
// Passage cards take the height their text needs and share what is left over, so no text is cut. measureFit picks, in
// this order: the full measurement lines at the largest of SCALES not below 1; else the one-line measurement at the
// largest scale that fits. When nothing fits, the lowest-ranked passage is left out and the rest are laid out again
// (the sheet says how many of the matching passages it carries).
// A single column always has WIDE_ROWS rows of about the same height: with fewer passages the rows below stay empty
// instead of the cards growing tall, and a card with more text takes what it needs from the others.
const SCALES = [1.3, 1.2, 1.1, 1, .93, .86], MAX_PASSAGES = 10, WIDE_ROWS = 5;
export interface Fit {scale: number; count: number; detail: boolean}
const START: Fit = {scale: 1, count: MAX_PASSAGES, detail: false};
const CHOICES = [...SCALES.filter(k => k >= 1).map(scale => ({scale, detail: true})), ...SCALES.map(scale => ({scale, detail: false}))];
/** Up to five passages go in one column of wide cards; more are split evenly over two, the first half on the left. */
export function splitColumns<T>(rows: T[]): T[][] {
  return rows.length <= WIDE_ROWS ? [rows] : [rows.slice(0, Math.ceil(rows.length / 2)), rows.slice(Math.ceil(rows.length / 2))];
}
type Point = [number, number];
type Side = 'L' | 'R';
export type Anchors = Record<string, Record<Side, Point>>;
/** Where each region's arrow lands, as fractions of the portrait (x to the right, y down), for a box on the left and
 *  for one on the right. These fit the placeholder drawing; a real portrait needs its own points (face landmarks of
 *  the picture that is shown). */
export const DEFAULT_ANCHORS: Anchors = {눈썹: {L: [.373, .368], R: [.627, .368]}, 눈: {L: [.383, .42], R: [.617, .42]}, 미간: {L: [.5, .378], R: [.5, .378]}, 코: {L: [.478, .512], R: [.522, .512]},
  광대: {L: [.3, .512], R: [.7, .512]}, 인중: {L: [.5, .557], R: [.5, .557]}, 입: {L: [.447, .598], R: [.553, .598]}, 얼굴형: {L: [.335, .632], R: [.665, .632]}};
// The regions that can carry a callout, with the column each box goes to when both columns have room.
const REGIONS: {key: string; label: string; side: Side}[] = [
  {key: '눈썹', label: '눈썹', side: 'L'}, {key: '눈', label: '눈', side: 'R'}, {key: '미간', label: '미간', side: 'R'}, {key: '코', label: '코', side: 'R'},
  {key: '광대', label: '광대', side: 'L'}, {key: '인중', label: '인중', side: 'L'}, {key: '입', label: '입', side: 'R'}, {key: '얼굴형', label: '턱', side: 'L'}];
const VALENCE: Record<string, string> = {좋음: 'favorable', 나쁨: 'unfavorable', 섞임: 'mixed', 중립: 'neutral'};
const two = (n: number | null | undefined) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '–';

/** The team's sentences (src/regionComments.json, made from the "부위별 문구" sheet): for each region its feature names in a
 *  fixed order and one sentence for each combination of their classes; a feature that is not confirmed is "(보류)". */
export interface RegionTable {features: string[]; comments: Record<string, string[]>}
const REGION_TABLE = (TABLE as unknown as {regions: Record<string, RegionTable>}).regions;
export interface Callout {region: string; label: string; tone: string; text: string}
/** One callout per region. When the server sends its sentences (report.callouts) those are shown as they are; until
 *  then each region gets the sentence for the combination of its confirmed classes from the page's copy of the table.
 *  A combination with no sentence (nothing confirmed, or only classes the books say nothing about) gives no callout. */
export function pickCallouts(report: ResearchReport, table: Record<string, RegionTable> = REGION_TABLE): Callout[] {
  const sent = report.callouts, judged = Object.values(report.judgements);
  if (sent) return REGIONS.flatMap(r => {const c = sent.find(c => c.region === r.key); return c ? [{region: r.key, label: r.label, tone: c.tone || '', text: c.text}] : [];});
  return REGIONS.flatMap(r => {
    const entry = table[r.key]; if (!entry) return [];
    const key = entry.features.map(name => {const j = judged.find(j => j.region === r.key && j.name === name); return j && j.status === '확정' && j.class ? j.class : '(보류)';}).join('|');
    const found = entry.comments[key];
    return found ? [{region: r.key, label: r.label, tone: found[0], text: found[1]}] : [];
  });
}
/** The passages of the sheet: the selected ones first, then the other matching ones by confidence, ten at most. */
export function topPassages(report: ResearchReport): ReportPassage[] {
  const chosen = new Set(report.plan_C_selected.map(p => p.representative));
  return [...report.plan_C_selected, ...report.all_matching.filter(p => !chosen.has(p.representative))].slice(0, MAX_PASSAGES);
}
/** Where the shown passage stands, and how many more passages of the same condition are not shown: the sheet carries
 *  the representative one only (2026-11-21); the others are listed in the research tables on screen. */
export function sourceText(p: ReportPassage): string {
  const book = p.source_detail?.title_ko || p.source || '미상', section = p.source_detail?.section, more = (p.entry_count ?? 1) - 1;
  const where = section ? `${book}(${section})` : book;
  return more > 0 ? `${where} 외 ${more}건` : where;
}
/** The numbers behind a passage in one line: the measured value of each feature it uses, or the CNN probability of its class. */
export function detailText(p: ReportPassage, judgements: Record<string, ReportJudgement>): string {
  return Object.keys(p.features).map(k => {
    const j = judgements[k], cls = p.features[k].class; if (!j) return '';
    if (j.measure && typeof j.measure.value === 'number') return `${j.name} ${withUnit(j.measure.value, j.measure.unit)}`;
    const pr = j.probabilities?.[cls]; return typeof pr === 'number' ? `${j.name} ${cls} ${two(pr)}` : `${j.name} ${cls}`;
  }).filter(Boolean).join(', ');
}
/** The first of CHOICES at which the cards fit, with a millimetre to spare for the printer's own line breaks; null when
 *  none does. In a single column the first choice at which every card stays within its own row is taken, so the rows
 *  come out the same height; only if there is none may a long card take room from the others. */
function measureFit(cards: HTMLElement): {scale: number; detail: boolean} | null {
  const scale = cards.style.getPropertyValue('--fr-k'), detail = cards.dataset.detail, spare = cards.clientWidth / 186;
  const parts = (col: Element) => ({gap: parseFloat(getComputedStyle(col).rowGap) || 0, items: [...col.querySelectorAll<HTMLElement>(':scope > .fr-card')]});
  const fits = () => [...cards.children].every(col => {const {gap, items} = parts(col), rows = Math.max(items.length, col.children.length);
    return items.reduce((n, c) => n + c.offsetHeight, 0) + gap * (rows - 1) <= col.clientHeight - spare;});
  const even = () => [...cards.children].every(col => {const {gap, items} = parts(col), row = (col.clientHeight - gap * (WIDE_ROWS - 1)) / WIDE_ROWS;
    return items.every(c => c.offsetHeight <= row - spare / 4);});
  const first = (test: () => boolean) => CHOICES.find(c => {cards.style.setProperty('--fr-k', String(c.scale)); cards.dataset.detail = c.detail ? 'full' : 'short'; return test();}) ?? null;
  cards.classList.add('fr-measure');
  const found = (cards.dataset.cols === '1' ? first(even) : null) ?? first(fits);
  cards.classList.remove('fr-measure'); cards.style.setProperty('--fr-k', scale); cards.dataset.detail = detail;
  return found;
}
/** What was measured for each feature a passage uses: the value and its class, then what the value is and where the
 *  cuts between the classes lie, or the probabilities the picture classifier gave. */
export function measureRows(p: ReportPassage, judgements: Record<string, ReportJudgement>): {name: string; value: string; cls: string; basis: string}[] {
  return Object.keys(p.features).flatMap(k => {
    const j = judgements[k], m = j?.measure; if (!j) return [];
    const basis = m ? `${m.label}${cutText(m) ? ` (경계: ${cutText(m)})` : ''}` : j.probabilities ? `분류 확률 ${Object.entries(j.probabilities).map(([c, v]) => `${c} ${two(v)}`).join(', ')}` : '';
    return [{name: j.name, value: m && typeof m.value === 'number' ? withUnit(m.value, m.unit) : '', cls: p.features[k].class, basis}];
  });
}
/** About how tall a box is before it has been measured: the label row and the lines its sentence takes. */
export function estimateHeight(text: string, scale = 1): number {
  return (7.6 + Math.ceil(text.length * scale / 19) * 3.83) * scale;
}
interface Placed extends Callout {side: Side; top: number; at: Point}
/** Boxes in two columns beside the portrait. Each region keeps its usual column when both columns can hold their boxes;
 *  otherwise the split with the least overflow is taken (then the most even one). A column is ordered by where its arrows
 *  land, so the lines do not cross. overflow is how far the taller column sticks out of the zone. */
export function placeCallouts(callouts: Callout[], anchors: Anchors, picture: {w: number; h: number}, boxes: Boxes = BOXES): {placed: Placed[]; overflow: number} {
  // object-fit: cover of the picture inside the frame
  const s = Math.max(FRAME.w / picture.w, FRAME.h / picture.h), ox = FRAME.x + (FRAME.w - picture.w * s) / 2, oy = FRAME.y + (FRAME.h - picture.h * s) / 2;
  const at = (c: Callout, side: Side): Point => {const a = (anchors[c.region] || DEFAULT_ANCHORS[c.region])[side]; return [ox + a[0] * picture.w * s, oy + a[1] * picture.h * s];};
  const height = (c: Callout) => boxes.heights[c.region] ?? estimateHeight(c.text, boxes.scale), usual = (c: Callout) => REGIONS.find(r => r.key === c.region)!.side;
  const used = (col: Callout[]) => col.reduce((n, c) => n + height(c), 0) + BOX_GAP_MIN * Math.max(col.length - 1, 0);
  let best: {cols: Record<Side, Callout[]>; cost: number[]} | null = null;
  for (let mask = 0; mask < 1 << callouts.length; mask++) {
    const cols: Record<Side, Callout[]> = {L: [], R: []};
    callouts.forEach((c, i) => cols[mask >> i & 1 ? 'R' : 'L'].push(c));
    const cost = [Math.max(0, Math.max(used(cols.L), used(cols.R)) - ZONE.h), callouts.filter((c, i) => (mask >> i & 1 ? 'R' : 'L') !== usual(c)).length, Math.abs(cols.L.length - cols.R.length)];
    const at = best ? cost.findIndex((v, i) => v !== best!.cost[i]) : 0;  // first cost that differs
    if (!best || at >= 0 && cost[at] < best.cost[at]) best = {cols, cost};
  }
  const placed = (['L', 'R'] as const).flatMap(k => {
    const col = (best?.cols[k] || []).map(c => ({c, at: at(c, k)})).sort((a, b) => a.at[1] - b.at[1]);
    const total = col.reduce((n, {c}) => n + height(c), 0), gap = Math.max(0, Math.min(BOX_GAP_MAX, (ZONE.h - total) / Math.max(col.length - 1, 1)));
    let top = Math.max(0, (ZONE.h - total - gap * (col.length - 1)) / 2);
    return col.map(({c, at}) => {const p = {...c, side: k, at, top}; top += height(c) + gap; return p;});
  });
  return {placed, overflow: best?.cost[0] ?? 0};
}
/** Stand-in for the portrait until the corrected picture arrives: a plain line drawing, not a person. */
function PlaceholderPortrait({note}: {note?: string}) {
  const ink = '#3a312a';
  return <svg viewBox={`0 0 ${PLACEHOLDER.w} ${PLACEHOLDER.h}`} preserveAspectRatio="xMidYMid slice" role="img" aria-label="사진이 들어갈 자리(예시 그림)">
    <rect width="300" height="400" fill="#f1e8d6"/>
    <path d="M22 400C30 338 92 322 122 308l28 26 28-26c30 14 92 30 100 92Z" fill="#e4d8bf" stroke={ink} strokeWidth="2"/>
    <path d="M122 308l28 26 28-26" fill="none" stroke={ink} strokeWidth="2"/>
    <path d="M126 262v44l24 24 24-24v-44Z" fill="#f3e3cd" stroke={ink} strokeWidth="2"/>
    <ellipse cx="73" cy="188" rx="9" ry="17" fill="#f3e3cd" stroke={ink} strokeWidth="2"/><ellipse cx="227" cy="188" rx="9" ry="17" fill="#f3e3cd" stroke={ink} strokeWidth="2"/>
    <path d="M76 170c0-62 30-98 74-98s74 36 74 98c0 60-34 102-74 102s-74-42-74-102Z" fill="#f6e8d4" stroke={ink} strokeWidth="2"/>
    <path d="M73 178C60 96 104 54 150 54s92 40 77 124c-4-34-16-58-38-70-26 12-58 14-80 6-18 12-30 34-36 64Z" fill="#2f2823"/>
    <path d="M94 150q20-9 42-2M164 148q22-7 42 2" fill="none" stroke={ink} strokeWidth="5.5" strokeLinecap="round"/>
    <path d="M98 170q17-11 34 0q-17 8-34 0ZM168 170q17-11 34 0q-17 8-34 0Z" fill="#fffaf2" stroke={ink} strokeWidth="2"/>
    <circle cx="115" cy="169.500" r="5" fill={ink}/><circle cx="185" cy="169.500" r="5" fill={ink}/>
    <path d="M147 160c-1 16-4 32-8 42q11 9 22 0" fill="none" stroke={ink} strokeWidth="2" strokeLinecap="round"/>
    <path d="M146 216v10M154 216v10" fill="none" stroke={ink} strokeWidth="1.2" strokeLinecap="round" opacity=".55"/>
    <path d="M128 238q11-5 22-1q11-4 22 1q-22 13-44 0Z" fill="#d9a394" stroke={ink} strokeWidth="1.6"/><path d="M128 238q22 4 44 0" fill="none" stroke={ink} strokeWidth="1.4"/>
    <text x="150" y="386" textAnchor="middle" fontSize="13" fill={ink} opacity=".7">{note || '사진이 들어갈 자리 (예시 그림)'}</text>
  </svg>;
}

function Sheet({report, notice, portraitUrl, portraitNote, anchors, titleId, fit, boxes, round, onFit, onBoxes}: {report: ResearchReport; notice?: string; portraitUrl?: string | null; portraitNote?: string; anchors?: Anchors; titleId?: string; fit: Fit; boxes: Boxes; round?: number; onFit?: (fit: Fit) => void; onBoxes?: (boxes: Boxes) => void}) {
  const cards = useRef<HTMLDivElement>(null), zone = useRef<HTMLElement>(null);
  // Only the sheet on screen measures; the printer's copy is laid out the same way and takes the result.
  useLayoutEffect(() => {
    if (!onFit || !cards.current) return;
    const found = measureFit(cards.current), least = {scale: SCALES[SCALES.length - 1], detail: false};
    onFit(found === null && fit.count > 1 ? {...least, count: fit.count - 1} : {...(found ?? least), count: fit.count});
  }, [report, fit.count, round, onFit]);
  // Size of the shown picture, kept with the URL it belongs to. (It used to be cleared in an effect when the URL changed;
  // a picture from memory could load before that effect ran, the size was then wiped and no load event came again, so
  // the callouts never appeared on screen.) An image that is already complete when mounted is read at once.
  const [picture, setPicture] = useState<{url: string; w: number; h: number} | null>(null), img = useRef<HTMLImageElement>(null);
  useLayoutEffect(() => {const el = img.current; if (portraitUrl && el && el.complete && el.naturalWidth) setPicture({url: portraitUrl, w: el.naturalWidth, h: el.naturalHeight});}, [portraitUrl]);
  const shown = portraitUrl && picture?.url === portraitUrl ? picture : null;
  const callouts = pickCallouts(report), matching = topPassages(report), passages = matching.slice(0, fit.count), columns = splitColumns(passages), ready = !portraitUrl || shown;
  const frame = shown ? {w: shown.w, h: shown.h} : PLACEHOLDER, placed = ready ? placeCallouts(callouts, anchors || DEFAULT_ANCHORS, frame, boxes).placed : [];
  // Measure the boxes at the current type size; if they cannot be placed inside the zone, try the next smaller size.
  useLayoutEffect(() => {
    if (!onBoxes || !zone.current || !ready) return;
    const unit = zone.current.clientWidth / ZONE.w, heights = Object.fromEntries([...zone.current.querySelectorAll<HTMLElement>('.fr-box')].map(el => [el.dataset.region!, Math.ceil(el.offsetHeight / unit * 10) / 10]));
    const next = BOX_SCALES[BOX_SCALES.indexOf(boxes.scale) + 1];
    onBoxes(next && placeCallouts(callouts, anchors || DEFAULT_ANCHORS, frame, {scale: boxes.scale, heights}).overflow > 0 ? {scale: next, heights: {}} : {scale: boxes.scale, heights});
  }, [report, boxes.scale, round, ready, onBoxes]);
  const mm = (n: number) => `calc(var(--mm) * ${n})`, edge = (p: Placed) => p.side === 'L' ? BOX_W : ZONE.w - BOX_W;
  return <div className="fr-sheet"><div className="fr-page">
    <header className="fr-head">
      <p className="fr-eyebrow">얼굴 형태와 옛 관상서의 기록</p>
      <h1 id={titleId}>페이스텔러 분석 리포트</h1>
      <p className="fr-motto"><i/><span>形</span><span>文</span><i/></p>
      <span className="fr-seal" aria-hidden="true">觀相</span>
    </header>
    <section className="fr-face" aria-label="부위별 풀이" ref={zone} style={{'--fr-bk': boxes.scale} as CSSProperties}>
      <div className="fr-portrait" style={{left: mm(FRAME.x), top: mm(FRAME.y), width: mm(FRAME.w), height: mm(FRAME.h)}}>
        {portraitUrl ? <img ref={img} src={portraitUrl} alt="AI가 그린 캐릭터 그림" onLoad={e => setPicture({url: portraitUrl, w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight})}/> : <PlaceholderPortrait note={portraitNote}/>}
      </div>
      <svg className="fr-lines" viewBox={`0 0 ${ZONE.w} ${ZONE.h}`} preserveAspectRatio="none" aria-hidden="true">
        {placed.map(p => {const x = edge(p), y = p.top + 3.5 * boxes.scale, bend = x + (p.side === 'L' ? 3.5 : -3.5);
          return <g key={p.region}><path d={`M${x} ${y}H${bend}L${p.at[0]} ${p.at[1]}`}/><circle cx={p.at[0]} cy={p.at[1]} r="1.05"/></g>;})}
      </svg>
      {placed.map(p => <article key={p.region} className="fr-box" data-region={p.region} data-tone={p.tone} style={{top: mm(p.top), width: mm(BOX_W), [p.side === 'L' ? 'left' : 'right']: 0}}>
        <h2><span>{p.label}</span></h2><p>{p.text}</p>
      </article>)}
      {callouts.length === 0 && <p className="fr-none">풀이를 붙일 수 있는 부위가 없습니다.</p>}
    </section>
    {report.summary?.trim() && <section className="fr-summary" aria-label="한줄평"><span>한줄평</span><p>{report.summary}</p></section>}
    <section className="fr-lit" aria-label="문헌 기록">
      <h2><span>옛 책에는 이렇게 적혀 있습니다</span><small>원문, 직역, 출처, 신뢰도, 측정값</small>{passages.length < report.all_matching.length && <small className="fr-count">조건이 맞는 기록 {report.all_matching.length}개 중 신뢰도 높은 {passages.length}개</small>}</h2>
      {passages.length ? <div className="fr-cards" data-cols={columns.length} data-detail={fit.detail ? 'full' : 'short'} style={{'--fr-k': fit.scale} as CSSProperties} ref={cards}>{columns.map((column, c) => <div key={c} className="fr-col">
        {column.map((p, i) => {const detail = detailText(p, report.judgements), where = sourceText(p);
          return <article key={p.representative} className="fr-card" data-valence={VALENCE[p.valence] || 'neutral'}>
            <div className="fr-main">
              <div className="fr-card-head"><span className="fr-rank">{c * columns[0].length + i + 1}</span><b>{conditionText(p.requirement)}</b><em>{p.valence}</em><span className="fr-conf">신뢰도 {two(p.G)}</span></div>
              {p.original_text && <p className="fr-orig" lang="zh">{p.original_text}</p>}
              <p className="fr-trans">{p.translation || '번역 없음'}</p>
              <p className="fr-meta"><b>출처</b>{where}{detail && <span className="fr-short"><b>측정</b>{detail}</span>}</p>
            </div>
            <ul className="fr-measures">{measureRows(p, report.judgements).map(r => <li key={r.name}><b>{r.name}</b>{r.value && <strong>{r.value}</strong>}<i>→ {r.cls}</i>{r.basis && <span>{r.basis}</span>}</li>)}</ul>
          </article>;})}
        {columns.length === 1 && Array.from({length: WIDE_ROWS - column.length}, (_, i) => <div key={i} className="fr-slot" aria-hidden="true"/>)}
      </div>)}</div> : <p className="fr-none">조건이 맞는 문헌 기록이 없습니다.</p>}
    </section>
    <footer className="fr-foot"><p>{notice || NOTICE} 신뢰도는 판정이 경계에서 얼마나 떨어져 있는지를 나타내며 정답 확률이 아닙니다.</p><span>페이스텔러</span></footer>
  </div></div>;
}

/** The report sheet on screen, a copy of it for the printer, and the research tables kept below as a fallback. */
export function FaceReport({report, notice, portraitUrl, portraitNote, anchors, onReset}: {report: ResearchReport; notice?: string; portraitUrl?: string | null; portraitNote?: string; anchors?: Anchors; onReset: () => void}) {
  const [mounted, setMounted] = useState(false), [fit, setFit] = useState<Fit>(START), [boxes, setBoxes] = useState<Boxes>(BOXES), [round, setRound] = useState(0);
  const onFit = useCallback((next: Fit) => setFit(prev => prev.scale === next.scale && prev.count === next.count && prev.detail === next.detail ? prev : next), []);
  const onBoxes = useCallback((next: Boxes) => setBoxes(prev => prev.scale === next.scale && JSON.stringify(prev.heights) === JSON.stringify(next.heights) ? prev : next), []);
  // Measure again from the start once the fonts are in: the first pass may have run with fallback type.
  useEffect(() => {let live = true; void document.fonts?.ready.then(() => {if (live) {setFit(START); setBoxes(BOXES); setRound(r => r + 1);}}); return () => {live = false;};}, [report]);
  useEffect(() => {setMounted(true); document.body.classList.add('fr-print-mode'); return () => document.body.classList.remove('fr-print-mode');}, []);
  return <section className="face-report" aria-labelledby="fr-title">
    <Sheet report={report} notice={notice} portraitUrl={portraitUrl} portraitNote={portraitNote} anchors={anchors} titleId="fr-title" fit={fit} boxes={boxes} round={round} onFit={onFit} onBoxes={onBoxes}/>
    <div className="actions center"><button className="button" onClick={() => window.print()}>A4 한 장 인쇄</button><button className="button ghost" onClick={onReset}>처음으로</button></div>
    <details className="fr-backup"><summary>연구용 상세 결과 (판정 과정 확인용)</summary><ResearchReportView report={report} notice={notice} onReset={onReset}/></details>
    {mounted && createPortal(<div className="fr-print" aria-hidden="true"><Sheet report={report} notice={notice} portraitUrl={portraitUrl} portraitNote={portraitNote} anchors={anchors} fit={fit} boxes={boxes}/></div>, document.body)}
  </section>;
}
