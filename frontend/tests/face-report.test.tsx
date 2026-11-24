import {describe, expect, it} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {Results} from '../src/Results';
import {DEFAULT_ANCHORS, detailText, estimateHeight, measureRows, sourceText, pickCallouts, placeCallouts, splitColumns, topPassages, type Callout, type RegionTable} from '../src/FaceReport';
import TABLE from '../src/regionComments.json';
import {validateResult} from '../src/api';
import type {ReportJudgement, ReportPassage, ResearchReport} from '../src/types';

// Authored synthetic values only; no real person's result.
const judgement = (name: string, region: string, cls: string, confidence: number, status = '확정', extra: Partial<ReportJudgement> = {}): ReportJudgement =>
  ({name, region, method: 'CNN', value: null, class: cls, confidence, threshold: .6, status, ...extra});
const passage = (id: string, valence: string, G: number, feature = 'F05', cls = '일자'): ReportPassage =>
  ({representative: id, requirement: [`${feature} 합성 특징:${cls}`], valence, G, regions: ['합성 부위'], features: {[feature]: {class: cls, confidence: G, held: false}}, translation: `합성 번역 ${id}`, source: '합성서', original_text: `合成${id}`});
const table: Record<string, RegionTable> = {눈썹: {features: ['눈썹 꼬리 길이', '눈썹 휨'], comments: {'긴 편|일자': ['좋음', '합성 눈썹 문구'], '(보류)|일자': ['좋음', '합성 일자 문구']}},
  코: {features: ['코 높이'], comments: {높음: ['조심', '합성 코 문구']}}, 입: {features: ['입 폭'], comments: {'넓은 편': ['좋음', '합성 입 문구']}}, 인중: {features: ['인중 길이'], comments: {짧음: ['조심', '합성 인중 문구']}}};
const report = (): ResearchReport => ({schema: 'faceteller-research-report-v1', structure: '합성 구조', summary: '합성 한줄평',
  judgements: {F04: judgement('눈썹 꼬리 길이', '눈썹', '긴 편', .7), F05: judgement('눈썹 휨', '눈썹', '일자', .9, '확정', {probabilities: {일자: .9, 곡선: .1}}),
    F10: judgement('코 높이', '코', '높음', .8, '확정', {method: 'VGGT 수치', measure: {label: '합성 높이', value: .205, unit: '비율'}}),
    F12: judgement('입 폭', '입', '넓은 편', .5, '보류'), F15: judgement('인중 길이', '인중', '보통', .95)},
  plan_C_selected: [passage('S-2', '좋음', .7)], all_matching: [passage('S-1', '좋음', .9), passage('S-2', '좋음', .7), ...Array.from({length: 12}, (_, i) => passage(`M-${i}`, '나쁨', .6 - i / 100))],
  blocked_by_hold: [], held_judgements: [], cnn_check: {}});

describe('report sheet', () => {
  it('gives each region the sentence for the combination of its confirmed classes', () => {
    expect(pickCallouts(report(), table)).toEqual([{region: '눈썹', label: '눈썹', tone: '좋음', text: '합성 눈썹 문구'}, {region: '코', label: '코', tone: '조심', text: '합성 코 문구'}]);
  });
  it('reads a feature that is not confirmed as held, and gives nothing for a combination without a sentence', () => {
    const r = report(); r.judgements.F04.status = '보류';
    expect(pickCallouts(r, table).find(c => c.region === '눈썹')?.text).toBe('합성 일자 문구');
    const regions = pickCallouts(report(), table).map(c => c.region);
    expect(regions).not.toContain('입');    // its only feature is held: no sentence for "(보류)"
    expect(regions).not.toContain('인중');  // confirmed as 보통, and the books say nothing about that class
  });
  it('shows the server\'s sentences as they are when it sends them, in the page\'s region order', () => {
    const r = report(); r.callouts = [{region: '코', tone: '조심', text: '서버 코 문구'}, {region: '눈', text: '서버 눈 문구'}, {region: '없는 부위', text: '무시'}];
    expect(pickCallouts(r, table)).toEqual([{region: '눈', label: '눈', tone: '', text: '서버 눈 문구'}, {region: '코', label: '코', tone: '조심', text: '서버 코 문구'}]);
    r.callouts = []; expect(pickCallouts(r, table)).toEqual([]);  // the server found none: nothing is looked up here
    const kept = validateResult({status: 'COMPLETE', items: [], passages: [], research_report: {...report(), callouts: [{region: '코', text: '서버 코 문구'}]}}).research_report;
    expect(kept?.callouts).toEqual([{region: '코', text: '서버 코 문구'}]);
    const dropped = validateResult({status: 'COMPLETE', items: [], passages: [], research_report: {...report(), callouts: [{region: '코'}]}}).research_report;
    expect(dropped).not.toBeNull(); expect(dropped?.callouts).toBeUndefined();  // malformed: the page falls back to its own table
  });
  it('finds the team\'s sentences for real feature names and classes', () => {
    const regions = (TABLE as unknown as {regions: Record<string, RegionTable>}).regions;
    expect(Object.keys(regions).sort()).toEqual(['광대', '눈', '눈썹', '미간', '얼굴형', '인중', '입', '코']);
    const r = report(); r.judgements = {F07: judgement('미간', '미간', '넓음', .8), F17: judgement('턱 폭', '얼굴형', '보통', .8)};
    const found = pickCallouts(r);
    expect(found.map(c => c.label)).toEqual(['미간']); expect(found[0].text).toBe(regions.미간.comments.넓음[1]);
    for (const e of Object.values(regions)) for (const [key, [tone, text]] of Object.entries(e.comments)) {
      expect(key.split('|')).toHaveLength(e.features.length); expect(['좋음', '조심', '섞임']).toContain(tone); expect(text.length).toBeGreaterThan(0); expect(text.length).toBeLessThanOrEqual(100);
    }
  });
  it('lists the selected passages first, then the other matching ones, ten at most', () => {
    const ids = topPassages(report()).map(p => p.representative);
    expect(ids).toHaveLength(10); expect(ids.slice(0, 3)).toEqual(['S-2', 'S-1', 'M-0']); expect(new Set(ids).size).toBe(10);
  });
  it('writes the numbers behind a passage: the measured value, or the probability of the judged class', () => {
    const r = report();
    expect(detailText(passage('X', '좋음', .9), r.judgements)).toBe('눈썹 휨 일자 0.90');
    expect(detailText(passage('Y', '좋음', .8, 'F10', '높음'), r.judgements)).toBe('코 높이 0.205');
  });
  it('names the source of the representative passage and counts the other passages of the condition', () => {
    const p = {...passage('X', '좋음', .9), source_detail: {title_ko: '합성서', section: '合成節'}};
    expect(sourceText(p)).toBe('합성서(合成節)');
    expect(sourceText({...p, entry_count: 1})).toBe('합성서(合成節)');
    expect(sourceText({...p, entry_count: 13})).toBe('합성서(合成節) 외 12건');
    expect(sourceText({...passage('Y', '좋음', .9), source: null, entry_count: 2})).toBe('미상 외 1건');
  });
  it('spells the measurement out: value, class, what was measured and the cuts, or the classifier probabilities', () => {
    const r = report(); r.judgements.F10.measure = {label: '합성 높이', value: .205, unit: '비율', cuts: [.16, .19], cut_classes: ['낮음', '보통', '높음']};
    expect(measureRows(passage('Y', '좋음', .8, 'F10', '높음'), r.judgements)).toEqual([{name: '코 높이', value: '0.205', cls: '높음', basis: '합성 높이 (경계: 0.160 미만 낮음, 0.190 이상 높음, 그 사이 보통)'}]);
    expect(measureRows(passage('X', '좋음', .9), r.judgements)).toEqual([{name: '눈썹 휨', value: '', cls: '일자', basis: '분류 확률 일자 0.90, 곡선 0.10'}]);
  });
  const box = (region: string, length: number): Callout => ({region, label: region, tone: '좋음', text: '가'.repeat(length)});
  const columns = (placed: ReturnType<typeof placeCallouts>['placed']) => (['L', 'R'] as const).map(side => placed.filter(p => p.side === side));
  it('keeps each region in its usual column when both columns have room, ordered by where the arrows land', () => {
    const {placed, overflow} = placeCallouts(Object.keys(DEFAULT_ANCHORS).map(r => box(r, 30)), DEFAULT_ANCHORS, {w: 300, h: 400});
    expect(overflow).toBe(0);
    expect(columns(placed).map(col => col.map(p => p.region))).toEqual([['눈썹', '광대', '인중', '얼굴형'], ['미간', '눈', '코', '입']]);
    for (const col of columns(placed)) for (let i = 1; i < col.length; i++) expect(col[i].top).toBeGreaterThanOrEqual(col[i - 1].top + estimateHeight(col[i - 1].text));
  });
  it('moves regions to the other column when the usual split does not fit, and reports what still sticks out', () => {
    // the three long ones on the right would not fit together with 미간
    const long = new Set(['눈', '코', '입']), all = Object.keys(DEFAULT_ANCHORS).map(r => box(r, long.has(r) ? 100 : 30));
    const {placed, overflow} = placeCallouts(all, DEFAULT_ANCHORS, {w: 300, h: 400});
    expect(overflow).toBe(0);
    for (const col of columns(placed)) {
      const last = col[col.length - 1]; expect(last.top + estimateHeight(last.text)).toBeLessThanOrEqual(100.01);
      expect(col.map(p => p.at[1])).toEqual([...col.map(p => p.at[1])].sort((a, b) => a - b));
    }
    expect(placeCallouts(Object.keys(DEFAULT_ANCHORS).map(r => box(r, 100)), DEFAULT_ANCHORS, {w: 300, h: 400}).overflow).toBeGreaterThan(0);
  });
  it('uses the measured heights once the boxes have been measured', () => {
    const {placed} = placeCallouts([box('눈썹', 30), box('광대', 30)], DEFAULT_ANCHORS, {w: 300, h: 400}, {scale: 1, heights: {눈썹: 40, 광대: 10}});
    const [first, second] = placed; expect(second.top - first.top).toBeGreaterThanOrEqual(40);
  });
  it('puts up to five passages in one column of wide cards and splits more evenly over two', () => {
    const sizes = (n: number) => splitColumns(Array.from({length: n}, (_, i) => i)).map(c => c.length);
    expect([1, 3, 5, 6, 7, 10].map(sizes)).toEqual([[1], [3], [5], [3, 3], [4, 3], [5, 5]]);
    expect(splitColumns([1, 2, 3, 4, 5, 6, 7]).flat()).toEqual([1, 2, 3, 4, 5, 6, 7]);  // reading order: left column first
  });
  it('shows the sheet for a server report and keeps the research tables below it', () => {
    const html = renderToStaticMarkup(<Results result={validateResult({status: 'COMPLETE', items: [], passages: [], research_report: report()})} onPreview={() => {}} onReset={() => {}}/>);
    for (const text of ['페이스텔러 분석 리포트', '합성 한줄평', '合成S-2', '연구용 상세 결과']) expect(html).toContain(text);
    expect(html.indexOf('fr-sheet')).toBeLessThan(html.indexOf('report-title'));
  });
});
