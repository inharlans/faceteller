import {describe, expect, it} from 'vitest';
import {LOG_DEFAULTS, STEPS, estimateFraction, featureLine, logPlan, logView, runFacts, stageView} from '../src/processSteps';
import {STAGES, type ReportJudgement, type ResearchReport} from '../src/types';

// Synthetic report; no real person's result.
const cnn = (name: string, status = '확정', confidence = .74): ReportJudgement => ({name, region: '눈', method: 'CNN', value: null, class: '보통', confidence, threshold: .6, status, probabilities: {'가는 편': .08, 보통: confidence, '넓은 편': .18}});
const measured = (name: string, method: string, value: number, cuts: number[], unit = '비율', status = '확정'): ReportJudgement => ({name, region: '코', method, value, class: cuts.length === 1 ? '높음' : '보통', confidence: .81, threshold: .6, status,
  measure: {label: '합성', value, unit, cuts, cut_classes: cuts.length === 1 ? ['보통', '높음'] : ['짧음', '보통', '김']}});
const passage = (i: number) => ({representative: `S-${i}`, requirement: ['F05 눈썹 휨:일자'], valence: '좋음', G: .9, regions: ['눈썹'], features: {}});
const report = {schema: 'faceteller-research-report-v1', judgements: {F09: measured('코 길이', '정면 수치', .3621, [.3552, .3769]), F02: cnn('눈 가늘기'), F03: cnn('눈꼬리 기울기', '보류', .52), F07: measured('미간', '수치', .2104, [.19, .23]),
    F06: measured('눈썹-눈 간격', '정면 수치', .5, [.4523, .6254], '비율', '보류'), F20: measured('광대 돌출', 'VGGT 맞춤식', -.12, [-.2983], '점수'),
    F10: {name: '코 높이', region: '코', method: 'VGGT 수치', value: null, class: null, confidence: null, threshold: .6, status: 'VGGT로 얼굴 점을 얻지 못함'}},
  plan_C_selected: [passage(0)], all_matching: [passage(0), passage(1), passage(2)], blocked_by_hold: [], held_judgements: [], cnn_check: {}, callouts: [{region: '눈', tone: '좋음', text: '합성'}],
  photos: [{yaw: 1.24}, {yaw: -27.6}, {yaw: 30}], face_width_px: 612.4, summary_info: {source: 'model', model: 'gemma3:12b', seconds: 4.2}} as unknown as ResearchReport;
const facts = runFacts(report)!;

describe('process account of the progress screen', () => {
  it('has an entry for every server stage', () => {expect(STEPS).toHaveLength(STAGES.length); for (const s of STEPS) expect(s.length).toBeGreaterThan(1);});
  it('writes one line per feature: probabilities or the measured value with its cuts, then the class', () => {
    expect(featureLine(report.judgements.F02)).toEqual({group: 'cnn', full: '눈 가늘기: 가는 편 0.08, 보통 0.74, 넓은 편 0.18 → 보통', short: '눈 가늘기: 보통 (신뢰도 0.74)'});
    expect(featureLine(report.judgements.F03).full).toBe('눈꼬리 기울기: 가는 편 0.08, 보통 0.52, 넓은 편 0.18 → 보류 (신뢰도 0.52)');
    expect(featureLine(report.judgements.F09)).toEqual({group: 'measured', full: '코 길이: 0.362 (경계 0.355, 0.377) → 보통, 신뢰도 0.81', short: '코 길이: 보통 (신뢰도 0.81)'});
    expect(featureLine(report.judgements.F06)).toMatchObject({full: '눈썹‑눈 간격: 0.500 (경계 0.452, 0.625) → 보류 (보통, 신뢰도 0.81)', short: '눈썹‑눈 간격: 보류 (보통, 신뢰도 0.81)'});
    expect(featureLine(report.judgements.F07).group).toBe('numeric');
    expect(featureLine(report.judgements.F20)).toMatchObject({group: 'depth', full: '광대 돌출: −0.12 (경계 −0.30) → 높음, 신뢰도 0.81'});
    expect(featureLine(report.judgements.F10)).toEqual({group: 'depth', full: '코 높이: VGGT로 얼굴 점을 얻지 못함', short: '코 높이: VGGT로 얼굴 점을 얻지 못함'});
  });
  it('reads the figures of the run and counts the judgements', () => {
    expect(facts).toMatchObject({yaw: [1.24, -27.6, 30], faceWidth: 612.4, features: 7, confirmed: 4, held: 2, unjudged: 1, matching: 3, selected: 1, regionSentences: 1, summarySource: 'model'});
    expect(facts.lines.map(l => l.group)).toEqual(['cnn', 'cnn', 'measured', 'numeric', 'measured', 'depth', 'depth']);   // in the order of the feature codes
    expect(runFacts(null)).toBeNull();
    expect(runFacts({...report, photos: [{yaw: 1}], summary_info: null} as unknown as ResearchReport)).toMatchObject({yaw: null, summarySource: null});
  });
  it('puts every feature under the stage that judges it, and the portrait last', () => {
    const plan = logPlan(STAGES, facts), at = (text: string) => plan.slots.find(s => s.text.startsWith(text))!;
    expect(plan.blocks).toEqual([...STAGES, '인물화 그리기']);
    expect([at('눈 가늘기').stage, at('미간').stage, at('코 길이').stage, at('광대 돌출').stage, at('코 높이').stage]).toEqual([2, 2, 2, 3, 3]);
    expect(plan.slots.slice(-2).map(s => [s.stage, s.block])).toEqual([[5, 6], [5, 6]]);
    expect(plan.factsFrom).toBe(1);
    expect(plan.slots.every(s => !s.pending && s.text)).toBe(true);
    const live = plan.slots.filter(s => s.live && !/^[가-힣‑ ]+: /.test(s.text) || s.text.startsWith('이번')).map(s => s.text);
    expect(live).toEqual(['이번 사진의 좌우 각도: 정면 1.2°, 사선 −27.6°와 30.0°', '이번 정면 사진의 얼굴 폭: 612픽셀', '이번 결과: 특징 7개 중 4개 확정, 2개 보류, 1개 판정 불가',
      '이번 결과: 조건이 맞는 조항 3개 중 1개 선정', '이번 결과: 설명이 붙는 부위 1곳', '이번 총평: 언어 모델(gemma3:12b)이 작성']);
  });
  it('follows the options: no feature lines, short ones, no figures, no portrait block', () => {
    expect(logPlan(STAGES, facts, {...LOG_DEFAULTS, features: false}).slots.some(s => s.text.startsWith('눈 가늘기'))).toBe(false);
    expect(logPlan(STAGES, facts, {...LOG_DEFAULTS, detail: 'short'}).slots.some(s => s.text === '눈 가늘기: 보통 (신뢰도 0.74)')).toBe(true);
    expect(logPlan(STAGES, facts, {...LOG_DEFAULTS, figures: false}).slots.some(s => s.text.startsWith('이번'))).toBe(false);
    expect(logPlan(STAGES, facts, {...LOG_DEFAULTS, drawing: false}).blocks).toEqual(STAGES);
    expect(logPlan(STAGES, null, {...LOG_DEFAULTS, features: false, figures: false}).factsFrom).toBe(1);
  });
  it('shares the wait between the lines and gives every stage the part its lines take', () => {
    for (const pace of ['chars', 'lines', 'stages'] as const) {
      const plan = logPlan(STAGES, facts, {...LOG_DEFAULTS, pace});
      expect(plan.starts[0]).toBe(0); expect(plan.bounds[0]).toBe(0); expect(plan.bounds[STAGES.length]).toBe(1);
      expect(plan.starts.every((s, i) => i === 0 || s > plan.starts[i - 1])).toBe(true); expect(plan.bounds.every((b, i) => i === 0 || b > plan.bounds[i - 1])).toBe(true);
    }
    const even = logPlan(STAGES, facts, {...LOG_DEFAULTS, pace: 'stages'}).bounds;
    even.forEach((b, i) => expect(b).toBeCloseTo(i / STAGES.length, 9));
    const lines = logPlan(STAGES, facts, {...LOG_DEFAULTS, pace: 'lines'});
    expect(lines.starts[1] - lines.starts[0]).toBeCloseTo(1 / lines.slots.length, 9);
  });
  it('waits before the first figure until the answer is in, keeping the time of the lines to come', () => {
    const plan = logPlan(STAGES, null);
    expect(plan.factsFrom).toBeGreaterThan(0); expect(plan.factsFrom).toBeLessThan(plan.bounds[1]);
    expect(plan.slots.filter(s => s.pending)).toHaveLength(6 + 19);                                    // 6 figures and the 19 features of the structure
    expect(logView(plan, 1).blocks.flatMap(b => b.lines).every(l => l.text)).toBe(true);                // a line without its answer is never shown
    expect(logView(plan, plan.factsFrom - 1e-6).blocks[0].lines).toHaveLength(3);
  });
  it('turns the clock into the stage and its gauge', () => {
    const b = [0, .2, .5, 1];
    expect(stageView(b, -1)).toEqual({paced: -1, percents: [0, 0, 0]});
    expect(stageView(b, 0)).toEqual({paced: 0, percents: [0, 0, 0]});
    expect(stageView(b, .38)).toEqual({paced: 1, percents: [100, 60, 0]});
    expect(stageView(b, 1)).toEqual({paced: 2, percents: [100, 100, 99]});
    expect(estimateFraction(30_000, 60_000)).toBe(.5); expect(estimateFraction(600_000, 60_000)).toBeLessThan(1); expect(estimateFraction(70_000, 60_000)).toBeGreaterThan(.9);
  });
  it('opens the lines in order and reports how far the newest one is', () => {
    const plan = logPlan(STAGES, facts), mid = (plan.starts[1] + plan.starts[2]) / 2, v = logView(plan, mid);
    expect(v.blocks.map(b => b.lines.length)).toEqual([2, 0, 0, 0, 0, 0, 0]); expect(v.blocks[0].state).toBe('active'); expect(v.blocks[1].state).toBe('waiting'); expect(v.newest).toBeCloseTo(.5, 6);
    const end = logView(plan, 1);
    expect(end.blocks.every(b => b.state === 'done')).toBe(true); expect(end.blocks.reduce((n, b) => n + b.lines.length, 0)).toBe(plan.slots.length); expect(end.newest).toBe(1);
    expect(logView(plan, -1).blocks.every(b => b.state === 'waiting' && !b.lines.length)).toBe(true);
    expect(logView(plan, plan.bounds[1]).blocks[0].state).toBe('done');
  });
});
