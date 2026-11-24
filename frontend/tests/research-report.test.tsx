import {beforeAll, describe, expect, it} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {Results} from '../src/Results';
import {ResearchReportView, conditionText, cutText, leftOutReason, scanText, withUnit} from '../src/ResearchReport';
import {validateResult, validReport} from '../src/api';
import type {AnalysisResult, ReportPassage, ResearchReport} from '../src/types';

// Authored synthetic values only; no real person's result. Temporary research view (2026-11-16).
const passage = (id: string, requirement: string[], valence: string, G: number, feature: string, cls: string) =>
  ({representative: id, requirement, valence, G, regions: ['합성 부위'], features: {[feature]: {class: cls, confidence: G, held: false}}, translation: `합성 번역 ${id}`, source: '합성 문헌',
    original_text: `合成原文${id}`, commentary: `합성 해설 ${id}`, annotation_status: 'ai_annotated_unreviewed', translation_kind: 'korean_literal_ai_draft', split_of: null,
    source_detail: {title: '合成書', title_ko: '합성서', chapter: '合成 第1卷', section: '合成節', char_span: [10, 22], scan_page: 'Page:Gujin Tushu Jicheng, Volume 400 (1700-1725).djvu/7', scan_page_end: 'Page:Gujin Tushu Jicheng, Volume 400 (1700-1725).djvu/7',
      attributed_to: null, url: 'https://example.org/source', license: '합성 이용 조건', text_witness: '合成底本'}});
const src = {title: '合成書', title_ko: '합성서', chapter: '合成 第2卷', section: '合成節二', char_span: [1, 9], scan_page: 'Page:Gujin Tushu Jicheng, Volume 401 (1700-1725).djvu/3', url: 'https://example.org/entry', license: '합성 이용 조건'};
const entry = (id: string, valence: string, translation: string, source: string, original: string) => ({id, valence, topics: ['합성 주제'], translation, source, original_text: original, commentary: `합성 조항 해설 ${id}`, source_detail: src, annotation_status: 'ai_annotated_unreviewed'});
const withLines = (p: ReportPassage): ReportPassage => ({...p, entry_count: 3, hidden_unfavourable: 2, omitted_entries: 1, lines: [
  {topic: '합성 주제 가', translation: '합성 내용 하나', valence: p.valence, sources: ['합성서', '합성 다른 책'], entries: [entry(`${p.representative}-e1`, p.valence, '합성 내용 하나', '합성서', '合成條一'), entry(`${p.representative}-e2`, p.valence, '합성 내용 하나와 비슷한 말', '합성 다른 책', '合成條二')]},
  {topic: '합성 주제 나', translation: '합성 내용 둘', valence: '중립', sources: ['합성서'], entries: [entry(`${p.representative}-e3`, '중립', '합성 내용 둘', '합성서', '合成條三')]}]});
const report = (): ResearchReport => ({schema: 'faceteller-research-report-v1', structure: '합성 구조',
  judgements: {F09: {name: '코 길이', region: '코', method: '정면 수치', value: .3712, class: '보통', confidence: .81, threshold: .755, status: '확정',
      measure: {label: '합성 길이 비율', value: .3712, unit: '비율', used_for_judgement: true, cuts: [.35, .38], cut_classes: ['짧음', '보통', '김'], components: [{label: '합성 구성값', value: 12.34, unit: '도'}]}},
    F05: {name: '눈썹 휨', region: '눈썹', method: 'CNN', value: null, class: '일자', confidence: .91, threshold: .6, status: '확정', probabilities: {일자: .91, 곡선: .09},
      reference: {label: '합성 휨 정도', value: .0421, unit: '비율', components: [{label: '합성 꺾임 각도', value: 7.5, unit: '도'}]}},
    F15: {name: '인중 길이', region: '인중', method: 'CNN', value: null, class: '보통', confidence: .55, threshold: .6, status: '보류'},
    F20: {name: '광대 돌출', region: '광대', method: 'VGGT 맞춤식', value: -.5, class: '보통', confidence: .7, threshold: null, status: '확정'}},
  plan_C_selected: [passage('S-1', ['F05 눈썹 휨:일자'], '좋음', .91, 'F05', '일자')],
  all_matching: [passage('S-1', ['F05 눈썹 휨:일자'], '좋음', .91, 'F05', '일자'), passage('S-2', ['F18 눈썹 숱:성김', 'F04 눈썹 꼬리 길이:긴 편+F06 눈썹-눈 간격:넓음'], '나쁨', .77, 'F18', '성김')],
  blocked_by_hold: [], held_judgements: [{feature: 'F15', name: '인중 길이', class: '보통', confidence: .55, threshold: .6, reason: '이 칸을 조건으로 쓰는 조항이 없음'}],
  cnn_check: {F09: {'합성 확인 모델': {class: '김', confidence: .66}}}});
const result = (r: unknown): AnalysisResult => validateResult({status: 'COMPLETE', items: [], passages: [], research_report: r});
// The research tables are tested on their own: Results now wraps them in the report sheet (tests/face-report.test.tsx).
const render = (r: AnalysisResult) => renderToStaticMarkup(r.research_report ? <ResearchReportView report={r.research_report} notice={r.notice} onReset={() => {}}/> : <Results result={r} onPreview={() => {}} onReset={() => {}}/>);

beforeAll(() => {(globalThis as unknown as {window: object}).window = {};});

describe('temporary research report view', () => {
  it('replaces the service cards and shows every table of the report', () => {
    const html = render(result(report()));
    expect(html).toContain('연구용 상세 결과');
    expect(html).not.toContain('생김새로 본 관상 풀이');
    for (const text of ['고른 조항 (C안, 최대 5개)', '그 밖에 해당하는 조항 (1개)', '조건이 맞는 조항은 모두 2개입니다. 위에 고른 1개 말고 아래 1개가 더 있습니다.', '보류 때문에 나오지 못한 조항은 없습니다.', '보류된 판정 (1개)', '전체 판정과 측정값 (4개)', '확인용 CNN (조항에는 쓰지 않음)']) expect(html).toContain(text);
    expect(html).toContain('눈썹 휨: 일자'); expect(html).toContain('합성 번역 S-1');
    expect(html).toContain('선정 규칙에서 빠짐'); expect(html).toContain('이 칸을 조건으로 쓰는 조항이 없음');
    expect(html).toContain('<td class="num">0.55</td>'); expect(html).toContain('<td class="num">없음</td>'); expect(html).toContain('합성 확인 모델');
    expect(html).toContain('정답 확률이 아닙니다');
  });

  it('shows the original text, the place in the source and the review state of each passage', () => {
    const html = render(result(report()));
    expect(html).toContain('<p class="report-original" lang="zh">合成原文S-1</p>'); expect(html).toContain('합성 해설 S-1');
    for (const text of ['합성서 (合成書)', '合成 第1卷', '合成節', '그 절 본문의 10~22번째 글자', '고금도서집성 스캔본 400책 7쪽', 'href="https://example.org/source"', '合成底本', '자동 작성한 초안이며 사람이 검수하기 전', '문헌 원문의 이용 조건: 합성 이용 조건']) expect(html).toContain(text);
    expect(scanText('Page:Other Book.djvu/3')).toBe('Other Book.djvu/3'); expect(scanText(null)).toBe(''); expect(scanText('pdf_page:31')).toBe('스캔 PDF 31쪽');
  });

  it('shows what was measured: value with unit, cuts, parts, CNN probabilities and the reference measure', () => {
    const html = render(result(report()));
    for (const text of ['합성 길이 비율', '경계: 0.350 미만 짧음, 0.380 이상 김, 그 사이 보통', '합성 구성값 <b>12.3°</b>', '일자 0.91 · 곡선 0.09', '참고 측정(판정에는 쓰지 않음): 합성 휨 정도 <b>0.042</b>', '합성 꺾임 각도 <b>7.5°</b>']) expect(html).toContain(text);
    expect(html).toContain('<td class="num">0.371</td>');
    expect(withUnit(102.18, '도')).toBe('102.2°'); expect(withUnit(-0.8633, '점수')).toBe('-0.86'); expect(withUnit(null, '비율')).toBe('–');
    expect(cutText({label: 'x', value: 1, unit: '점수', cuts: [.99], cut_classes: ['보통', '높음']})).toBe('0.99 이상 높음, 미만 보통');
  });

  it('lists the matching passages that were not selected without repeating the selected ones', () => {
    const html = render(result(report())); const rest = html.slice(html.indexOf('그 밖에 해당하는 조항'), html.indexOf('보류 때문에 나오지 못한 조항'));
    expect(rest).toContain('합성 번역 S-2'); expect(rest).not.toContain('합성 번역 S-1'); expect(html.split('합성 번역 S-1').length).toBe(2);  // the selected passage appears once on the page
    const all = report(); all.all_matching = [all.all_matching[0]];
    expect(render(result(all))).toContain('조건이 맞는 조항은 위에 고른 1개가 전부입니다.');
    const none = report(); none.plan_C_selected = []; none.all_matching = [];
    expect(render(result(none))).toContain('그 밖에 해당하는 조항 (0개)');
    const p = (valence: string) => ({valence}) as ReportPassage;
    expect(leftOutReason(p('나쁨'), [p('나쁨'), p('나쁨'), p('좋음')])).toBe('나쁜 풀이는 2개까지');
    expect(leftOutReason(p('좋음'), [p('나쁨'), p('나쁨'), p('좋음'), p('좋음'), p('중립')])).toBe('5개가 먼저 참');
    expect(leftOutReason(p('나쁨'), [p('나쁨'), p('좋음'), p('좋음'), p('좋음'), p('중립')])).toBe('5개가 먼저 참');
  });

  it('shows one line per kind of statement with every book in brackets, and the passages behind it', () => {
    const r = report(); r.plan_C_selected = [withLines(r.plan_C_selected[0])]; r.all_matching = [withLines(r.all_matching[0]), r.all_matching[1]];
    const html = render(result(r));
    expect(html).toContain('<p class="report-line-main">합성 내용 하나 <span class="report-book">[합성서, 합성 다른 책]</span></p>');
    // only the sentences and their books: no counts, topic labels or notes about hidden passages around them
    for (const extra of ['이 생김새를 다룬 문헌 조항', '내용 2가지', 'report-topic', '보여 주지 않습니다', '뺐습니다']) expect(html).not.toContain(extra);
    expect(html).toContain('이 내용의 조항 2개: 원문과 위치'); expect(html).toContain('<summary>원문과 위치</summary>');
    for (const text of ['합성 내용 하나와 비슷한 말', '[합성 다른 책]', '合成條二', '合成 第2卷', '고금도서집성 스캔본 401책 3쪽']) expect(html).toContain(text);
    // a line whose direction differs from the group's carries its own tag; a line in the group's direction does not repeat it
    const second = html.slice(html.indexOf('합성 내용 둘 <span'), html.indexOf('합성 내용 둘 <span') + 220); expect(second).toContain('data-valence="neutral">중립</span>');
    const first = html.slice(html.indexOf('합성 내용 하나 <span'), html.indexOf('합성 내용 하나 <span') + 120); expect(first).not.toContain('class="valence"');
    // the group without lines still falls back to its single passage
    expect(html).toContain('합성 번역 S-2');
    const bad = report() as unknown as {plan_C_selected: Record<string, unknown>[]}; bad.plan_C_selected[0].lines = [{topic: 'x'}];
    expect(validReport(bad)!.plan_C_selected[0].lines).toBeUndefined();
  });

  it('puts the numbers behind a condition on its card: measured value and cuts, or CNN probabilities and the reference measure', () => {
    const r = report(); r.plan_C_selected = [passage('S-9', ['F09 코 길이:보통'], '좋음', .81, 'F09', '보통'), r.plan_C_selected[0]];
    const html = render(result(r)), card = (id: string) => {const a = html.indexOf(`합성 번역 ${id}`); return html.slice(html.lastIndexOf('<article', a), a);};
    expect(card('S-9')).toContain('<b>코 길이</b> <span class="report-number">0.371</span> <span>→ 보통</span> <span class="report-g">신뢰도 0.81</span>');
    for (const text of ['합성 길이 비율', '경계: 0.350 미만 짧음, 0.380 이상 김, 그 사이 보통', '합성 구성값 <b>12.3°</b>']) expect(card('S-9')).toContain(text);
    expect(card('S-1')).toContain('<b>눈썹 휨</b>  <span>→ 일자</span>'); expect(card('S-1')).toContain('일자 0.91 · 곡선 0.09'); expect(card('S-1')).toContain('합성 꺾임 각도 <b>7.5°</b>');
  });

  it('drops only a malformed optional detail and keeps the rest of the report', () => {
    const r = report() as unknown as {judgements: Record<string, Record<string, unknown>>; plan_C_selected: Record<string, unknown>[]};
    r.judgements.F09.measure = {label: 'x'}; r.plan_C_selected[0].source_detail = {char_span: 'bad'};
    const ok = validReport(r)!; expect(ok).not.toBeNull();
    expect(ok.judgements.F09.measure).toBeUndefined(); expect(ok.plan_C_selected[0].source_detail).toBeUndefined(); expect(ok.judgements.F05.reference).toBeDefined();
  });

  it('writes conditions without the internal feature codes', () => {
    expect(conditionText(['F17 턱 폭:좁음'])).toBe('턱 폭: 좁음');
    expect(conditionText(['F18 눈썹 숱:짙음', 'F04 눈썹 꼬리 길이:긴 편+F06 눈썹-눈 간격:넓음'])).toBe('눈썹 숱: 짙음 + 눈썹 꼬리 길이: 긴 편 그리고 눈썹-눈 간격: 넓음');
  });

  it('drops a malformed report instead of repairing it, and keeps the normal view', () => {
    const bad = {...report(), plan_C_selected: [{representative: 'x'}]};
    expect(validReport(bad)).toBeNull();
    expect(validReport({...report(), schema: 'other'})).toBeNull();
    expect(validReport({...report(), plan_C_selected: Array.from({length: 6}, (_, i) => passage(`S-${i}`, ['F05 눈썹 휨:일자'], '좋음', .91, 'F05', '일자'))})).toBeNull();
    const html = render(result(bad));
    expect(html).not.toContain('연구용 상세 결과'); expect(html).toContain('생김새로 본 관상 풀이');
  });

  it('results without a report are untouched', () => {
    expect(validateResult({status: 'COMPLETE', items: [], passages: []}).research_report).toBeUndefined();
  });
});
