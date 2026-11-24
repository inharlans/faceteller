import {expect, test, type Page} from '@playwright/test';
import {mkdirSync, readFileSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import type {ReportJudgement, ReportPassage, ResearchReport} from '../src/types';

// The report sheet through the real page flow: three uploads, a mocked server answer with a research report, the
// sheet on screen and on A4. Authored synthetic values only; no real person's result.
const artifacts = path.join(os.tmpdir(), 'faceteller-release-frontend'); mkdirSync(artifacts, {recursive: true});
const good = {face: true, direction: true, size: true, brightness: true, expression: true, message: '합성 프레임 조건 충족'};
const judgement = (name: string, region: string, cls: string, status = '확정'): ReportJudgement => ({name, region, method: 'CNN', value: null, class: cls, confidence: .8, threshold: .6, status, probabilities: {[cls]: .8, 합성: .2}});
const passage = (i: number, original = `合成原文${i}。`, translation = `합성 번역 ${i}번 문장이다.`): ReportPassage => ({representative: `S-${i}`, requirement: ['F05 눈썹 휨:일자'], valence: i % 4 ? '좋음' : '나쁨', G: .95 - i / 50, regions: ['눈썹'],
  features: {F05: {class: '일자', confidence: .9, held: false}}, translation, source: '합성서', original_text: original, source_detail: {title_ko: '합성서', section: '合成節'}});
// Real feature names and classes, so the sentences come from the page's copy of the team's table.
const report = (passages: number): ResearchReport => ({schema: 'faceteller-research-report-v1', structure: '합성 구조', summary: '합성 한줄평입니다.',
  judgements: {F01: judgement('눈 크기', '눈', '큰 편'), F02: judgement('눈 가늘기', '눈', '보통'), F03: judgement('눈꼬리 기울기', '눈', '올라간 편'), F05: judgement('눈썹 휨', '눈썹', '일자'),
    F04: judgement('눈썹 꼬리 길이', '눈썹', '긴 편'), F06: judgement('눈썹-눈 간격', '눈썹', '넓음'), F18: judgement('눈썹 숱', '눈썹', '짙음'), F07: judgement('미간', '미간', '넓음'),
    F15: judgement('인중 길이', '인중', '보통'), F17: judgement('턱 폭', '얼굴형', '넓음'), F20: judgement('광대 돌출', '광대', '높음'), F12: judgement('입 폭', '입', '넓은 편', '보류')},
  plan_C_selected: Array.from({length: Math.min(passages, 5)}, (_, i) => passage(i)), all_matching: Array.from({length: passages}, (_, i) => passage(i)), blocked_by_hold: [], held_judgements: [], cnn_check: {}});
const answer = (r: ResearchReport) => JSON.stringify({type: 'progress', stage: '문헌 선정'}) + '\n' + JSON.stringify({type: 'result', data: {status: 'COMPLETE', items: [], passages: [], research_report: r}}) + '\n';

async function analyse(page: Page, r: ResearchReport) {
  await page.addInitScript(check => {window.__FACETELLER_TEST__ = {fixture: true, evaluate: async () => check};}, good);
  await page.route('**/api/v1/analysis', route => route.fulfill({status: 200, contentType: 'application/x-ndjson', body: answer(r)}));
  await page.goto('/');
  await page.getByRole('button', {name: '이미지 파일 업로드', exact: true}).click();
  const data = await page.evaluate(() => {const c = document.createElement('canvas'); c.width = 640; c.height = 480; const x = c.getContext('2d')!; x.fillStyle = '#adb9c7'; x.fillRect(0, 0, 640, 480); return c.toDataURL('image/png').split(',')[1];});
  for (const d of ['front', 'left_oblique', 'right_oblique']) await page.locator('#file-' + d).setInputFiles({name: 'synthetic-solid.png', mimeType: 'image/png', buffer: Buffer.from(data, 'base64')});
  await expect(page.getByRole('heading', {name: '페이스텔러 분석 리포트'})).toBeVisible();
  await page.evaluate(() => document.fonts.ready); await page.waitForTimeout(200);
}
/** What must hold for a sheet (on screen or the printer's copy): nothing cut off, nothing outside its area, no box on another. */
const faults = (page: Page, root: string) => page.evaluate(root => {
  const all = (q: string) => [...document.querySelectorAll<HTMLElement>(`${root} ${q}`)], shown = (e: HTMLElement) => e.offsetParent !== null;
  const zone = document.querySelector(`${root} .fr-face`)!.getBoundingClientRect(), boxes = all('.fr-box').map(e => e.getBoundingClientRect()), sheet = document.querySelector<HTMLElement>(`${root} .fr-page`)!;
  return {cutText: all('.fr-card p, .fr-card b, .fr-card li, .fr-box p').filter(e => shown(e) && (e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1)).length,
    clippedCards: all('.fr-card').filter(e => e.scrollHeight > e.clientHeight + 1).length, fullColumns: all('.fr-col').filter(e => e.scrollHeight > e.clientHeight + 1).length,
    boxesOutside: boxes.filter(b => b.top < zone.top - 1 || b.bottom > zone.bottom + 1).length,
    boxesOverlap: boxes.filter((a, i) => boxes.some((b, j) => i < j && Math.abs(a.left - b.left) < 2 && a.top < b.bottom - .5 && b.top < a.bottom - .5)).length,
    pageOverflow: Math.max(0, sheet.scrollHeight - sheet.clientHeight)};
}, root);
const clean = {cutText: 0, clippedCards: 0, fullColumns: 0, boxesOutside: 0, boxesOverlap: 0, pageOverflow: 0};
const SCREEN = '.face-report > .fr-sheet', PRINT = '.fr-print .fr-sheet';
async function bothClean(page: Page) {
  expect(await faults(page, SCREEN)).toEqual(clean);
  await page.emulateMedia({media: 'print'}); expect(await faults(page, PRINT)).toEqual(clean); await page.emulateMedia({media: null});
}

test('server report becomes the one-page sheet: region sentences, summary, passages, research tables kept below', async ({page}) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await analyse(page, report(7));
  const sheet = page.locator(SCREEN);
  await expect(sheet.locator('.fr-box')).toHaveCount(5);  // 눈·눈썹·미간·턱·광대 have a sentence; 인중 보통 has none, 입 is held, 코 was not judged
  expect((await sheet.locator('.fr-box h2').allInnerTexts()).sort()).toEqual(['광대', '눈', '눈썹', '미간', '턱'].sort());
  await expect(sheet.locator('.fr-summary')).toContainText('합성 한줄평입니다.');
  await expect(sheet.locator('.fr-card')).toHaveCount(7);
  expect(await sheet.locator('.fr-col').evaluateAll(cols => cols.map(c => c.querySelectorAll('.fr-card').length))).toEqual([4, 3]);
  await expect(sheet.locator('.fr-card').first()).toContainText('合成原文0。');
  await bothClean(page);
  await expect(page.locator('.fr-backup')).toContainText('연구용 상세 결과');
  const pdf = path.join(artifacts, 'report-sheet.pdf'); await page.pdf({path: pdf, preferCSSPageSize: true, printBackground: true});
  const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs'), doc = await pdfjs.getDocument({data: new Uint8Array(readFileSync(pdf))}).promise;
  expect(doc.numPages).toBe(1);
  const text = (await (await doc.getPage(1)).getTextContent()).items.map(i => 'str' in i ? i.str : '').join('').replace(/\s/g, '');
  for (const part of ['페이스텔러분석리포트', '합성한줄평입니다.', '合成原文6。', '합성번역6번문장이다.']) expect(text).toContain(part);
  expect(text).not.toContain('연구용상세결과');  // the paper carries the sheet only
  await doc.loadingTask.destroy();
  expect(errors).toEqual([]);
});

test('sentences sent by the server are shown as they are', async ({page}) => {
  const r = report(3); r.callouts = [{region: '입', tone: '좋음', text: '서버가 보낸 입 문구입니다.'}, {region: '코', tone: '조심', text: '서버가 보낸 코 문구입니다.'}];
  await analyse(page, r);
  expect((await page.locator(`${SCREEN} .fr-box`).allInnerTexts()).map(t => t.replace(/\s+/g, ' ')).sort()).toEqual(['입 서버가 보낸 입 문구입니다.', '코 서버가 보낸 코 문구입니다.']);
  await bothClean(page);
});

test('up to five passages are wide cards of about the same height; none shows a note instead', async ({page}) => {
  await analyse(page, report(4));
  const cards = page.locator(`${SCREEN} .fr-card`); await expect(cards).toHaveCount(4);
  const heights = await cards.evaluateAll(list => list.map(c => c.getBoundingClientRect().height));
  expect(Math.max(...heights) - Math.min(...heights)).toBeLessThan(2);
  await expect(page.locator(`${SCREEN} .fr-cards`)).toHaveAttribute('data-cols', '1');
  await bothClean(page);
  const r = report(0); r.judgements = {F15: judgement('인중 길이', '인중', '보통')};
  await page.unroute('**/api/v1/analysis'); await analyse(page, r);
  await expect(page.locator(`${SCREEN} .fr-card`)).toHaveCount(0); await expect(page.locator(`${SCREEN} .fr-box`)).toHaveCount(0);
  await expect(page.locator(SCREEN)).toContainText('조건이 맞는 문헌 기록이 없습니다.'); await expect(page.locator(SCREEN)).toContainText('풀이를 붙일 수 있는 부위가 없습니다.');
  await bothClean(page);
});

test('long sentences and long passages are never cut: smaller type first, then fewer passages with a count', async ({page}) => {
  const long = '合成長文'.repeat(21), translation = '합성으로 길게 쓴 번역 문장입니다. '.repeat(10).trim(), r = report(12);
  r.all_matching = r.all_matching.map((p, i) => passage(i, long, translation)); r.plan_C_selected = r.all_matching.slice(0, 5);
  r.callouts = ['눈썹', '눈', '미간', '코', '광대', '인중', '입', '얼굴형'].map(region => ({region, tone: '섞임', text: '합성으로 길게 쓴 부위 문구입니다. '.repeat(5).trim()}));
  await analyse(page, r);
  await expect(page.locator(`${SCREEN} .fr-box`)).toHaveCount(8);
  const shown = await page.locator(`${SCREEN} .fr-card`).count();
  expect(shown).toBeGreaterThan(0); expect(shown).toBeLessThan(10);
  await expect(page.locator(`${SCREEN} .fr-count`)).toHaveText(`조건이 맞는 기록 12개 중 신뢰도 높은 ${shown}개`);
  await expect(page.locator(`${SCREEN} .fr-card`).first()).toContainText(translation);
  await bothClean(page);
});
