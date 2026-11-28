import {expect, test} from '@playwright/test';
import type {ResearchReport} from '../src/types';

// The written account of the processing on the progress screen, with a slow simulated drawing. Synthetic values only.
const good = {face: true, direction: true, size: true, brightness: true, expression: true, message: '합성 프레임 조건 충족'};
const judgement = (name: string, status = '확정') => ({name, region: '눈썹', method: 'CNN', value: null, class: '일자', confidence: .8, threshold: .6, status, probabilities: {일자: .8, 합성: .2}});
const passage = (i: number) => ({representative: `S-${i}`, requirement: ['F05 눈썹 휨:일자'], valence: '좋음', G: .9, regions: ['눈썹'], features: {F05: {class: '일자', confidence: .9, held: false}}, translation: `합성 번역 ${i}.`, source: '합성서'});
const report = {schema: 'faceteller-research-report-v1', structure: '합성 구조', judgements: {F05: judgement('눈썹 휨'), F04: judgement('눈썹 꼬리 길이', '보류')},
  plan_C_selected: [passage(0)], all_matching: [passage(0), passage(1)], blocked_by_hold: [], held_judgements: [], cnn_check: {},
  photos: [{yaw: 1.2}, {yaw: -27.6}, {yaw: 30.1}], face_width_px: 612.4, vggt: {network_seconds: .7}, seconds: 6.3, summary_info: {source: 'template'}} as unknown as ResearchReport;
const answer = JSON.stringify({type: 'progress', stage: '총평 작성'}) + '\n' + JSON.stringify({type: 'result', data: {status: 'COMPLETE', items: [], passages: [], research_report: report}}) + '\n';
const DRAW_MS = 6000, STEPS = 20;

for (const width of [1280, 390]) test(`progress screen writes out the processing while the drawing is made (${width}px)`, async ({page}) => {
  await page.setViewportSize({width, height: 900});
  let t0 = 0;
  await page.addInitScript(check => {window.__FACETELLER_TEST__ = {fixture: true, evaluate: async () => check};}, good);
  await page.route('**/api/v1/analysis', route => route.fulfill({status: 200, contentType: 'application/x-ndjson', body: answer}));
  await page.route('**/api/v1/stylize/healthz', route => route.fulfill({status: 200, contentType: 'application/json', body: '{}'}));
  await page.route('**/api/v1/stylize/progress/**', route => {const f = t0 ? (Date.now() - t0) / DRAW_MS : 0;
    return route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({state: f >= 1 ? 'done' : 'drawing', step: Math.min(STEPS, Math.floor(f * STEPS)), steps: STEPS})});});
  await page.route('**/api/v1/stylize/stylize', async route => {t0 = Date.now(); await new Promise(done => setTimeout(done, DRAW_MS)); await route.fulfill({status: 500, body: 'synthetic: no picture'}).catch(() => {});});
  await page.goto('/');
  await page.getByRole('button', {name: '이미지 파일 업로드', exact: true}).click();
  const data = await page.evaluate(() => {const c = document.createElement('canvas'); c.width = 640; c.height = 480; const x = c.getContext('2d')!; x.fillStyle = '#adb9c7'; x.fillRect(0, 0, 640, 480); return c.toDataURL('image/png').split(',')[1];});
  for (const d of ['front', 'left_oblique', 'right_oblique']) await page.locator('#file-' + d).setInputFiles({name: 'synthetic-solid.png', mimeType: 'image/png', buffer: Buffer.from(data, 'base64')});
  const log = page.locator('.process-log'), text = log.locator('.process-scroll'), lines = text.locator('li');
  await expect(log.getByRole('heading', {name: '처리 과정'})).toBeVisible();
  await expect(text).toContainText('얼굴 점 478개');                                                   // the first stage opens at once
  const early = await lines.count();
  await expect(text).toContainText('이번 사진의 좌우 각도: 정면 1.2°, 사선 −27.6°와 30.1°');            // a figure of this run
  await expect(text).toContainText('눈썹 휨: 일자 0.80, 합성 0.20 → 일자', {timeout: 8000});            // one line per feature, with its numbers
  await expect(text).toContainText('눈썹 꼬리 길이: 일자 0.80, 합성 0.20 → 보류 (신뢰도 0.80)');
  const gauges = await page.locator('.progress-list small').allTextContents();
  expect(gauges.slice(0, 2)).toEqual(['100%', '100%']); expect(gauges[5]).toBe('0%');                 // the gauges run on the same clock as the lines
  await expect(text).toContainText('이번 결과: 특징 2개 중 1개 확정, 1개 보류', {timeout: 8000});
  expect(await lines.count()).toBeGreaterThan(early);
  expect(await text.evaluate(el => el.scrollHeight - el.scrollTop - el.clientHeight < 40)).toBe(true);   // the newest line is in view
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await log.locator('li').evaluateAll(els => els.every(e => e.scrollWidth <= e.clientWidth + 1))).toBe(true);
  await expect(log).not.toContainText('단계 중');                                                      // the drawing's own step count is not shown
  await expect(page.getByRole('heading', {name: '페이스텔러 분석 리포트'})).toBeVisible({timeout: 12000});   // the results follow once the drawing request ends
});
