import {expect,test,type Page} from '@playwright/test';
import {mkdirSync,readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import {fixture,overlay19} from './fixtures';
import type {Passage} from '../src/types';
const artifacts=path.join(os.tmpdir(),'faceteller-release-frontend');mkdirSync(artifacts,{recursive:true});
const good={face:true,direction:true,size:true,brightness:true,expression:true,message:'합성 프레임 조건 충족'};
const payload=(data=fixture())=>JSON.stringify({type:'progress',stage:'문헌 선정'})+'\n'+JSON.stringify({type:'result',data})+'\n';
async function syntheticImage(page:Page){const data=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=640;c.height=480;const x=c.getContext('2d')!;x.fillStyle='#adb9c7';x.fillRect(0,0,640,480);return c.toDataURL('image/png').split(',')[1];});return {name:'synthetic-solid.png',mimeType:'image/png',buffer:Buffer.from(data,'base64')};}
async function upload(page:Page){await page.getByRole('button',{name:'이미지 파일 업로드',exact:true}).click();const image=await syntheticImage(page);for(const d of ['front','left_oblique','right_oblique'])await page.locator('#file-'+d).setInputFiles(image);}
async function testAdapter(page:Page){await page.addInitScript(check=>{window.__FACETELLER_TEST__={fixture:true,evaluate:async()=>check};const oldCreate=URL.createObjectURL.bind(URL),oldRevoke=URL.revokeObjectURL.bind(URL);const active=new Set<string>();(window as unknown as {testURLs:Set<string>}).testURLs=active;URL.createObjectURL=blob=>{const u=oldCreate(blob);active.add(u);return u;};URL.revokeObjectURL=u=>{active.delete(u);oldRevoke(u);};},good);}
test.beforeEach(async({page})=>{await page.route('**/api/v1/analysis',route=>route.fulfill({status:200,contentType:'application/x-ndjson',body:payload()}));});
test('mockup home, 3 slots, mixed results, mobile, complete single-page A4',async({page})=>{
  await testAdapter(page);const errors:string[]=[],external:string[]=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'||m.type()==='warning')errors.push(m.text());});page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith('http://127.0.0.1:5173/'))external.push(new URL(r.url()).origin);});
  let requests=0;
  await page.route('**/api/v1/analysis',async route=>{requests++;const body=route.request().postDataBuffer()!.toString('latin1');for(const field of ['front','left_oblique','right_oblique','input_mode'])expect(body).toContain(`name="${field}"`);expect(body).toContain('upload');await route.fulfill({status:200,contentType:'application/x-ndjson',body:payload(fixture(5,3))});});
  await page.goto('/');await expect(page).toHaveTitle('페이스텔러 · 얼굴 형태와 문헌');await expect(page.locator('h1')).toContainText('얼굴 형태를 분류하고');await page.screenshot({path:path.join(artifacts,'home-desktop.png')});
  await page.getByRole('button',{name:'이미지 파일 업로드',exact:true}).click();const im=await syntheticImage(page);await page.locator('#file-front').setInputFiles(im);await page.locator('#file-left_oblique').setInputFiles(im);await expect(page.locator('.valid')).toHaveCount(2);expect(requests).toBe(0);await page.locator('#file-right_oblique').setInputFiles(im);
  await expect(page.locator('#result-title')).toBeVisible();expect(requests).toBe(1);await expect(page.locator('.passage-card')).toHaveCount(5);await expect(page.locator('.type-card')).toHaveCount(0);await expect(page.locator('#passages-title')).toHaveText('관상 풀이');await expect(page.getByText('해당하는 관상 유형')).toHaveCount(0);await expect(page.locator('.stage h2')).toHaveCount(1);await expect(page.locator('.passage-card').first()).toContainText('일치 정도 90.0점');
  await page.locator('.passage-card').first().getByText('원문과 출처',{exact:true}).click();await expect(page.locator('.passage-card').first()).toContainText('코뿌리 높이 · 높은 편 — 깊이 경계점수 86.4점');await expect(page.locator('.passage-card').first()).toContainText('정답 확률이 아닙니다');
  expect(await page.evaluate(()=>({photos:document.querySelectorAll('img').length,urls:(window as unknown as {testURLs:Set<string>}).testURLs.size,local:localStorage.length,session:sessionStorage.length}))).toEqual({photos:0,urls:0,local:0,session:0});
  await page.screenshot({path:path.join(artifacts,'results-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});await expect(page.locator('#result-title')).toBeVisible();expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);await page.screenshot({path:path.join(artifacts,'results-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await page.evaluate(()=>document.fonts.ready);await expect(page.getByRole('button',{name:'A4 한 장 인쇄',exact:true})).toBeEnabled();await expect(page.locator('#print-sheet')).not.toContainText('FIXTURE_COMMENTARY_ONLY');await expect(page.locator('#print-sheet')).toContainText('5. 합성 유형 5 · 길흉이 섞인 기록');
  const pdf=path.join(artifacts,'synthetic-a4-five-passages.pdf');await page.pdf({path:pdf,format:'A4',preferCSSPageSize:true,printBackground:true});const pdfjs=await import('pdfjs-dist/legacy/build/pdf.mjs');const pdfDocument=await pdfjs.getDocument({data:new Uint8Array(readFileSync(pdf))}).promise;expect(pdfDocument.numPages).toBe(1);const text=(await (await pdfDocument.getPage(1)).getTextContent()).items.map(i=>'str' in i?i.str:'').join('').replace(/\s/g,'');for(const p of fixture(5,3).passages){expect(text).toContain(p.original_text.replace(/\s/g,''));expect(text).toContain(p.translation.replace(/\s/g,''));}expect(text).not.toContain('FIXTURE_COMMENTARY_ONLY');await pdfDocument.loadingTask.destroy();
  expect(errors).toEqual([]);expect(external).toEqual([]);expect(await page.locator('vite-error-overlay').count()).toBe(0);
  await page.getByRole('button',{name:'결과로 돌아가기',exact:true}).click();await expect(page.locator('#result-title')).toBeVisible();await page.getByRole('button',{name:'다시 시작',exact:true}).click();await expect(page.locator('#print-sheet')).toHaveCount(0);
});
test('permission denied offers upload, no inference or retained stream',async({page})=>{await testAdapter(page);await page.addInitScript(()=>{Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{throw new DOMException('denied','NotAllowedError');}});});await page.goto('/');await page.getByRole('button',{name:'웹캠으로 촬영하기'}).click();await expect(page.locator('#upload-title')).toBeVisible();await expect(page.getByRole('alert')).toContainText('권한이 거부');await expect(page.locator('video')).toHaveCount(0);});
test('synthetic video auto-captures 3 directions exactly once and stops tracks',async({page})=>{
  await testAdapter(page);await page.addInitScript(()=>{Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{const c=document.createElement('canvas');c.width=640;c.height=480;const ctx=c.getContext('2d')!;let n=0;const draw=()=>{ctx.fillStyle=n++%2?'#bbc0cc':'#bbc1cc';ctx.fillRect(0,0,640,480);};draw();const timer=setInterval(draw,50);const stream=c.captureStream(20);(window as unknown as {syntheticTracks:MediaStreamTrack[]}).syntheticTracks=stream.getTracks();for(const t of stream.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};}return stream;}});});
  let requests=0;await page.route('**/api/v1/analysis',route=>{requests++;const body=route.request().postDataBuffer()!.toString('latin1');expect(body).toContain('camera');return route.fulfill({status:200,contentType:'application/x-ndjson',body:payload()});});
  await page.goto('/');await page.getByRole('button',{name:'웹캠으로 촬영하기'}).click();await expect(page.locator('#result-title')).toBeVisible({timeout:20000});expect(requests).toBe(1);expect(await page.evaluate(()=>(window as unknown as {syntheticTracks:MediaStreamTrack[]}).syntheticTracks.every(t=>t.readyState==='ended'))).toBe(true);await expect(page.locator('video')).toHaveCount(0);
});
test('late permission resolution after leaving closes newly opened stream',async({page})=>{await testAdapter(page);await page.addInitScript(()=>{Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:()=>new Promise(resolve=>setTimeout(()=>{const c=document.createElement('canvas');c.width=640;c.height=480;const s=c.captureStream();(window as unknown as {lateTracks:MediaStreamTrack[]}).lateTracks=s.getTracks();resolve(s);},600))});});await page.goto('/');await page.getByRole('button',{name:'웹캠으로 촬영하기'}).click();await page.getByRole('button',{name:'촬영 취소',exact:true}).click();await expect.poll(()=>page.evaluate(()=>(window as unknown as {lateTracks?:MediaStreamTrack[]}).lateTracks?.every(t=>t.readyState==='ended'))).toBe(true);await expect(page.locator('h1')).toContainText('얼굴 형태를 분류하고');});
test('input/server errors return to clean upload; retry can succeed',async({page})=>{await testAdapter(page);let count=0;await page.route('**/api/v1/analysis',route=>{count++;return route.fulfill(count===1?{status:422,contentType:'application/x-ndjson',body:JSON.stringify({type:'error',code:'FACE_NOT_FOUND',direction:'left_oblique',message:'얼굴을 찾지 못했습니다.'})}:{status:200,contentType:'application/x-ndjson',body:payload()});});await page.goto('/');await upload(page);await expect(page.getByRole('alert')).toContainText('화면 왼쪽을 향한 사진');await expect(page.locator('.upload-slot.empty')).toHaveCount(3);const im=await syntheticImage(page);for(const d of ['front','left_oblique','right_oblique'])await page.locator('#file-'+d).setInputFiles(im);await expect(page.locator('#result-title')).toBeVisible();expect(count).toBe(2);});
test('cancel discards late response and clears images',async({page})=>{await testAdapter(page);let finish:(()=>void)|undefined;await page.route('**/api/v1/analysis',async route=>{await new Promise<void>(r=>{finish=r;});await route.fulfill({status:200,contentType:'application/x-ndjson',body:payload()}).catch(()=>{});});await page.goto('/');await upload(page);await expect(page.locator('#progress-title')).toBeVisible();await page.getByRole('button',{name:'분석 취소',exact:true}).click();finish?.();await expect(page.locator('#upload-title')).toBeVisible();await expect(page.locator('.upload-slot.empty')).toHaveCount(3);await expect(page.locator('#result-title')).toHaveCount(0);});
test('no literature preserves judgments; no confirmed result requires reinput',async({page})=>{await testAdapter(page);let count=0;await page.route('**/api/v1/analysis',route=>{const data=fixture(0);if(count++)data.status='REINPUT_REQUIRED';return route.fulfill({status:200,contentType:'application/x-ndjson',body:payload(data)});});await page.goto('/');await upload(page);await expect(page.getByRole('heading',{name:'맞는 관상 풀이가 없습니다'})).toBeVisible();await expect(page.locator('.result-counts')).toContainText('판정');await page.getByRole('button',{name:'다시 시작',exact:true}).click();await upload(page);await expect(page.locator('#upload-title')).toBeVisible();await expect(page.getByRole('alert')).toContainText('확정 가능한 항목이 없습니다');});
test('oversized print preserves text and blocks one-page claim',async({page})=>{await testAdapter(page);await page.route('**/api/v1/analysis',route=>route.fulfill({status:200,contentType:'application/x-ndjson',body:payload(fixture(5,35))}));await page.goto('/');await upload(page);await expect(page.locator('#result-title')).toBeVisible();await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await expect(page.getByRole('button',{name:'A4 한 장 인쇄',exact:true})).toBeDisabled();await expect(page.getByRole('button',{name:'본문 보존 · 여러 장 인쇄'})).toBeVisible();await expect(page.locator('#print-sheet')).toContainText(fixture(5,35).passages[4].translation);});
test('real local CPU MediaPipe rejects a solid image; no external calls or backend request',async({page})=>{
  let posts=0;const external:string[]=[];page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith('http://127.0.0.1:5173/'))external.push(new URL(r.url()).origin);});await page.route('**/api/v1/analysis',route=>{posts++;return route.abort();});await page.goto('/');await page.getByRole('button',{name:'이미지 파일 업로드',exact:true}).click();await page.locator('#file-front').setInputFiles(await syntheticImage(page));await expect(page.locator('.upload-slot').first().locator('.slot-status')).toContainText('얼굴을 안내선',{timeout:30000});expect(posts).toBe(0);expect(external).toEqual([]);
});
test('actual corpus originals, source links and translations fit A4 without truncation',async({page})=>{
  const corpus=JSON.parse(readFileSync(path.resolve('../data/release/literature/reference_corpus_v1.json'),'utf8'));
  const real=corpus.passages.filter((p:{service_eligible:boolean})=>p.service_eligible)
    .sort((a:Passage,b:Passage)=>(b.original_text.length+b.translation.length)-(a.original_text.length+a.translation.length)).slice(0,5) as (Passage & {annotation_status:string})[];
  expect(real.length).toBe(5);
  const data=fixture(5);data.passages=data.passages.map((p,i)=>({...p,original_text:real[i].original_text,
    translation:real[i].translation,commentary:real[i].commentary,source:{...real[i].source,annotation_status:real[i].annotation_status}}));
  await testAdapter(page);await page.route('**/api/v1/analysis',r=>r.fulfill({status:200,contentType:'application/x-ndjson',body:payload(data)}));
  await page.goto('/');await upload(page);await expect(page.locator('#result-title')).toBeVisible();
  await expect(page.locator('.source a')).toHaveCount(5);await expect(page.locator('.passage-card').first()).toContainText('원문과 출처');
  await page.screenshot({path:path.join(artifacts,'actual-corpus-layout-desktop.png')});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(artifacts,'actual-corpus-layout-mobile.png')});
  await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await page.evaluate(()=>document.fonts.ready);
  await expect(page.getByRole('button',{name:'A4 한 장 인쇄',exact:true})).toBeEnabled();
  const pdf=path.join(artifacts,'actual-corpus-synthetic-scores-a4.pdf');await page.pdf({path:pdf,format:'A4',preferCSSPageSize:true,printBackground:true});
  const pdfjs=await import('pdfjs-dist/legacy/build/pdf.mjs');const doc=await pdfjs.getDocument({data:new Uint8Array(readFileSync(pdf))}).promise;
  expect(doc.numPages).toBe(1);const annotations=await(await doc.getPage(1)).getAnnotations();
  for(const p of real)expect(annotations.some(a=>a.url===(p.source.url||p.source.source_url))).toBe(true);
  if(real.some(p=>String(p.source.license||'').includes('CC BY-SA')))expect(annotations.some(a=>a.url==='https://creativecommons.org/licenses/by-sa/4.0/')).toBe(true);
  const text=(await (await doc.getPage(1)).getTextContent()).items.map(i=>'str' in i?i.str:'').join('').replace(/\s/g,'');
  for(const p of real){expect(text).toContain(p.original_text.replace(/\s/g,''));expect(text).toContain(p.translation.replace(/\s/g,''));}
  await doc.loadingTask.destroy();
});
test('actual training triplet passes local CPU acquisition and is cleared after mocked response',async({page})=>{
  // Photos travel only through process pipes/browser memory and loopback HTTP.
  // This tests real browser acquisition; the CNN/VGGT response is deliberately mocked.
  const python=process.platform==='win32'?path.resolve('../.venv/Scripts/python.exe'):'python3';
  const script=`import base64,json,pathlib,zipfile
r=pathlib.Path.cwd().parent
p=json.loads((r/'models/selection-20261014/protocol.json').read_text(encoding='utf-8'))
s='19062431'
assert s in p['groups']['train']
# Fixed TRAIN participant, cached three-view widths 192/195/187px; no label or result selection.
with zipfile.ZipFile(r/'data/raw/kface/zips'/f'{s}.zip') as z:
 images=[base64.b64encode(z.read(f'S001/L1/E01/{c}.jpg')).decode() for c in ('C7','C4','C10')]
with zipfile.ZipFile(r/'data/raw/kface/zips'/f"{sorted(p['groups']['train'])[0]}.zip") as z:
 images.append(base64.b64encode(z.read('S001/L1/E01/C7.jpg')).decode())
print(json.dumps(images))`;
  const encoded=JSON.parse(execFileSync(python,['-c',script],{encoding:'utf8',maxBuffer:30*1024*1024})) as string[];
  const external:string[]=[];page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith('http://127.0.0.1:5173/'))external.push(new URL(r.url()).origin);});
  let requests=0;await page.route('**/api/v1/analysis',r=>{requests++;return r.fulfill({status:200,contentType:'application/x-ndjson',body:payload(fixture(0))});});
  await page.goto('/');await page.getByRole('button',{name:'이미지 파일 업로드',exact:true}).click();
  // Regression: first TRAIN frontal is 157px wide in an 864px image, valid despite <25% framing.
  await page.locator('#file-front').setInputFiles({name:'frontal-framing-regression.jpg',mimeType:'image/jpeg',buffer:Buffer.from(encoded[3],'base64')});
  await expect(page.locator('.upload-slot.valid')).toHaveCount(1,{timeout:30000});
  await page.getByRole('button',{name:'정면 사진 지우기',exact:true}).click();
  for(const [i,direction] of ['front','left_oblique','right_oblique'].entries()){
    await page.locator('#file-'+direction).setInputFiles({name:direction+'.jpg',mimeType:'image/jpeg',buffer:Buffer.from(encoded[i],'base64')});
    if(i<2)await expect(page.locator('.upload-slot.valid')).toHaveCount(i+1,{timeout:30000});
  }
  encoded.fill('');await expect(page.locator('#result-title')).toBeVisible({timeout:30000});expect(requests).toBe(1);
  await expect(page.locator('img')).toHaveCount(0);expect(external).toEqual([]);
});
test('opt-in frontal photo with measured lines: default off, exact placement, print toggle, cleanup, dark mode',async({page})=>{
  await testAdapter(page);const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  // Synthetic 640x480 upload and synthetic measured paths in the same pixel frame.
  const data=fixture(2);data.face_overlay={width:640,height:480,features:[
    {item_id:'nose_tip',label:'코끝 너비',source:'landmark',paths:[[[280,300],[360,300]]],value:.125,unit:'ratio'},
    {item_id:'nose_bridge',label:'코뿌리 높이',source:'landmark',paths:[[[320,180],[320,290]]]},
    {item_id:'region_ear',label:'귀 길이',source:'segmentation',paths:[[[560,160],[565,300]]]}]};
  await page.route('**/api/v1/analysis',route=>route.fulfill({status:200,contentType:'application/x-ndjson',body:payload(data)}));
  const urls=()=>page.evaluate(()=>(window as unknown as {testURLs:Set<string>}).testURLs.size);
  await page.goto('/');await upload(page);await expect(page.locator('#result-title')).toBeVisible();
  const toggle=page.locator('.photo-panel').getByLabel('사진과 측정선 표시');
  await expect(toggle).not.toBeChecked();await expect(page.locator('img')).toHaveCount(0);expect(await urls()).toBe(0);
  await toggle.check();const photo=page.locator('.photo-panel .photo-frame img');await expect(photo).toHaveCount(1);
  await expect.poll(()=>photo.evaluate(i=>(i as HTMLImageElement).naturalWidth)).toBe(640);expect(await urls()).toBe(1);await expect(page.locator('#print-sheet img')).toHaveCount(1);
  await expect(page.locator('.photo-panel .line')).toHaveCount(3);
  await expect(page.locator('.photo-panel .overlay-legend')).toContainText('판정 보류');
  // The synthetic (320,180)-(320,290) line must land on the same pixels of the rendered image.
  const geometry=await page.evaluate(()=>{const img=document.querySelector('.photo-panel .photo-frame img')!.getBoundingClientRect(),svg=document.querySelector('.photo-panel .photo-frame svg')!.getBoundingClientRect(),line=document.querySelector('.photo-panel [data-item="nose_bridge"] .line')!.getBoundingClientRect();return {img:[img.x,img.y,img.width,img.height],svg:[svg.x,svg.y,svg.width,svg.height],line:[line.x+line.width/2,line.y,line.y+line.height]};});
  geometry.svg.forEach((v,i)=>expect(Math.abs(v-geometry.img[i])).toBeLessThan(1));
  const [x,y,w,h]=geometry.img;expect(Math.abs(geometry.line[0]-(x+w*320/640))).toBeLessThan(1.5);expect(Math.abs(geometry.line[1]-(y+h*180/480))).toBeLessThan(1.5);expect(Math.abs(geometry.line[2]-(y+h*290/480))).toBeLessThan(1.5);
  await page.locator('.photo-panel .overlay-legend').getByLabel(/귀 길이/).uncheck();await expect(page.locator('.photo-panel .line')).toHaveCount(2);
  await page.screenshot({path:path.join(artifacts,'photo-overlay-desktop.png'),fullPage:true});
  await page.getByRole('button',{name:'어두운 테마로 전환'}).click();await page.screenshot({path:path.join(artifacts,'photo-overlay-dark.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:path.join(artifacts,'photo-overlay-mobile-dark.png'),fullPage:true});await page.setViewportSize({width:1280,height:1000});
  await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await page.evaluate(()=>document.fonts.ready);
  await expect(page.locator('#print-sheet .print-photo img')).toHaveCount(1);await expect(page.locator('#print-sheet .line')).toHaveCount(2);
  await expect(page.locator('#print-sheet')).toHaveAttribute('data-fits','true');await page.screenshot({path:path.join(artifacts,'photo-overlay-a4-preview.png'),fullPage:true});
  const pdf=path.join(artifacts,'photo-overlay-a4.pdf');await page.pdf({path:pdf,format:'A4',preferCSSPageSize:true,printBackground:true});
  const pdfjs=await import('pdfjs-dist/legacy/build/pdf.mjs');const doc=await pdfjs.getDocument({data:new Uint8Array(readFileSync(pdf))}).promise;expect(doc.numPages).toBe(1);await doc.loadingTask.destroy();
  await page.locator('.preview-controls').getByLabel('사진과 측정선 표시').uncheck();await expect(page.locator('#print-sheet img')).toHaveCount(0);expect(await urls()).toBe(0);
  await page.locator('.preview-controls').getByLabel('사진과 측정선 표시').check();await expect(page.locator('#print-sheet img')).toHaveCount(1);
  await page.getByRole('button',{name:'결과로 돌아가기',exact:true}).click();await page.getByRole('button',{name:'다시 시작',exact:true}).click();
  await expect(page.locator('img')).toHaveCount(0);expect(await urls()).toBe(0);
  expect(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length}))).toEqual({local:0,session:0});
  expect(errors).toEqual([]);
});
test('19 backend overlay features: 4 default lines, all reachable, mobile/dark/print without clutter or overflow',async({page})=>{
  await testAdapter(page);const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  const data=fixture(5,2);data.face_overlay=overlay19();
  await page.route('**/api/v1/analysis',route=>route.fulfill({status:200,contentType:'application/x-ndjson',body:payload(data)}));
  await page.goto('/');await expect(page.locator('.privacy')).toContainText('처음으로 돌아가거나 창을 닫으면 지워집니다');
  await upload(page);await expect(page.locator('#result-title')).toBeVisible();
  await page.locator('.photo-panel').getByLabel('사진과 측정선 표시').check();
  const panel=page.locator('.photo-panel');await expect(panel.locator('.photo-frame img')).toHaveCount(1);
  await expect(panel.locator('svg .overlay-feature')).toHaveCount(4);
  expect(await panel.locator('svg .overlay-feature').evaluateAll(g=>g.map(e=>e.getAttribute('data-item')))).toEqual(['eye_tilt','brow_length','nose_width','mouth_width']);
  await expect(panel.locator('.overlay-legend').first()).toContainText('3.42°');await expect(panel.locator('.legend-more summary')).toHaveText('다른 측정선 15개');
  await page.screenshot({path:path.join(artifacts,'overlay19-default-desktop.png'),fullPage:true});
  await panel.locator('.legend-more summary').click();await panel.locator('.legend-more').getByLabel(/턱 너비/).click();await expect(panel.locator('.overlay-legend').first().getByLabel(/턱 너비/)).toBeChecked();
  await expect(panel.locator('svg .overlay-feature')).toHaveCount(5);await expect(panel.locator('.legend-more summary')).toHaveText('다른 측정선 14개');
  await panel.getByRole('button',{name:'전체 19개'}).click();await expect(panel.locator('svg .overlay-feature')).toHaveCount(19);
  await page.screenshot({path:path.join(artifacts,'overlay19-all-desktop.png'),fullPage:true});
  await panel.getByRole('button',{name:'기본 4개'}).click();await expect(panel.locator('svg .overlay-feature')).toHaveCount(4);
  await page.getByRole('button',{name:'어두운 테마로 전환'}).click();await page.setViewportSize({width:390,height:844});
  await panel.locator('.legend-more summary').click();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:path.join(artifacts,'overlay19-mobile-dark.png'),fullPage:true});
  await page.setViewportSize({width:1280,height:1000});
  await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await page.evaluate(()=>document.fonts.ready);
  await expect(page.locator('#print-sheet .overlay-feature')).toHaveCount(4);await expect(page.locator('#print-sheet .overlay-legend li')).toHaveCount(4);
  await expect(page.locator('#print-sheet .legend-more')).toHaveCount(0);await expect(page.locator('#print-sheet')).toHaveAttribute('data-fits','true');
  await page.screenshot({path:path.join(artifacts,'overlay19-a4-default.png'),fullPage:true});
  const pdf=path.join(artifacts,'overlay19-a4-default.pdf');await page.pdf({path:pdf,format:'A4',preferCSSPageSize:true,printBackground:true});
  const pdfjs=await import('pdfjs-dist/legacy/build/pdf.mjs');let doc=await pdfjs.getDocument({data:new Uint8Array(readFileSync(pdf))}).promise;expect(doc.numPages).toBe(1);
  const text=(await (await doc.getPage(1)).getTextContent()).items.map(i=>'str' in i?i.str:'').join('').replace(/\s/g,'');for(const p of data.passages){expect(text).toContain(p.original_text.replace(/\s/g,''));expect(text).toContain(p.translation.replace(/\s/g,''));}await doc.loadingTask.destroy();
  // All 19 on paper: either still one page or the explicit full-text multi-page fallback, never truncation.
  await page.getByRole('button',{name:'결과로 돌아가기',exact:true}).click();await panel.getByRole('button',{name:'전체 19개'}).click();
  await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await expect(page.locator('#print-sheet .overlay-legend li')).toHaveCount(19);
  const fits=await page.locator('#print-sheet').getAttribute('data-fits');
  if(fits==='false')await expect(page.getByRole('button',{name:'본문 보존 · 여러 장 인쇄'})).toBeVisible();
  const pdfAll=path.join(artifacts,'overlay19-a4-all.pdf');await page.pdf({path:pdfAll,format:'A4',preferCSSPageSize:true,printBackground:true});
  doc=await pdfjs.getDocument({data:new Uint8Array(readFileSync(pdfAll))}).promise;if(fits==='true')expect(doc.numPages).toBe(1);
  let all='';for(let n=1;n<=doc.numPages;n++)all+=(await (await doc.getPage(n)).getTextContent()).items.map(i=>'str' in i?i.str:'').join('');
  all=all.replace(/\s/g,'');for(const p of data.passages){expect(all).toContain(p.original_text.replace(/\s/g,''));expect(all).toContain(p.translation.replace(/\s/g,''));}
  console.log(`overlay19 all-lines A4: fits=${fits} pages=${doc.numPages}`);await doc.loadingTask.destroy();
  await page.screenshot({path:path.join(artifacts,'overlay19-a4-all.png'),fullPage:true});
  await page.getByRole('button',{name:'결과로 돌아가기',exact:true}).click();await page.getByRole('button',{name:'다시 시작',exact:true}).click();
  await expect(page.locator('img')).toHaveCount(0);expect(await page.evaluate(()=>(window as unknown as {testURLs:Set<string>}).testURLs.size)).toBe(0);
  expect(errors).toEqual([]);
});
