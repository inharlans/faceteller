// Opt-in actual local service integration. No route interception or fixture injection.
import {test,expect} from '@playwright/test';
import {execFileSync} from 'node:child_process';
import {mkdirSync,writeFileSync,readFileSync} from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {validateResult} from '../src/api';
import {DEFAULT_OVERLAY_FEATURES} from '../src/PhotoOverlay';

test('production upload uses actual models and DB, clears photos and preserves the result in PDF',async({page,request,baseURL})=>{
  const root=path.resolve('..'),origin=new URL(baseURL!).origin;
  const hash=(name:string)=>createHash('sha256').update(readFileSync(path.join(root,name))).digest('hex');
  const expected={deployment_sha256:hash('models/release/deployment.json'),depth_config_sha256:hash('data/release/depth_config.json'),
    registry_sha256:hash('data/release/training_registry.json'),corpus_sha256:JSON.parse(readFileSync(path.join(root,'data/release/literature_ready.json'),'utf8')).corpus_sha256};
  const health=await request.get('/api/v1/health');expect(health.ok()).toBe(true);
  const state=await health.json();expect(state.ready).toBe(true);expect(state.corpus_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(state.repository).toBe('postgresql_readonly');
  for(const [key,value] of Object.entries(expected))expect(state[key]).toBe(value);
  const python=path.join(root,'.venv','Scripts','python.exe');
  const script=`import base64,json,pathlib,zipfile
r=pathlib.Path.cwd().parent
p=json.loads((r/'models/selection-20261014/protocol.json').read_text(encoding='utf-8'))
s='19062431'
assert s in p['groups']['train']
with zipfile.ZipFile(r/'data/raw/kface/zips'/f'{s}.zip') as z:
 print(json.dumps([base64.b64encode(z.read(f'S001/L1/E01/{c}.jpg')).decode() for c in ('C7','C4','C10')]))`;
  const images=JSON.parse(execFileSync(python,['-c',script],{encoding:'utf8',maxBuffer:30*1024*1024})) as string[];
  const external:string[]=[],errors:string[]=[];let posts=0;
  // Observe the actual stream before the app's successful-result cleanup aborts
  // its request. CDP response.text() can lose that body after cleanup on Windows.
  // The original response is forwarded unchanged; no route or model is mocked.
  await page.addInitScript(()=>{
    const owner=window as unknown as {observedAnalysisEvents:Record<string,any>[]};
    owner.observedAnalysisEvents=[];const originalFetch=window.fetch.bind(window);
    window.fetch=async(input,init)=>{
      const response=await originalFetch(input,init);
      if(init?.body instanceof FormData){
        const reader=response.clone().body!.getReader();
        void (async()=>{
          const decoder=new TextDecoder();let buffer='';
          try{for(;;){
            const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
            let end:number;
            while((end=buffer.indexOf('\n'))>=0){
              const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(!line.trim())continue;
              const event=JSON.parse(line);owner.observedAnalysisEvents.push(event);
              if(event.type==='result'||event.type==='error'){void reader.cancel().catch(()=>{});return;}
            }
          }}finally{reader.releaseLock();}
        })().catch(()=>{});
      }
      return response;
    };
  });
  page.on('request',r=>{if(r.method()==='POST'&&new URL(r.url()).pathname==='/api/v1/analysis')posts++;
    if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==origin)external.push(new URL(r.url()).origin);});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');await expect(page.locator('h1')).toContainText('얼굴 형태를 분류하고');
  await page.getByRole('button',{name:'이미지 파일 업로드',exact:true}).click();
  const responsePromise=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/analysis',{timeout:60000});
  for(const [i,d] of ['front','left_oblique','right_oblique'].entries()){
    await page.locator('#file-'+d).setInputFiles({name:d+'.jpg',mimeType:'image/jpeg',buffer:Buffer.from(images[i],'base64')});
    if(i<2)await expect(page.locator('.upload-slot.valid')).toHaveCount(i+1,{timeout:30000});
  }
  images.fill('');const response=await responsePromise;expect(response.ok()).toBe(true);
  await page.waitForFunction(()=>(window as unknown as {observedAnalysisEvents:Record<string,any>[]}).observedAnalysisEvents.some(e=>e.type==='result'||e.type==='error'));
  const records=await page.evaluate(()=>(window as unknown as {observedAnalysisEvents:Record<string,any>[]}).observedAnalysisEvents);
  expect(records.filter(r=>r.type==='result')).toHaveLength(1);expect(records.some(r=>r.type==='error')).toBe(false);
  const raw=records.find(r=>r.type==='result')!.data,result=validateResult(raw);
  for(const [key,value] of Object.entries(expected))expect(raw.runtime?.[key]).toBe(value);
  expect(result.items.some(i=>i.status==='CONFIRMED')).toBe(true);expect(result.status).not.toBe('REINPUT_REQUIRED');
  await expect(page.locator('#result-title')).toBeVisible();expect(posts).toBe(1);
  await expect(page.locator('.passage-card')).toHaveCount(result.passages.length);
  if(state.reading_catalog_kind==='gwansang_grouped'){
    expect(raw.runtime.gwansang_catalog_sha256).toBe(state.gwansang_catalog_sha256);
    expect(raw.named_types).toEqual([]);
    expect(new Set(raw.passages.map((p:any)=>p.source.shape_id)).size).toBe(raw.passages.length);
    await expect(page.locator('.type-card')).toHaveCount(0);
    await expect(page.getByRole('heading',{name:'관상 풀이',exact:true,level:2})).toHaveCount(1);
    for(const p of result.passages)await expect(page.locator('.passage-card .card-title').filter({hasText:p.reading!.title})).toHaveCount(1);
  }
  await expect(page.locator('img')).toHaveCount(0);
  expect(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length}))).toEqual({local:0,session:0});
  expect(external).toEqual([]);expect(errors).toEqual([]);
  const folder=path.join(os.tmpdir(),'faceteller-production-ui');mkdirSync(folder,{recursive:true});
  // Only the photo-free results screen is captured.
  await page.screenshot({path:path.join(folder,'actual-result-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:path.join(folder,'actual-result-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();await page.evaluate(()=>document.fonts.ready);
  const pdf=path.join(folder,'actual-sample-result.pdf');await page.pdf({path:pdf,format:'A4',preferCSSPageSize:true,printBackground:true});
  const pdfjs=await import('pdfjs-dist/legacy/build/pdf.mjs'),doc=await pdfjs.getDocument({data:new Uint8Array(readFileSync(pdf))}).promise;
  const all=[];for(let n=1;n<=doc.numPages;n++)all.push((await(await doc.getPage(n)).getTextContent()).items.map(i=>'str'in i?i.str:'').join(''));
  const text=all.join('').replace(/\s/g,'');
  for(const p of result.passages){expect(text).toContain(p.original_text.replace(/\s/g,''));expect(text).toContain(p.translation.replace(/\s/g,''));}
  const onePage=await page.getByRole('button',{name:'A4 한 장 인쇄',exact:true}).isEnabled();
  if(onePage)expect(doc.numPages).toBe(1);
  // Opt-in only (FACETELLER_PHOTO_CHECK=1): verify the frontal photo/overlay geometry with scalar checks.
  // No screenshot or PDF is taken while the real photo is shown, and the photo is turned off again.
  let photoCheck:Record<string,unknown>|undefined;
  if(process.env.FACETELLER_PHOTO_CHECK==='1'){
    await page.setViewportSize({width:1280,height:1000});
    await page.getByRole('button',{name:'결과로 돌아가기',exact:true}).click();
    const toggle=page.locator('.photo-panel').getByLabel('사진과 측정선 표시');await expect(toggle).not.toBeChecked();
    await toggle.check();const photo=page.locator('.photo-panel .photo-frame img');await expect(photo).toHaveCount(1);
    await expect.poll(()=>photo.evaluate(i=>(i as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
    const overlay=raw.face_overlay as {width:number,height:number,features:{item_id:string,paths:number[][][]}[]}|null|undefined;
    photoCheck=await page.evaluate(()=>{const img=document.querySelector('.photo-panel .photo-frame img') as HTMLImageElement,svg=document.querySelector('.photo-panel .photo-frame svg'),a=img.getBoundingClientRect(),b=svg?.getBoundingClientRect();
      return {natural:[img.naturalWidth,img.naturalHeight],viewBox:svg?.getAttribute('viewBox')??null,lines:document.querySelectorAll('.photo-panel .line').length,
        box_delta:b?Math.max(Math.abs(a.x-b.x),Math.abs(a.y-b.y),Math.abs(a.width-b.width),Math.abs(a.height-b.height)):null};});
    if(overlay?.features?.length){
      expect(photoCheck.viewBox).toBe(`0 0 ${overlay.width} ${overlay.height}`);
      const [w,h]=photoCheck.natural as number[];expect(Math.abs((w/h)/(overlay.width/overlay.height)-1)).toBeLessThan(.01);
      const initial=overlay.features.filter(f=>DEFAULT_OVERLAY_FEATURES.includes(f.item_id));
      expect(photoCheck.lines).toBe(initial.reduce((n,f)=>n+f.paths.length,0));expect(photoCheck.box_delta as number).toBeLessThan(1);
      await page.locator('.photo-panel').getByRole('button',{name:`전체 ${overlay.features.length}개`,exact:true}).click();
      await expect(page.locator('.photo-panel .line')).toHaveCount(overlay.features.reduce((n,f)=>n+f.paths.length,0));
      await page.locator('.photo-panel').getByRole('button',{name:'기본 4개',exact:true}).click();
      await page.getByRole('button',{name:'결과지 미리보기',exact:true}).click();
      await expect(page.locator('#print-sheet .photo-frame img')).toBeVisible();
      await expect(page.locator('#print-sheet .line')).toHaveCount(initial.reduce((n,f)=>n+f.paths.length,0));
      await page.getByRole('button',{name:'결과로 돌아가기',exact:true}).click();
      photoCheck.print_photo_verified=true;
    }else expect(photoCheck.lines).toBe(0);
    await toggle.uncheck();await expect(page.locator('img')).toHaveCount(0);
    expect(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length}))).toEqual({local:0,session:0});
    photoCheck={pass:true,overlay_features:overlay?.features?.length??0,...photoCheck};
  }
  writeFileSync(path.join(folder,'verification.json'),JSON.stringify({pass:true,actual_models:true,mocked:false,artifact_identity_verified:true,...expected,
    pdf_passage_text_verified_count:result.passages.length,source_origin:origin,
    participants:1,split:'train',scope:'fixed training participant UI integration only; not independent model accuracy',
    corpus_sha256:state.corpus_sha256,passages:result.passages.length,confirmed:result.items.filter(i=>i.status==='CONFIRMED').length,
    held:result.items.filter(i=>i.status==='HELD').length,pdf_pages:doc.numPages,one_page_button:onePage,
    photos_saved:false,external_requests:external.length,page_errors:errors,received_to_result_seconds:raw.runtime?.timings?.received_to_result_seconds??null,
    ...(state.reading_catalog_kind==='gwansang_grouped'?{unified_readings_verified:true,gwansang_catalog_sha256:state.gwansang_catalog_sha256}:{}),
    ...(photoCheck?{photo_overlay_check:photoCheck}:{})},null,2));
  await doc.loadingTask.destroy();
});
