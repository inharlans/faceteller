// Actual local detector/models with an in-memory recorded-image video source.
// This verifies the acquisition pipeline, not a physical camera or natural motion.
import {test,expect} from '@playwright/test';
import {execFileSync} from 'node:child_process';
import {mkdirSync,writeFileSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';

test('recorded face frames pass real acquisition and camera/upload produce equivalent results',async({page,baseURL})=>{
  const root=path.resolve('..'), origin=new URL(baseURL!).origin;
  const raw=JSON.parse(execFileSync(path.join(root,'.venv/Scripts/python.exe'),['-B','-X','utf8','-c',`import base64,json,pathlib,zipfile
r=pathlib.Path.cwd().parent
p=json.loads((r/'models/selection-20261014/protocol.json').read_text(encoding='utf-8'))
s='19062431'; assert s in p['groups']['train']
with zipfile.ZipFile(r/'data/raw/kface/zips'/f'{s}.zip') as z:
 print(json.dumps([base64.b64encode(z.read(f'S001/L1/E01/{c}.jpg')).decode() for c in ('C7','C4','C10')]))`],{encoding:'utf8',maxBuffer:30*1024*1024})) as string[];
  const errors:string[]=[],external:string[]=[];let posts=0;
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',r=>{if(r.method()==='POST'&&new URL(r.url()).pathname==='/api/v1/analysis')posts++;
    if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==origin)external.push(new URL(r.url()).origin);});
  await page.addInitScript(async(images:string[])=>{
    const originalFetch=window.fetch.bind(window);
    (window as unknown as {recordedResults:unknown[]}).recordedResults=[];
    window.fetch=async(input,init)=>{
      if(init?.body instanceof FormData && init.body.get('input_mode')==='camera'){
        (window as unknown as {recordedInputs:Record<string,Blob>}).recordedInputs=Object.fromEntries(
          ['front','left_oblique','right_oblique'].map(d=>[d,(init.body as FormData).get(d) as Blob]));
      }
      const response=await originalFetch(input,init);
      if(init?.body instanceof FormData){
        const reader=response.clone().body!.getReader();
        void (async()=>{
          const decoder=new TextDecoder();let buffer='';
          try{
            for(;;){
              const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
              let end:number;
              while((end=buffer.indexOf('\n'))>=0){
                const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(!line.trim())continue;
                const event=JSON.parse(line);
                if(event.type==='result'||event.type==='error'){
                  (window as unknown as {recordedResults:unknown[]}).recordedResults.push(event);
                  void reader.cancel().catch(()=>{});return;
                }
              }
            }
          }finally{reader.releaseLock();}
        })().catch(()=>{});
      }
      return response;
    };
    const decoded=await Promise.all(images.map(data=>new Promise<HTMLImageElement>((resolve,reject)=>{
      const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src='data:image/jpeg;base64,'+data;
    })));
    images.fill('');
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{
      const canvas=document.createElement('canvas');canvas.width=480;canvas.height=480;
      const ctx=canvas.getContext('2d')!;
      const draw=()=>{
        const direction=document.querySelector('[data-capture-direction]')?.getAttribute('data-capture-direction');
        const index=direction==='left_oblique'?1:direction==='right_oblique'?2:0;
        const im=decoded[index];
        // Centre window simulates bringing a face inside the camera guide;
        // no synthetic pixels, face detection replacement or model mocks.
        ctx.drawImage(im,(im.width-480)/2,(im.height-480)/2,480,480,0,0,480,480);
      };
      draw();const stream=canvas.captureStream(20),timer=setInterval(draw,50);
      (window as unknown as {recordedTracks:MediaStreamTrack[]}).recordedTracks=stream.getTracks();
      const stop=stream.getTracks()[0].stop.bind(stream.getTracks()[0]);
      stream.getTracks()[0].stop=()=>{clearInterval(timer);canvas.width=0;canvas.height=0;decoded.forEach(im=>im.src='');stop();};
      return stream;
    }});
  },raw);
  raw.fill('');await page.goto('/');await expect(page.locator('h1')).toContainText('얼굴 형태를 분류하고');
  const cameraResponse=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/analysis',{timeout:90000});
  await page.getByRole('button',{name:'웹캠으로 촬영하기',exact:true}).click();
  const response=await cameraResponse;expect(response.ok()).toBe(true);
  // CDP omits file bodies from request.postDataBuffer. Retain the exact camera
  // Blobs only in test memory while forwarding fetch unchanged to the server.
  const captured=await page.evaluate(async()=>{
    const owner=window as unknown as {recordedInputs?:Record<string,Blob>};
    const result=await Promise.all(Object.entries(owner.recordedInputs!).map(async([direction,blob])=>({
      direction,mime:blob.type,data:btoa(Array.from(new Uint8Array(await blob.arrayBuffer()),n=>String.fromCharCode(n)).join(''))})));
    delete owner.recordedInputs;return result;
  });
  expect(captured).toHaveLength(3);
  await page.waitForFunction(()=>(window as unknown as {recordedResults:unknown[]}).recordedResults.length===1);
  const camera=await page.evaluate(()=>(window as unknown as {recordedResults:{type:string;data:Record<string,any>}[]}).recordedResults[0].data);
  expect(camera).toBeTruthy();expect(camera.items.some((i:{status:string})=>i.status==='CONFIRMED')).toBe(true);
  await expect(page.locator('#result-title')).toBeVisible();expect(posts).toBe(1);
  expect(await page.evaluate(()=>(window as unknown as {recordedTracks:MediaStreamTrack[]}).recordedTracks.every(t=>t.readyState==='ended'))).toBe(true);
  await expect(page.locator('video')).toHaveCount(0);
  await page.getByRole('button',{name:'다시 시작',exact:true}).click();
  await page.getByRole('button',{name:'이미지 파일 업로드',exact:true}).click();
  const uploadResponse=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/analysis',{timeout:60000});
  for(const [index,direction] of ['front','left_oblique','right_oblique'].entries()){
    const file=captured.find(f=>f.direction===direction)!;
    await page.locator('#file-'+direction).setInputFiles({name:direction+'.jpg',mimeType:file.mime,buffer:Buffer.from(file.data,'base64')});
    if(index<2)await expect(page.locator('.upload-slot.valid')).toHaveCount(index+1,{timeout:30000});
    file.data='';
  }
  expect((await uploadResponse).ok()).toBe(true);
  await page.waitForFunction(()=>(window as unknown as {recordedResults:unknown[]}).recordedResults.length===2);
  const upload=await page.evaluate(()=>(window as unknown as {recordedResults:{type:string;data:Record<string,any>}[]}).recordedResults[1].data);
  expect(upload).toBeTruthy();expect(posts).toBe(2);await expect(page.locator('#result-title')).toBeVisible();
  let maxScoreDifference=0;
  for(const [index,item] of camera.items.entries()){
    const other=upload.items[index];expect(other.item_id).toBe(item.item_id);expect(other.status).toBe(item.status);
    expect(other.label_id).toBe(item.label_id);expect(other.hold_reasons).toEqual(item.hold_reasons);
    for(const key of ['confidence','measurement_score'])if(typeof item[key]==='number'){
      const delta=Math.abs(item[key]-other[key]);maxScoreDifference=Math.max(maxScoreDifference,delta);expect(delta).toBeLessThan(1e-4);
    }
  }
  expect(upload.passages.map((p:{passage_id:string})=>p.passage_id)).toEqual(camera.passages.map((p:{passage_id:string})=>p.passage_id));
  expect(errors).toEqual([]);expect(external).toEqual([]);await expect(page.locator('img')).toHaveCount(0);
  const folder=path.join(os.tmpdir(),'faceteller-recorded-camera');mkdirSync(folder,{recursive:true});
  writeFileSync(path.join(folder,'verification.json'),JSON.stringify({pass:true,physical_camera:false,
    recorded_source:true,detector_mocked:false,actual_models:true,photos_saved:false,participants:1,split:'train',
    capture_posts:1,total_posts:posts,tracks_stopped:true,equivalent_item_count:camera.items.length,
    max_score_difference:maxScoreDifference,external_requests:external.length,page_errors:errors,
    deployment_sha256:camera.runtime.deployment_sha256,quality_version:camera.runtime.quality_version},null,2));
});
