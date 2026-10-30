import {describe,expect,it} from 'vitest';
import {ACQUISITION,CaptureState,eulerXYZ,guideFit,quality} from '../src/capture';
import {readNDJSON,validateResult} from '../src/api';
import {DIRECTIONS,type Conditions} from '../src/types';
const good:Conditions={face:true,direction:true,size:true,brightness:true,expression:true,message:'좋음'};
describe('capture state (synthetic frames; no webcam claim)',()=>{
  it('requires 800ms AND 8 fresh consecutive frames, captures each direction once',()=>{
    const s=new CaptureState(); let time=0;
    for(const direction of DIRECTIONS){const token=s.token();expect(token.direction).toBe(direction);for(let j=0;j<8;j++)expect(s.update(good,time+j*100,time+j*100, time+j*100,token).capture).toBe(false);expect(s.update(good,time+800,time+800,time+800,token).capture).toBe(true);expect(s.update(good,time+900,time+900,time+900,token).ignored).toBe(true);expect(s.commit(token)).toBe(true);expect(s.commit(token)).toBe(false);time+=1000;}
    expect(s.submitOnce()).toBe(true);expect(s.submitOnce()).toBe(false);
  });
  it('resets on condition failure, long gaps, duplicate frame and stale evaluation',()=>{
    const s=new CaptureState(),t=s.token();s.update(good,0,0,0,t);s.update(good,100,100,1,t);
    expect(s.update({...good,brightness:false},200,200,2,t).progress).toBe(0);
    expect(s.update(good,300,300,3,t).progress).toBe(0);
    expect(s.update(good,1000,1000,4,t).progress).toBe(0);
    expect(s.update(good,1100,1100,4,t).ignored).toBe(true);
    expect(s.update(good,1200,1700,5,t).ignored).toBe(true);
    expect(s.update(good,1800,1800,6,t).progress).toBe(0);
  });
  it('ignores prior generation after reset or direction commit',()=>{const s=new CaptureState(),t=s.token();s.reset();expect(s.update(good,0,0,0,t).ignored).toBe(true);expect(s.commit(t)).toBe(false);expect(s.submitOnce()).toBe(false);});
});
describe('shared acquisition defaults and Euler convention',()=>{
  it('uses unmirrored column-major rotation with xyz Euler signs',()=>{
    for(const [pitch,yaw,roll] of [[-5,-30,0],[-5,30,8],[10,0,-10]]){
      const [x,y,z]=[pitch,yaw,roll].map(v=>v*Math.PI/180),cx=Math.cos(x),sx=Math.sin(x),cy=Math.cos(y),sy=Math.sin(y),cz=Math.cos(z),sz=Math.sin(z);
      const m=[cz*cy,sz*cy,-sy,0,cz*sy*sx-sz*cx,sz*sy*sx+cz*cx,cy*sx,0,cz*sy*cx+sz*sx,sz*sy*cx-cz*sx,cy*cx,0,0,0,0,1];
      const p=eulerXYZ(m);expect(p.pitch).toBeCloseTo(pitch);expect(p.yaw).toBeCloseTo(yaw);expect(p.roll).toBeCloseTo(roll);
    }
  });
  it('matches front/left/right thresholds and expression/brightness/size checks',()=>{
    const q=(yaw:number,d:typeof DIRECTIONS[number])=>quality({yaw,pitch:-5,roll:0},d,1,200,640,128,0,0);
    expect(q(12,'front').direction).toBe(true);expect(q(13,'front').direction).toBe(false);
    expect(q(-20,'left_oblique').direction).toBe(true);expect(q(30,'left_oblique').direction).toBe(false);expect(q(60,'right_oblique').direction).toBe(true);expect(q(61,'right_oblique').direction).toBe(false);
    const bad=quality({yaw:0,pitch:12,roll:13},'front',2,100,640,20,.31,.51);
    expect([bad.face,bad.direction,bad.size,bad.brightness,bad.expression]).toEqual([false,false,false,false,false]);expect(ACQUISITION.goodFrames).toBe(8);
  });
});
const result={status:'COMPLETE',items:[],passages:[]};
function response(text:string,status=200){const bytes=new TextEncoder().encode(text);return new Response(new ReadableStream({start(c){for(const b of bytes)c.enqueue(new Uint8Array([b]));c.close();}}),{status,headers:{'content-type':'application/x-ndjson'}});}
describe('stream contract',()=>{
  it('decodes Korean across byte boundaries and final line without newline',async()=>{const stages:string[]=[];expect(await readNDJSON(response(JSON.stringify({type:'progress',stage:'얼굴 정렬'})+'\n'+JSON.stringify({type:'result',data:result})),s=>stages.push(s))).toEqual(result);expect(stages).toEqual(['얼굴 정렬']);});
  it('preserves prestream HTTP NDJSON direction errors',async()=>{await expect(readNDJSON(response(JSON.stringify({type:'error',code:'FACE_NOT_FOUND',direction:'left_oblique',message:'얼굴 없음'}),422),()=>{})).rejects.toMatchObject({direction:'left_oblique',code:'FACE_NOT_FOUND',message:'얼굴 없음'});});
  it('rejects truncated, malformed and duplicate terminal streams',async()=>{for(const text of ['{"type":"progress","stage":"형태 분류"}\n','broken\n',JSON.stringify({type:'result',data:result})+'\n'+JSON.stringify({type:'result',data:result})])await expect(readNDJSON(response(text),()=>{})).rejects.toThrow();});
  it('rejects unknown state, too many passages and incomplete sources',()=>{expect(()=>validateResult({...result,status:'unknown'})).toThrow();expect(()=>validateResult({...result,passages:Array(6).fill({})})).toThrow();expect(()=>validateResult({...result,passages:[{original_text:'原文',translation:'직역',commentary:'해설',source:{},evidence_item_ids:[]}]})).toThrow();});
  it('aborts outstanding stream instead of displaying completion',async()=>{const abort=new AbortController();const p=readNDJSON(new Response(new ReadableStream(),{headers:{'content-type':'application/x-ndjson'}}),()=>{},abort.signal);abort.abort();await expect(p).rejects.toThrow();});
});
it('uses the framing ratio only for camera guidance; upload still needs 150 inspected face pixels',()=>{
  const p={yaw:0,pitch:0,roll:0};
  expect(quality(p,'front',1,157,864,128,0,0,'camera').size).toBe(false);
  expect(quality(p,'front',1,157,864,128,0,0,'upload').size).toBe(true);
  expect(quality(p,'front',1,149,864,128,0,0,'upload').size).toBe(false);
});
it('camera framing requires the face inside the guide ellipse; uploads ignore it',()=>{
  const p={yaw:0,pitch:0,roll:0},ring=(cx:number,cy:number)=>Array.from({length:36},(_,i)=>({x:cx+.15*Math.cos(i/36*2*Math.PI),y:cy+.28*Math.sin(i/36*2*Math.PI)}));
  expect(guideFit(ring(.5,.5))).toBe(1);
  expect(guideFit(ring(.2,.5))).toBeLessThan(ACQUISITION.minInGuide);
  expect(guideFit([])).toBe(0);
  expect(quality(p,'front',1,300,864,128,0,0,'camera',guideFit(ring(.5,.5))).size).toBe(true);
  const off=quality(p,'front',1,300,864,128,0,0,'camera',guideFit(ring(.2,.5)));
  expect(off.size).toBe(false); expect(off.message).toContain('안내선 안');
  expect(quality(p,'front',1,300,864,128,0,0,'upload',0).size).toBe(true);
});
