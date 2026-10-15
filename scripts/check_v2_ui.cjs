// Browser check of /v2 with a public FRLL photo (never a K-FACE photo, so screenshots are shareable).
// Usage: node scripts/check_v2_ui.cjs [baseUrl]   (Windows node, Playwright chromium)
const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=path.join(root,'outputs','v2-ui-check');fs.mkdirSync(out,{recursive:true});
 const base=process.argv[2]||'http://127.0.0.1:8766';
 const executablePath=process.env.KNOWLEDGE_CHROME_PATH||path.join(process.env.LOCALAPPDATA,'ms-playwright','chromium-1228','chrome-win64','chrome.exe');
 const browser=await chromium.launch({headless:true,executablePath});
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];const result={};
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(base+'/v2');
  await page.locator('#type-list .type').first().waitFor();
  result.dictionary=await page.locator('#type-list .type').count();assert.equal(result.dictionary,103);
  await page.locator('#f-q').fill('귀미');assert.equal(await page.locator('#type-list .type').count(),1);
  assert((await page.locator('#type-list').innerText()).includes('옛 관상서'));
  await page.locator('#f-q').fill('');await page.locator('#f-overall').selectOption('길');
  result.good=await page.locator('#type-list .type').count();assert(result.good>0&&result.good<103);
  await page.locator('#f-overall').selectOption('');
  await page.locator('#perf table').first().waitFor();
  await page.locator('#front').setInputFiles(path.join(root,'data','raw','frll','neutral_front','neutral_front','025_03.jpg'));
  await page.locator('#run').click();
  await page.locator('#result .parts').waitFor({timeout:120000});
  const text=await page.locator('#result').innerText();
  assert(text.includes('부위별 형태'));
  result.feature_rows=await page.locator('#result .feat').count();assert(result.feature_rows>=15);
  result.type_cards=await page.locator('#result .type').count();
  await page.screenshot({path:path.join(out,'v2_분석화면_FRLL예시.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  result.mobile_no_hscroll=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
  await page.screenshot({path:path.join(out,'v2_모바일.png'),fullPage:false});
  assert.deepEqual(errors,[]);result.page_errors=errors.length;result.ok=true;
 }catch(e){result.ok=false;result.error=String(e);result.page_errors=errors}
 finally{fs.writeFileSync(path.join(out,'browser_check_v2.json'),JSON.stringify(result,null,1));await browser.close();console.log(JSON.stringify(result));if(!result.ok)process.exit(1)}
})();
