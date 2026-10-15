const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=path.join(root,'outputs','knowledge-v2');
 const executablePath=process.env.KNOWLEDGE_CHROME_PATH||path.join(process.env.LOCALAPPDATA,'ms-playwright','chromium-1228','chrome-win64','chrome.exe');
 const browser=await chromium.launch({headless:true,executablePath});
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto('file:///'+path.join(out,'관상문헌_측정사전.html').replaceAll('\\','/'));
  assert.equal(await page.locator('#records .source-card').count(),103);
  await page.locator('#query').fill('귀미');assert.equal(await page.locator('#records .source-card').count(),1);
  assert((await page.locator('#records').innerText()).includes('복록의 유무를 직접 말하지 않는다'));
  await page.screenshot({path:path.join(out,'사전_화면.png'),fullPage:true});
  await page.locator('#query').fill('');await page.locator('#topic').selectOption({label:'재물'});await page.locator('#polarity').selectOption({label:'조건부'});
  const filtered=await page.locator('#records .source-card').count();assert(filtered>0&&filtered<103);
  await page.locator('#definitions-tab').click();assert.equal(await page.locator('#definitions .source-card').count(),17);
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'측정정의_모바일.png'),fullPage:false});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.setViewportSize({width:1280,height:900});await page.goto('http://127.0.0.1:8765');
  await page.locator('#content').waitFor({state:'visible'});
  await page.locator('[data-view="literature"]').click();assert.equal(await page.locator('#literature-list .source-card').count(),103);
  await page.locator('#literature-search').fill('녹비');assert.equal(await page.locator('#literature-list .source-card').count(),1);
  await page.locator('[data-view="definitions"]').click();assert.equal(await page.locator('#definition-list .source-card').count(),17);
  await page.locator('[data-view="analyze"]').click();
  await page.locator('#upload').setInputFiles(path.join(root,'data','raw','frll','neutral_front','neutral_front','025_03.jpg'));
  await page.locator('#analyze-button').click();await page.locator('#analysis-results table').waitFor({timeout:60000});
  assert.equal(await page.locator('#analysis-results table tr').count(),11);
  assert((await page.locator('#analysis-results').innerText()).includes('실제 mm나 크다/작다 판정은 아닙니다.'));
  for(const row of await page.locator('#analysis-results table tr').allTextContents())assert(row.includes('구간 기준 미확정')||row.includes('촬영 조건 재검토'));
  await page.screenshot({path:path.join(out,'분석_비율표.png'),fullPage:true});
  await page.locator('[data-view="review"]').click();
  const reference=page.locator('#review-body button[onclick^="openLiterature"]');
  await reference.first().waitFor();await reference.first().click();assert.equal(await page.locator('#literature-list .source-card').count(),1);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser_checks.json'),JSON.stringify({passed:true,checks:['offline_103','search','topic_polarity_filter','17_definitions','mobile_no_overflow','live_dictionary','live_definitions','real_upload_11_ratios','partial_shape_dictionary_link'],page_errors:errors},null,2));
  console.log('Browser checks passed');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
