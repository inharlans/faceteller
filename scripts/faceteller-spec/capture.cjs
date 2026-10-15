const {chromium}=require('playwright');
const path=require('path');
(async()=>{
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
const page=await browser.newPage({viewport:{width:1120,height:1200},deviceScaleFactor:1});
await page.goto('file:///C:/Users/user/Documents/카카오톡%20받은%20파일/페이스텔러%20프로토타입.html');
await page.evaluate(()=>document.fonts.ready);
for(const name of ['main','capture','upload','analyze','result','sheet']){
await page.locator(`#steps [data-go="${name}"]`).click();
if(name==='capture'){await page.locator('#btn-start').click();await page.waitForTimeout(850);}
if(name==='analyze')await page.waitForTimeout(600);
await page.locator('.device').screenshot({path:path.resolve('outputs/faceteller-spec/figures/mockup-'+name+'.png')});
}
console.log(JSON.stringify(await page.evaluate(()=>({title:document.title,inputs:[...document.querySelectorAll('input')].map(x=>x.type),screens:[...document.querySelectorAll('.stage')].map(x=>({id:x.id,text:x.innerText.slice(0,250)}))})),null,2));
await browser.close();
})();
