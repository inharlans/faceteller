const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});const p=await b.newPage({viewport:{width:1120,height:1200},deviceScaleFactor:1});await p.goto('file:///C:/Users/user/Documents/카카오톡%20받은%20파일/페이스텔러%20프로토타입.html');await p.evaluate(()=>document.fonts.ready);await p.locator('#steps [data-go="result"]').click();
await p.locator('#cards > li').first().screenshot({path:'outputs/faceteller-spec/figures/mockup-card.png'});
const r=await p.locator('.device').boundingBox();await p.screenshot({path:'outputs/faceteller-spec/figures/mockup-result-top.png',clip:{x:r.x,y:r.y,width:r.width,height:735}});
await b.close();})();
