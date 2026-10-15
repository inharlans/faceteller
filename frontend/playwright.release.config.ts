import {defineConfig} from '@playwright/test';
import os from 'node:os';
import path from 'node:path';
const baseURL=process.env.FACETELLER_TEST_URL||'http://127.0.0.1:8767';
const url=new URL(baseURL);
if(!['http:','https:'].includes(url.protocol)||!['127.0.0.1','localhost','[::1]'].includes(url.hostname)||url.username||url.password||url.pathname!=='/'||url.search||url.hash)throw new Error('Actual-photo test requires a loopback origin');
export default defineConfig({
  testDir:'./tests',testMatch:'production.integration.ts',timeout:90000,workers:1,retries:0,reporter:'list',
  outputDir:path.join(os.tmpdir(),'faceteller-production-browser'),
  use:{baseURL,ignoreHTTPSErrors:process.env.FACETELLER_ALLOW_SELF_SIGNED==='1',viewport:{width:1280,height:1000},headless:true,
    trace:'off',screenshot:'off',launchOptions:{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
      args:process.env.FACETELLER_TLS_SPKI?['--ignore-certificate-errors-spki-list='+process.env.FACETELLER_TLS_SPKI]:[]}},
});
