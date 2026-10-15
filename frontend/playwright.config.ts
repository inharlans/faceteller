import {defineConfig} from '@playwright/test';
import os from 'node:os';
import path from 'node:path';
export default defineConfig({testDir:'./tests',testMatch:'**/*.spec.ts',timeout:45000,workers:1,retries:0,reporter:'list',outputDir:path.join(os.tmpdir(),'faceteller-release-browser'),use:{baseURL:'http://127.0.0.1:5173',viewport:{width:1280,height:1000},headless:true,trace:'off',screenshot:'off',launchOptions:{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}},webServer:{command:'npm run dev -- --mode test',url:'http://127.0.0.1:5173',reuseExistingServer:false,timeout:30000}});
