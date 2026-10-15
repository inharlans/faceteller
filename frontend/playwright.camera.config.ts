import {defineConfig} from '@playwright/test';
import base from './playwright.release.config';
export default defineConfig({...base,testMatch:'recorded-camera.integration.ts',timeout:120000});
