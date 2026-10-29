/* Optional: install Playwright; OCR_TEST_BROWSER / OCR_TEST_URL override local defaults. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '../..');
const out = path.join(root, 'output/ocr_replacement_validation');
(async () => {
 const executablePath = process.env.OCR_TEST_BROWSER;
 const browser = await chromium.launch({headless:true,...(executablePath?{executablePath}:{})});
 const context = await browser.newContext({viewport:{width:1500,height:1050},acceptDownloads:true});
 const page = await context.newPage();
 const errors=[];page.on('pageerror', e=>errors.push(e.message));
 try {
  await page.goto(process.env.OCR_TEST_URL || 'http://127.0.0.1:8765');
  await page.locator('.recent-card').first().waitFor();
  assert.equal(await page.locator('.recent-card').count(), Number(await page.locator('#recent-count').textContent()));
  const migration=JSON.parse(fs.readFileSync(path.join(root,'transfer_kernel_ocr/output/migration-report.json'),'utf8'));
  const migrated=JSON.parse(fs.readFileSync(path.join(root,`transfer_kernel_ocr/output/${migration.jobs[0].id}/summary/document.json`),'utf8'));
  await page.locator('.recent-card').filter({has:page.locator('strong',{hasText:migrated.name})}).first().click();
  await page.locator('#editor-screen').waitFor({state:'visible'});
  assert.equal(await page.evaluate(()=>state.doc.migration.schema),'transfer_kernel_ocr/1');
  await page.screenshot({path:path.join(out,'migrated-editor.png')});
  await page.locator('.rail-button[data-screen="input"]').click();
  const samples=path.join(root,'input/pdf/test/graph');
  const sample=fs.readdirSync(samples).find(name=>name.endsWith('.png'));
  await page.locator('#file-input').setInputFiles(path.join(samples,sample));
  await page.locator('#parse-files').click();
  await page.locator('#editor-screen').waitFor({state:'visible',timeout:180000});
  await page.locator('#busy-overlay').waitFor({state:'hidden',timeout:180000});
  assert.equal(await page.evaluate(()=>state.doc.pages.length),1);
  await page.locator('#add-text').click();
  await page.locator('[data-prop="text"]').fill('兼容性验证 Compatibility');
  await page.locator('[data-prop="text"]').press('Tab');
  await page.locator('[data-prop="width"]').fill('300');await page.locator('[data-prop="width"]').press('Tab');
  await page.locator('[data-prop="x"]').fill('42');await page.locator('[data-prop="x"]').press('Tab');
  assert.equal(await page.evaluate(()=>selected().x),42);
  await page.locator('#undo').click();
  assert.notEqual(await page.evaluate(()=>selected()?.x),42);
  await page.locator('#redo').click();
  assert.equal(await page.evaluate(()=>selected().x),42);
  await page.locator('#json-open').click();
  const scene=JSON.parse(await page.locator('#json-editor').inputValue());
  assert(scene.pages[0].elements.some(e=>e.text==='兼容性验证 Compatibility'&&e.x===42));
  await page.locator('#json-close').click();
  await page.screenshot({path:path.join(out,'sample-editor.png')});
  await page.locator('#go-output').click();
  const wait=page.waitForEvent('download',{timeout:120000});
  await page.locator('[data-export="json"]').click();
  const download=await wait;
  await download.saveAs(path.join(out,'browser-download.json'));
  const doc=JSON.parse(fs.readFileSync(path.join(out,'browser-download.json'),'utf8'));
  assert(doc.pages[0].elements.some(e=>e.text==='兼容性验证 Compatibility'&&e.x===42));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify({passed:true,sample,documentId:doc.id,errors,
   checks:['all recent documents visible','legacy document opens','sample PNG upload','text edit','coordinate edit','undo','redo','JSON sync','download']},null,2));
  console.log('Browser workflow passed. Screenshots: '+out);
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
