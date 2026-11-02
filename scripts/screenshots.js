/**
 * BrainScan AI - Automated Screenshot Generator
 * Uses Playwright to capture portfolio-quality screenshots of the Streamlit application.
 *
 * Usage:
 *   node scripts/screenshots.js [BASE_URL]
 *
 * Inside Docker:
 *   docker run --rm -v "${PWD}:/work" -w /work mcr.microsoft.com/playwright:v1.48.0-noble bash -c "npm install playwright@1.48.0 --no-save && node scripts/screenshots.js http://host.docker.internal:8501"
 */

const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function run() {
  const baseUrl = process.argv[2] || process.env.BASE_URL || 'http://localhost:8501';
  const screenshotsDir = path.resolve(__dirname, '../docs/screenshots');
  const sampleImagesDir = path.resolve(__dirname, '../docs/sample_images');

  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  console.log(`Connecting to BrainScan AI Streamlit at ${baseUrl}...`);

  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 2
  });

  const page = await context.newPage();

  try {
    // 1. Home / Initial Landing Page
    console.log('1. Capturing home landing page...');
    await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);

    const homeScreenshotPath = path.join(screenshotsDir, '01-home.png');
    await page.screenshot({ path: homeScreenshotPath, fullPage: false });
    console.log(`Saved: ${homeScreenshotPath}`);

    // 2. Upload MRI Sample 1: Glioma scan
    console.log('2. Uploading glioma sample MRI...');
    const gliomaPath = path.join(sampleImagesDir, 'glioma_sample.jpg');
    await page.locator('input[type="file"]').setInputFiles(gliomaPath);
    await page.waitForSelector('text=Prediction:', { timeout: 25000 });
    await page.waitForTimeout(3000);

    const gliomaScreenshotPath = path.join(screenshotsDir, '02-mri-analysis.png');
    await page.screenshot({ path: gliomaScreenshotPath, fullPage: false });
    console.log(`Saved: ${gliomaScreenshotPath}`);

    // 3. Upload MRI Sample 2: Pituitary scan
    console.log('3. Uploading pituitary sample MRI...');
    await page.goto(baseUrl, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    const pituitaryPath = path.join(sampleImagesDir, 'pituitary_sample.jpg');
    await page.locator('input[type="file"]').setInputFiles(pituitaryPath);
    await page.waitForSelector('text=Prediction:', { timeout: 25000 });
    await page.waitForTimeout(3000);

    const pituitaryScreenshotPath = path.join(screenshotsDir, '03-probability-chart.png');
    await page.screenshot({ path: pituitaryScreenshotPath, fullPage: false });
    console.log(`Saved: ${pituitaryScreenshotPath}`);

    // 4. Test error handling with an invalid corrupt file
    console.log('4. Testing error state with invalid file...');
    await page.goto(baseUrl, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    const corruptFilePath = path.join(sampleImagesDir, 'corrupt_file.png');
    await page.locator('input[type="file"]').setInputFiles(corruptFilePath);

    await page.waitForSelector('text=Could not read this file as an image.', { timeout: 20000 });
    await page.waitForTimeout(2000);

    const errorScreenshotPath = path.join(screenshotsDir, '04-error-handling.png');
    await page.screenshot({ path: errorScreenshotPath, fullPage: false });
    console.log(`Saved: ${errorScreenshotPath}`);

    console.log('All screenshots captured successfully!');
  } catch (err) {
    console.error('Error during screenshot capture:', err);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

run();
