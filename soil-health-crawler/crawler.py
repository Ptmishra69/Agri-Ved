from playwright.sync_api import sync_playwright

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

with sync_playwright() as p:

    browser = p.chromium.launch(
        executable_path=CHROME_PATH,
        headless=False
    )

    page = browser.new_page()

    page.goto(
        "https://www.soilhealth.dac.gov.in/NutrientMapModelVillages",
        wait_until="networkidle"
    )

    print("Page loaded:", page.title())

    browser.close()