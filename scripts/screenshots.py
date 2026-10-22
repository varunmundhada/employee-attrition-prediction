"""Takes screenshots of the running Streamlit app for the README, report and slides.

Start the app first (run_app.bat), then: python -m scripts.screenshots
"""

import time

from playwright.sync_api import sync_playwright

from src import config

URL = "http://localhost:8501"
OUT = config.ROOT / "docs" / "screenshots"


def wait_ready(page, text, timeout=120_000):
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
                           timeout=timeout)
    time.sleep(2.5)


def goto(page, label):
    page.locator("[data-testid=stSidebar]").get_by_text(label, exact=True).click()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(URL)
        page.set_viewport_size({"width": 1440, "height": 1900})
        wait_ready(page, "HR Attrition Dashboard")
        page.screenshot(path=str(OUT / "01_dashboard.png"))

        goto(page, "Predict an Employee")
        wait_ready(page, "Predict attrition risk for one employee")
        page.set_viewport_size({"width": 1440, "height": 1750})
        time.sleep(1.5)
        page.screenshot(path=str(OUT / "02_predict_form.png"))
        page.get_by_role("button", name="Predict attrition risk").click()
        wait_ready(page, "Predicted to", timeout=240_000)
        # Streamlit scrolls inside its own container, so render the whole page in a tall
        # viewport and crop to the result section
        page.set_viewport_size({"width": 1440, "height": 3200})
        time.sleep(2.5)
        top = page.get_by_text("Predicted to").first.bounding_box()["y"] - 160
        page.screenshot(path=str(OUT / "03_predict_result.png"),
                        clip={"x": 0, "y": max(top, 0), "width": 1440, "height": 760})

        goto(page, "Batch Prediction")
        page.set_viewport_size({"width": 1440, "height": 1500})
        wait_ready(page, "Employees scored", timeout=240_000)
        page.screenshot(path=str(OUT / "04_batch.png"))

        goto(page, "Model Performance")
        page.set_viewport_size({"width": 1440, "height": 1500})
        wait_ready(page, "Top 15 features")
        page.screenshot(path=str(OUT / "05_performance.png"))
        browser.close()
    print("Screenshots saved to", OUT)


if __name__ == "__main__":
    main()
