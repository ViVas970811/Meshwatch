"""Trigger an investigation in a fresh Selenium session, then re-capture the
three dashboard pages that depend on populated client state:

  * alert_detail.png       (with the investigation populated and risk factors visible)
  * ai_investigation.png   (AI investigation panel section)
  * cases.png              (now contains the case created by the investigation)

The other three pages (monitor, network, model) come from server-side data
and were already captured by `capture_dashboard.py`.
"""

from __future__ import annotations

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
OUT.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:5173"
ALERT_ID = "inv-3013429"
VIEWPORT = (1440, 900)


def build_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--force-device-scale-factor=1.25")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    drv = webdriver.Chrome(options=opts)
    drv.set_window_size(*VIEWPORT)
    return drv


def screenshot(drv: webdriver.Chrome, name: str) -> None:
    path = OUT / name
    drv.save_screenshot(str(path))
    print(f"  -> {path.name}")


def click_run_investigation(drv: webdriver.Chrome) -> None:
    """Find and click the 'Run investigation' button on the Alert detail page."""
    # Try by text first (most robust against className changes)
    btn = drv.execute_script("""
        const buttons = Array.from(document.querySelectorAll('button'));
        const b = buttons.find(el => el.textContent.trim().toLowerCase().includes('run investigation'));
        if (b) { b.click(); return true; } return false;
    """)
    if not btn:
        raise RuntimeError("could not locate 'Run investigation' button")
    print("  clicked Run investigation")


def wait_for_investigation(drv: webdriver.Chrome, timeout: float = 30.0) -> None:
    """Wait until the AI investigation panel shows a SUMMARY (= populated)."""
    end = time.time() + timeout
    while time.time() < end:
        ok = drv.execute_script("""
            return Array.from(document.querySelectorAll('*'))
                .some(el => el.textContent && el.textContent.trim().toUpperCase().startsWith('SUMMARY'));
        """)
        if ok:
            return
        time.sleep(0.5)
    raise TimeoutError("investigation did not populate within timeout")


def main() -> None:
    print(f"Launching headless Chrome at {VIEWPORT[0]}x{VIEWPORT[1]}...")
    drv = build_driver()
    try:
        # 1. Open the alert detail page
        print(f"Opening /alerts/{ALERT_ID}...")
        drv.get(f"{BASE_URL}/alerts/{ALERT_ID}")
        WebDriverWait(drv, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "main"))
        )
        time.sleep(2.0)

        # 2. Click 'Run investigation' to populate the panel + create the case
        click_run_investigation(drv)

        # 3. Wait for the investigation to come back from the API
        print("Waiting for investigation to populate...")
        wait_for_investigation(drv)
        time.sleep(1.5)  # let the React state settle

        # 4. Capture the alert detail top (investigation has now populated risk
        #    factors and connected entities under the hood)
        print("Capturing alert_detail.png (with investigation populated)...")
        drv.execute_script("window.scrollTo({top: 0, behavior: 'instant'});")
        time.sleep(0.6)
        screenshot(drv, "alert_detail.png")

        # 5. Capture the AI investigation panel section
        #    Scroll past the top KPI strip so the investigation panel is the
        #    visual focus of the screenshot (different from alert_detail.png).
        print("Capturing ai_investigation.png...")
        drv.execute_script("""
            const cards = Array.from(document.querySelectorAll('h2, h3'));
            const h = cards.find(e => /ai investigation/i.test(e.textContent || ''));
            if (h) {
                const r = h.getBoundingClientRect();
                window.scrollTo({top: window.scrollY + r.top - 80, behavior: 'instant'});
            }
        """)
        time.sleep(1.0)
        screenshot(drv, "ai_investigation.png")

        # 6. Navigate to Cases via SPA click (NOT drv.get -- a full page load
        #    would wipe the in-memory Zustand case store).
        print("Capturing cases.png (populated)...")
        clicked = drv.execute_script("""
            const links = Array.from(document.querySelectorAll('a'));
            const cases = links.find(a => /cases/i.test(a.textContent || '') && a.getAttribute('href') === '/cases');
            if (cases) { cases.click(); return true; } return false;
        """)
        if not clicked:
            raise RuntimeError("could not find Cases nav link for SPA navigation")
        WebDriverWait(drv, 8).until(
            lambda d: "/cases" in d.current_url
        )
        time.sleep(1.5)
        drv.execute_script("window.scrollTo({top: 0, behavior: 'instant'});")
        time.sleep(0.5)
        screenshot(drv, "cases.png")

        print("\nDone. Three pages re-captured with populated state.")
    finally:
        drv.quit()


if __name__ == "__main__":
    main()
