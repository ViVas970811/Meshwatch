"""Capture desktop-viewport screenshots of the running Meshwatch dashboard.

Prereqs:
  * Dashboard running at http://localhost:5173 (`make dashboard`)
  * API running with seeded data; alert `inv-3013429` should have a populated investigation
  * Selenium >= 4.x (auto-manages chromedriver)
  * Chrome installed on the host

Outputs land in `medium/images/`.
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
ALERT_ID = "inv-3013429"  # populated investigation from the LinkedIn screenshot session

# Desktop viewport. 1440x900 reads as a 13" laptop; gives the dashboard room to
# breathe without making the tables look stretched.
VIEWPORT = (1440, 900)


def build_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--force-device-scale-factor=1.25")  # crisper output at native res
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    drv = webdriver.Chrome(options=opts)
    drv.set_window_size(*VIEWPORT)
    return drv


def navigate(drv: webdriver.Chrome, path: str, settle: float = 1.5) -> None:
    """Navigate via in-page React Router by clicking the matching nav <a>."""
    drv.get(f"{BASE_URL}{path}")
    time.sleep(settle)


def screenshot(drv: webdriver.Chrome, name: str, *, full_page: bool = False) -> None:
    target = OUT / name
    if full_page:
        # Resize window to document height so the screenshot captures everything.
        h = drv.execute_script("return document.documentElement.scrollHeight")
        drv.set_window_size(VIEWPORT[0], max(h, VIEWPORT[1]))
        time.sleep(0.4)
        drv.save_screenshot(str(target))
        drv.set_window_size(*VIEWPORT)
    else:
        drv.save_screenshot(str(target))
    print(f"  -> {target.name}")


def scroll_to(drv: webdriver.Chrome, y: int) -> None:
    drv.execute_script(f"window.scrollTo({{ top: {y}, behavior: 'instant' }});")
    time.sleep(0.5)


def wait_for(drv: webdriver.Chrome, css: str, timeout: float = 8.0) -> None:
    WebDriverWait(drv, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, css))
    )


def main() -> None:
    print(f"Launching headless Chrome at {VIEWPORT[0]}x{VIEWPORT[1]}...")
    drv = build_driver()
    try:
        # Monitor (root)
        print("Capturing Monitor page...")
        navigate(drv, "/", settle=2.5)
        wait_for(drv, "main")
        scroll_to(drv, 0)
        screenshot(drv, "monitor.png")

        # Network
        print("Capturing Network page...")
        navigate(drv, "/network", settle=2.5)
        wait_for(drv, "main")
        scroll_to(drv, 0)
        # Force-directed graphs animate; let them settle past the early kinetic phase.
        time.sleep(3.0)
        screenshot(drv, "network.png")

        # Model
        print("Capturing Model page...")
        navigate(drv, "/model", settle=2.0)
        wait_for(drv, "main")
        scroll_to(drv, 0)
        time.sleep(1.0)
        screenshot(drv, "model.png")

        # Cases (populated by the prior investigation we ran)
        print("Capturing Cases page...")
        navigate(drv, "/cases", settle=2.0)
        wait_for(drv, "main")
        scroll_to(drv, 0)
        screenshot(drv, "cases.png")

        # Alert detail top
        print("Capturing Alert detail (top)...")
        navigate(drv, f"/alerts/{ALERT_ID}", settle=2.0)
        wait_for(drv, "main")
        scroll_to(drv, 0)
        screenshot(drv, "alert_detail.png")

        # AI investigation panel (scroll into view)
        print("Capturing AI investigation panel...")
        # Scroll the AI investigation section into view.
        drv.execute_script("""
            const headings = Array.from(document.querySelectorAll('h2, h3'));
            const h = headings.find(e => e.textContent.trim().toLowerCase().includes('ai investigation'));
            if (h) h.scrollIntoView({block: 'start'});
        """)
        time.sleep(1.2)
        screenshot(drv, "ai_investigation.png")

        print("\nAll screenshots saved.")
    finally:
        drv.quit()


if __name__ == "__main__":
    main()
