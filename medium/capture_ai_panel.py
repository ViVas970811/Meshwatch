"""Take a full-page screenshot of the alert detail (with investigation populated),
then crop to just the AI investigation panel for ai_investigation.png.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
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


def main() -> None:
    drv = build_driver()
    try:
        print("Opening alert detail and running investigation...")
        drv.get(f"{BASE_URL}/alerts/{ALERT_ID}")
        WebDriverWait(drv, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "main")))
        time.sleep(2.0)

        # Click 'Run investigation'
        drv.execute_script("""
            const btns = Array.from(document.querySelectorAll('button'));
            const b = btns.find(el => /run investigation/i.test(el.textContent || ''));
            if (b) b.click();
        """)

        # Wait for SUMMARY to appear. Llama 3 8B on CPU can take 2+ minutes
        # for the full investigation, so give it a generous budget.
        end = time.time() + 240
        populated = False
        while time.time() < end:
            ok = drv.execute_script("""
                return Array.from(document.querySelectorAll('*'))
                    .some(el => el.textContent && el.textContent.trim().toUpperCase().startsWith('SUMMARY'));
            """)
            if ok:
                populated = True
                break
            time.sleep(0.5)
        if not populated:
            raise TimeoutError("investigation did not populate")
        time.sleep(1.5)

        # Full-page screenshot: resize window to document height so the AI investigation
        # panel content below the fold is captured.
        scroll_h = drv.execute_script("return document.documentElement.scrollHeight")
        print(f"Document height: {scroll_h}px. Resizing for full-page capture...")
        drv.set_window_size(VIEWPORT[0], scroll_h + 50)
        time.sleep(1.0)
        drv.execute_script("window.scrollTo({top: 0, behavior: 'instant'});")
        time.sleep(0.5)

        tmp_path = OUT / "_alert_fullpage.png"
        drv.save_screenshot(str(tmp_path))
        print(f"  -> {tmp_path.name}")

        # Get the AI investigation panel's bounding rect in CSS pixels
        rect = drv.execute_script("""
            const cards = Array.from(document.querySelectorAll('h2, h3'));
            const h = cards.find(e => /ai investigation/i.test(e.textContent || ''));
            if (!h) return null;
            // Walk up to the enclosing card container
            let card = h;
            while (card && card.parentElement) {
                const cs = getComputedStyle(card);
                if (cs.backgroundColor && cs.backgroundColor !== 'rgba(0, 0, 0, 0)' && cs.borderRadius) {
                    if (parseFloat(cs.borderRadius) > 0) break;
                }
                card = card.parentElement;
            }
            const r = card.getBoundingClientRect();
            return {x: r.x, y: r.y + window.scrollY, w: r.width, h: r.height};
        """)
        print(f"AI panel rect (CSS px): {rect}")

        # The screenshot is taken at device pixel ratio 1.25, so multiply by 1.25.
        # Also widen the crop slightly to include some padding around the panel.
        scale = 1.25
        pad = 30  # CSS px of breathing room

        img = Image.open(tmp_path)
        print(f"Full-page image size: {img.size}")

        # Crop coordinates in image pixels
        left = max(0, int((rect["x"] - pad) * scale))
        top = max(0, int((rect["y"] - pad) * scale))
        right = min(img.size[0], int((rect["x"] + rect["w"] + pad) * scale))
        bottom = min(img.size[1], int((rect["y"] + rect["h"] + pad) * scale))
        print(f"Crop box: ({left}, {top}) -> ({right}, {bottom})")

        crop = img.crop((left, top, right, bottom))
        out = OUT / "ai_investigation.png"
        crop.save(out)
        print(f"  -> {out.name}  ({crop.size})")

        # Clean up the fullpage temp
        tmp_path.unlink()

    finally:
        drv.quit()


if __name__ == "__main__":
    main()
