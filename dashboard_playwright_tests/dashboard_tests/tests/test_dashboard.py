"""
PMO Dashboard - Playwright Regression Tests
============================================
Run locally:  python tests/test_dashboard.py
Run via pytest: pytest tests/test_dashboard.py -v

Dashboard HTML is always located at:
    <project_root>/package/dashboard/<dashboard_name>/<file>.html
"""

import re
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

# ─────────────────────────────────────────────
# CONFIG — update dashboard name and file here
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DASHBOARD_NAME = "audit"       # folder name under package/dashboard/
DASHBOARD_FILE = "a.html"      # generated HTML filename

DASHBOARD_PATH = PROJECT_ROOT / "package" / "dashboard" / DASHBOARD_NAME / DASHBOARD_FILE
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def load_dashboard(page: Page):
    """Load the generated HTML file in the browser."""
    assert DASHBOARD_PATH.exists(), (
        f"Dashboard HTML not found at: {DASHBOARD_PATH}\n"
        f"Run your Python generator script first."
    )
    page.goto(f"file:///{DASHBOARD_PATH.resolve()}")
    page.wait_for_load_state("domcontentloaded")


# ─────────────────────────────────────────────
# 1. Structure checks
# ─────────────────────────────────────────────

def test_structure(page: Page):
    """Verify all key dashboard elements are present after generation."""
    load_dashboard(page)

    # Page title
    assert page.title() != "", "Page title is empty"

    # Tables
    assert page.locator("table").count() > 0, "No tables found"

    # KPI cards — adjust selector to match your actual CSS class
    assert page.locator(".kpi-card, .card, [class*='kpi']").count() > 0, "No KPI cards found"

    # Charts (canvas for Chart.js / D3, svg for inline charts)
    chart_count = (
        page.locator("canvas").count() +
        page.locator("svg").count()
    )
    assert chart_count > 0, "No charts (canvas or svg) found"

    # Filters / dropdowns
    assert page.locator("select, [class*='filter']").count() > 0, "No filters found"


# ─────────────────────────────────────────────
# 2. Security checks — CSP, XSS, JS
#    NOTE: Commented out for now.
#    Uncomment once vulnerability fixes are stable.
# ─────────────────────────────────────────────

# def test_csp_meta_tag(page: Page):
#     """CSP meta tag must exist with script-src and no unsafe-inline."""
#     load_dashboard(page)
#
#     csp = page.locator("meta[http-equiv='Content-Security-Policy']")
#     assert csp.count() > 0, "CSP meta tag missing"
#
#     content = csp.get_attribute("content") or ""
#     assert "script-src" in content, "CSP missing script-src directive"
#     assert "unsafe-inline" not in content, "CSP allows unsafe-inline — XSS risk"


# def test_no_inline_event_handlers(page: Page):
#     """No dangerous inline event attributes should exist (XSS remnants)."""
#     load_dashboard(page)
#
#     dangerous = ["onclick", "onmouseover", "onerror", "onload", "onfocus"]
#     html = page.content()
#
#     for attr in dangerous:
#         matches = re.findall(
#             rf'<(?!body|html)[^>]+\s{attr}\s*=', html, re.IGNORECASE
#         )
#         assert len(matches) == 0, f"Inline {attr} handler found — potential XSS"


# def test_no_javascript_protocol(page: Page):
#     """No javascript: protocol in href attributes."""
#     load_dashboard(page)
#
#     html = page.content()
#     matches = re.findall(r'href\s*=\s*["\']javascript:', html, re.IGNORECASE)
#     assert len(matches) == 0, "javascript: protocol found in href — XSS risk"


# def test_no_eval_in_scripts(page: Page):
#     """No eval() usage in inline scripts."""
#     load_dashboard(page)
#
#     html = page.content()
#     eval_matches = re.findall(
#         r'<script[^>]*>.*?eval\s*\(', html, re.IGNORECASE | re.DOTALL
#     )
#     assert len(eval_matches) == 0, "eval() found in inline script — JS injection risk"


# def test_console_no_errors(page: Page):
#     """No JS console errors after page load."""
#     errors = []
#     page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
#     load_dashboard(page)
#     page.wait_for_timeout(1000)
#
#     assert len(errors) == 0, f"Console errors found: {errors}"


# ─────────────────────────────────────────────
# 3. Visual / screenshot
# ─────────────────────────────────────────────

def test_screenshot(page: Page):
    """Take a full-page screenshot for visual verification."""
    load_dashboard(page)
    page.wait_for_timeout(500)  # allow charts to finish rendering

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    screenshot_path = SCREENSHOT_DIR / f"{DASHBOARD_NAME}.png"

    page.screenshot(path=screenshot_path, full_page=True)
    print(f"  Screenshot saved → {screenshot_path}")


# ─────────────────────────────────────────────
# 4. Local runner
# ─────────────────────────────────────────────

def run_all():
    """Run all active tests and print a summary."""
    active_tests = [
        test_structure,
        test_screenshot,
        # --- Security tests (uncomment when ready) ---
        # test_csp_meta_tag,
        # test_no_inline_event_handlers,
        # test_no_javascript_protocol,
        # test_no_eval_in_scripts,
        # test_console_no_errors,
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        passed, failed = 0, []

        for t in active_tests:
            try:
                t(page)
                print(f"  ✅  {t.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"  ❌  {t.__name__}: {e}")
                failed.append(t.__name__)
            except Exception as e:
                print(f"  💥  {t.__name__} (unexpected error): {e}")
                failed.append(t.__name__)

        browser.close()

        print(f"\n  {passed}/{len(active_tests)} passed")
        if failed:
            print(f"  Failed: {', '.join(failed)}")
            exit(1)


if __name__ == "__main__":
    run_all()
