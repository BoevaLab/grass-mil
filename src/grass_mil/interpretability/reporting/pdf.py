from __future__ import annotations

from pathlib import Path


def render_pdf_via_playwright(
    *,
    html_path: Path,
    pdf_path: Path,
    timeout_ms: int = 60_000,
) -> Path:
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Playwright is unavailable. Install playwright and browser binaries to enable PDF export."
        ) from exc

    with sync_playwright() as p:  # pragma: no cover - depends on browser runtime
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{html_path}", wait_until="networkidle", timeout=timeout_ms)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "15mm", "bottom": "15mm", "left": "12mm", "right": "12mm"},
        )
        browser.close()
    return pdf_path
