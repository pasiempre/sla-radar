from __future__ import annotations
from pathlib import Path
from weasyprint import HTML

def export_ops_brief(html_str: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_str).write_pdf(str(out_path))
    return out_path