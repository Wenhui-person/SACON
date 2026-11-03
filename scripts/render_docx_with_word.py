from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import win32com.client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    docx = args.docx.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{docx.stem}.pdf"

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx), ReadOnly=True, AddToRecentFiles=False)
        try:
            doc.ExportAsFixedFormat(str(pdf), 17)
        finally:
            doc.Close(False)
    finally:
        word.Quit()

    subprocess.run(
        ["pdftoppm", "-png", "-r", "150", str(pdf), str(out_dir / "page")],
        check=True,
    )
    print(pdf)


if __name__ == "__main__":
    main()
