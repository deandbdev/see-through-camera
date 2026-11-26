"""Прави PDF с шахматна дъска за печат (A4 пейзаж, векторен, точен мащаб).

Пускане:  .venv/Scripts/python.exe src/make_chessboard.py
"""
from pathlib import Path

COLS, ROWS = 10, 7   # квадрати -> 9x6 вътрешни ъгъла
SQUARE_MM = 25.0
PAGE_W_MM, PAGE_H_MM = 297.0, 210.0
OUT = Path(__file__).resolve().parent.parent / "calib" / "chessboard_9x6_25mm_A4.pdf"


def pt(mm):
    return mm * 72.0 / 25.4


def rect(x, y, w, h):
    return f"{pt(x):.3f} {pt(y):.3f} {pt(w):.3f} {pt(h):.3f} re f"


def text(x, y, s):
    return f"BT /F1 9 Tf {pt(x):.3f} {pt(y):.3f} Td ({s}) Tj ET"


def page_content():
    x0 = (PAGE_W_MM - COLS * SQUARE_MM) / 2
    y0 = (PAGE_H_MM - ROWS * SQUARE_MM) / 2
    ops = ["0 g"]
    ops += [rect(x0 + c * SQUARE_MM, y0 + r * SQUARE_MM, SQUARE_MM, SQUARE_MM)
            for r in range(ROWS) for c in range(COLS) if (r + c) % 2 == 0]
    # контролна линия 100 mm с чертички в краищата
    ops += [rect(x0, 9, 100, 0.3), rect(x0, 7, 0.3, 4), rect(x0 + 99.7, 7, 0.3, 4)]
    ops.append(text(x0 + 104, 8, f"= 100 mm. Print at 100% / Actual size. "
                                 f"Square {SQUARE_MM:g} mm, inner corners {COLS - 1}x{ROWS - 1}."))
    return "\n".join(ops).encode("ascii")


def write_pdf(path, pages, page_w=PAGE_W_MM, page_h=PAGE_H_MM):
    """pages: списък със съдържанието (PDF оператори) на всяка страница."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, content in enumerate(pages):
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pt(page_w):.3f} {pt(page_h):.3f}] "
                        f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>").encode("ascii"))
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


if __name__ == "__main__":
    write_pdf(OUT, [page_content()])
    print(OUT)
