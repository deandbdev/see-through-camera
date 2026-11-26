"""PDF с ArUco маркери за печат: по един на A4 (портрет), векторно, точен мащаб.

  .venv/Scripts/python.exe src/make_markers.py
"""
import cv2

import common as c
from make_chessboard import pt, rect, text, write_pdf

IDS = (0, 1, 2, 3)
MARKER_MM = 120.0
DICTIONARY = cv2.aruco.DICT_4X4_50
OUT = c.CALIB_DIR / "aruco_4x4_ids0-3_120mm_A4.pdf"


def page(marker_id):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    n = dictionary.markerSize + 2  # битовете плюс черната рамка
    bits = cv2.aruco.generateImageMarker(dictionary, marker_id, n)
    cell = MARKER_MM / n
    x0, y0 = (210 - MARKER_MM) / 2, (297 - MARKER_MM) / 2
    ops = ["0 g"]
    ops += [rect(x0 + col * cell, y0 + (n - 1 - row) * cell, cell, cell)
            for row in range(n) for col in range(n) if bits[row, col] == 0]
    ops.append(f"BT /F1 48 Tf {pt(x0):.3f} {pt(y0 + MARKER_MM + 30):.3f} Td (ID {marker_id}    TOP ^) Tj ET")
    ops += [rect(x0, 40, 100, 0.3), rect(x0, 38, 0.3, 4), rect(x0 + 99.7, 38, 0.3, 4)]
    ops.append(text(x0, 32, f"line = 100 mm, marker = {MARKER_MM:g} mm. Print at 100% / Actual size."))
    # кръстче в центъра на листа отстрани, за мерене с рулетка
    cy = 297 / 2
    ops += [rect(8, cy - 0.15, 20, 0.3), rect(182, cy - 0.15, 20, 0.3),
            rect(105 - 0.15, 62, 0.3, 20)]
    return "\n".join(ops).encode("ascii")


if __name__ == "__main__":
    write_pdf(OUT, [page(i) for i in IDS], page_w=210, page_h=297)
    print(OUT)
