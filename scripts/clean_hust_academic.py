#!/usr/bin/env python3
"""Remove known CMS boilerplate from the small HUST academic corpus.

Run this after ``fetch_public_pages.py``.  The markers are intentionally tied
to the eight reviewed public pages in ``data/hust_academic/urls.csv``; the
script refuses to rewrite a file if a marker is missing or the result is too
short.
"""

from __future__ import annotations

import re
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "hust_academic"

# filename: (first text to keep, first text after the article to discard)
ARTICLE_MARKERS = {
    "hust-course-registration-2026-1.md": (
        "KẾ HOẠCH MỞ ĐĂNG KÝ LỚP KỲ 1 NĂM HỌC 2026-2027 (20261)",
        "BAN ĐÀO TẠO",
    ),
    "hust-summer-registration-2025-3.md": (
        "KẾ HOẠCH MỞ ĐĂNG KÝ LỚP KỲ HÈ 20253",
        "BAN ĐÀO TẠO",
    ),
    "hust-study-plan-2025-2.md": (
        "Đăng ký kế hoạch học tập cho học kỳ 2 năm học 2025-2026 (2025.2)",
        "BAN ĐÀO TẠO",
    ),
    "hust-equivalent-course-process.md": ("Bước 1:", "Tin nổi bật"),
    "hust-graduation-project-registration.md": (
        "THÔNG BÁO VỀ VIỆC ĐIỀU CHỈNH ĐIỀU KIỆN ĐĂNG KÝ HỌC PHẦN ĐỒ ÁN TỐT NGHIỆP",
        "Tin nổi bật",
    ),
    "hust-full-math-class-registration.md": (
        "Để đăng ký học tập vào các lớp đầy",
        "Các thông báo khác",
    ),
    "hust-unavailable-course-guidance.md": (
        "Với sinh viên K61 trở về trước",
        "Các thông báo khác",
    ),
    "hust-postgraduate-registration-2026-1.md": (
        "Thông báo kế hoạch đăng ký học phần và giảng dạy học kỳ 1 2026 2027 đợt 1.",
        "Các tin liên quan",
    ),
}

PUBLIC_CONTACT_REDACTIONS = {
    "hust-equivalent-course-process.md": [
        (
            r"cho cô Nguyễn Thị Hà Thu \(thu\.nguyenthiha@hust\.edu\.vn\), phòng Đào tạo, "
            r"cc cho thầy Cao Tuấn Dũng: dung\.caotuan@hust\.edu\.vn, Hiệu phó Trường CNTT&TT "
            r"và cô Vân Thu thu\.truongthivan@hust\.edu\.vn, giáo vụ Trường",
            "cho phòng Đào tạo theo hướng dẫn trên nguồn chính thức",
        ),
        (
            r"cho cô Nguyễn Thị Hà Thu \(thu\.nguyenthiha@hust\.edu\.vn\), phòng Đào tạo",
            "cho phòng Đào tạo theo hướng dẫn trên nguồn chính thức",
        ),
    ],
    "hust-graduation-project-registration.md": [
        (
            r"gửi tới giáo vụ phụ trách \(cô Trương Thị Vân Thu\s+thuttv@soict\.hust\.edu\.vn, "
            r"phòng 505 B1\), cc hòm thư tuvanhoctap@soict\.hust\.edu\.vn",
            "gửi tới giáo vụ phụ trách tại phòng 505 B1 và hòm thư tư vấn học tập của Trường",
        ),
    ],
}


def split_document(text: str) -> tuple[str, str]:
    match = re.match(r"\A(---\n.*?\n---\n\n# .*?\n\n)(.*)\Z", text, flags=re.DOTALL)
    if not match:
        raise ValueError("missing expected front matter and title")
    return match.group(1), match.group(2)


def clean_file(path: Path, start_marker: str, end_marker: str) -> None:
    prefix, body = split_document(path.read_text(encoding="utf-8"))
    start = body.find(start_marker)
    if start < 0:
        raise ValueError(f"{path.name}: start marker not found: {start_marker!r}")
    end = body.find(end_marker, start + len(start_marker))
    if end < 0:
        if body.lstrip().startswith(start_marker):
            article = body.strip()
        else:
            raise ValueError(f"{path.name}: end marker not found: {end_marker!r}")
    else:
        article = body[start:end]
    article = article.replace("\r\n", "\n").replace("\xa0", " ")
    article = re.sub(r"[ \t]+\n", "\n", article)
    article = re.sub(r"\n{3,}", "\n\n", article).strip()
    for pattern, replacement in PUBLIC_CONTACT_REDACTIONS.get(path.name, []):
        article = re.sub(pattern, replacement, article)
    if len(article) < 250:
        raise ValueError(f"{path.name}: cleaned article is unexpectedly short")
    path.write_text(f"{prefix}{article}\n", encoding="utf-8")
    print(f"Cleaned {path.name}: {len(article)} content characters")


def main() -> int:
    for filename, markers in ARTICLE_MARKERS.items():
        clean_file(DATA_DIR / filename, *markers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
