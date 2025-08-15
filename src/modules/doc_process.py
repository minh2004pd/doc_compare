import os
import time
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
import string
from unidecode import unidecode
from src.utils import (
    group_boxes_by_line, is_scanned_pdf, is_stamp_or_signature, extract_boxes_from_diff,
    extract_boxes_from_diff_v2,
    draw_boxes_on_image,
    ocr_from_image,
    ocr_image_with_gpt4,
    draw_polygons_on_image,
    crop_box_from_image,
    normalize_text,
    normalize_line
)
import diff_match_patch as dmp_module
import re
from collections import defaultdict
from PIL import Image
import string
import base64
import uuid
import fitz
import io
import logging

# Tạo bảng dịch để loại bỏ dấu câu
translator = str.maketrans('', '', string.punctuation)

# Danh sách các ký tự hoặc chuỗi đặc biệt muốn loại bỏ thêm
special_tokens = [
    "...", "…", ",-", "--", "–", "—", "―", "•", "·", "“", "”", "‘", "’", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "=", "+", "-", "*", "&", "^", "%", "$", "#", "@", "~", "`", "'", '"', "…", "…", "…", "…"
]

# Thiết lập logger
def setup_logger(log_path="./logs/doc_pdf.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("doc_pdf_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
    return logger

logger = setup_logger()

def is_bbox_inside(inner, outer):
    # inner, outer: (x0, y0, x1, y1)
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

def is_table_header_valid(table):
    raw = table.extract()
    if not raw or len(raw) == 0:
        return False
    header = " ".join((c or "") for c in raw[0]).lower()
    keywords = ["stt", "tt", "số thứ tự"]
    return any(kw in header for kw in keywords)

def process_doc_file(pdf_path, output_dir="./output"):
    """
    Xử lý file PDF, trích xuất text ngoài bảng (theo từng dòng từ words) và dữ liệu bảng ra 2 file riêng.
    Lọc bảng lặp, bảng con, bảng nhỏ.
    """
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Đang xử lý tệp PDF: {pdf_path}")

    output_txt_path = os.path.join(output_dir, f"{file_name}_result_doc_text.txt")
    output_table_path = os.path.join(output_dir, f"{file_name}_tables.txt")
    seen_tables_global = set()

    with fitz.open(pdf_path) as pdf, \
         open(output_txt_path, "w", encoding="utf-8") as f_text, \
         open(output_table_path, "w", encoding="utf-8") as f_table:

        for i, page in enumerate(pdf):
            tables = page.find_tables()
            # Lọc bảng con: chỉ giữ bảng lớn nhất (không nằm hoàn toàn trong bảng khác)
            bboxes = [table.bbox for table in tables if table.extract() and len(table.extract()) > 1]
            keep = [True] * len(bboxes)
            for idx, bb in enumerate(bboxes):
                for jdx, obb in enumerate(bboxes):
                    if idx != jdx and is_bbox_inside(bb, obb):
                        keep[idx] = False
                        break
            
            valid_tables = [t for t, k in zip(tables, keep) if k and is_table_header_valid(t)]
            valid_bboxes = [t.bbox for t in valid_tables]

            # Trích xuất văn bản ngoài bảng bằng words
            y_tol = 10
            words = page.get_text("words", sort=True)
            line_groups = []
            for w in words:
                x0, y0, x1, y1, txt = w[:5]
                mid_y = (y0 + y1) / 2
                word_bbox = (x0, y0, x1, y1)
                inside = any(is_bbox_inside(word_bbox, bb) for bb in valid_bboxes)
                if inside:
                    continue
                found = False
                for group in line_groups:
                    g_yc = group['yc']
                    if abs(mid_y - g_yc) <= y_tol:  # y_tol có thể điều chỉnh
                        group['words'].append((x0, txt))
                        found = True
                        break
                if not found:
                    line_groups.append({'yc': mid_y, 'words': [(x0, txt)]})

            # Sắp xếp các dòng theo y-center tăng dần (dòng trên trước)
            line_groups.sort(key=lambda g: g['yc'])

            lines = []
            for group in line_groups:
                group['words'].sort(key=lambda t: t[0])  # sắp xếp từ trái qua phải
                line_text = " ".join(t[1] for t in group['words'])
                lines.append(line_text)

            # Làm sạch và ghi từng dòng văn bản ngoài bảng
            for line in lines:
                for token in special_tokens:
                    line = line.replace(token, " ")
                clean_line = line.translate(translator)
                no_accent_line = unidecode(clean_line)
                single_space_line = ' '.join(no_accent_line.split())
                check_text = normalize_text(single_space_line)
                if not single_space_line or re.fullmatch(r"\s*\d+\s*", check_text):
                    continue
                if "tham dinh phap ly" in check_text or "tham dinh" in check_text:
                    logger.info(f" Page: {i+1}  🚫 Bỏ qua dòng chứa 'tham dinh phap ly'")
                    continue
                if single_space_line and single_space_line[0].lower() == 'l' and single_space_line[1] == ' ':
                    single_space_line = '1' + single_space_line[1:]

                f_text.write(single_space_line + "\n")

            # Xử lý và ghi các bảng (giữ nguyên như cũ)
            for ti, table in enumerate(valid_tables):
                raw = table.extract()
                if not raw or len(raw) <= 1:
                    continue
                cleaned = []
                prev = None
                for row in raw:
                    cells = [" ".join((c or "").replace("\n", " ").split()) for c in row]
                    non_empty = [j for j, c in enumerate(cells) if c]
                    if prev and len(non_empty) == 1 and all(not cells[k] for k in range(len(cells)) if k != non_empty[0]):
                        prev[non_empty[0]] += ' ' + cells[non_empty[0]]
                    else:
                        if prev:
                            cleaned.append(prev)
                        prev = cells
                if prev:
                    cleaned.append(prev)
                if len(cleaned) < 2 or all(len(r) == 1 for r in cleaned):
                    continue
                key = tuple(tuple(r) for r in cleaned)
                if key in seen_tables_global:
                    continue
                seen_tables_global.add(key)
                for r in cleaned:
                    r = [normalize_text(cell) for cell in r]
                    f_table.write("\t".join(r) + "\n")

    return output_txt_path, output_table_path