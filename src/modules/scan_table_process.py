import os
import time
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import torch
import string
from paddleocr import TextDetection, TableCellsDetection
from unidecode import unidecode
from src.utils import (
    group_boxes_by_line, is_scanned_pdf, is_stamp_or_signature, extract_boxes_from_diff,
    extract_boxes_from_diff_v2,
    draw_boxes_on_image,
    ocr_from_image,
    ocr_image_with_gpt4,
    draw_polygons_on_image,
    normalize_line,
    crop_box_from_image,
    normalize_text
)
import pdfplumber
import fitz
import diff_match_patch as dmp_module
import re
from collections import defaultdict
from PIL import Image
import torch
from transformers import pipeline
import string
import base64
import uuid
import fitz
from unidecode import unidecode

import io

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

# Gom thành dòng/cột
def x_center(c): return (c['box'][0] + c['box'][2]) / 2
def y_center(c): return (c['box'][1] + c['box'][3]) / 2

def is_bbox_inside(inner, outer):
    # inner, outer: (x0, y0, x1, y1)
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3] 

def build_table_cell_map_from_box_text(cell_box_to_text_map, table_map, output_dir="./output", file_name="file1"):
    """
    Gom cell thành bảng thực sự từ cell_box_to_text_map cho nhiều trang, ghi bảng ra file.
    Trả về cell_map, output_table_file.
    """

    # Gom cell theo (page, table_idx)
    table_cells = defaultdict(list)
    for (pg, tbl_idx, box), text in cell_box_to_text_map.items():
        table_cells[(pg, tbl_idx)].append({'box': box, 'text': text})

    cell_map = {}
    output_table_file = os.path.join(output_dir, f"tables_{file_name}.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_table_file, "w", encoding="utf-8") as f_table:
        glob_line_idx = 1
        for (page, table_idx) in sorted(table_cells):
            local_cells = table_cells[(page, table_idx)]

            cells_sorted = sorted(local_cells, key=y_center)
            rows = []
            thresh_y = 30
            for c in cells_sorted:
                yc = y_center(c)
                for r in rows:
                    if abs(yc - y_center(r[0])) < thresh_y:
                        r.append(c)
                        break
                else:
                    rows.append([c])
            for r in rows:
                r.sort(key=x_center)

            # Viết từng row, và dùng thẳng cell object
            for row_cells in rows:
                texts = [normalize_line(c['text']) for c in row_cells]
                f_table.write(" ".join(texts) + "\n")

                char_idx = 0
                for col_idx, cell_obj in enumerate(row_cells):
                    cell_text = normalize_line(cell_obj['text'])
                    for ch in cell_text:
                        
                        key = (glob_line_idx, char_idx)
                        cell_map[key] = {
                            'table_path': table_map.get((page, table_idx), (None, None))[0],
                            'table_box':  table_map.get((page, table_idx), (None, None))[1],
                            'page':       page,
                            'table_idx':  table_idx,
                            'col_idx':    col_idx,
                            'char':       ch,
                            'box':        cell_obj['box'],
                        }
                        char_idx += 1

                    # thêm tab giữa các cell (nếu cần), chỉ tăng index
                    if col_idx < len(row_cells) - 1:
                        char_idx += 1

                glob_line_idx += 1

    return cell_map, output_table_file

def process_tables(table_paths, table_crop_dir, page_number=1, all_cells=None, cell_box_to_text_map=None, text_detector=None, table_cells_detector=None, detector=None):
    """
    Nhận vào list các table_paths, xử lý từng bảng, lưu toàn bộ bảng vào 1 file txt.
    """
    all_tables=[]
    for table_idx, table_path in enumerate(table_paths):
        image_table = cv2.imread(table_path)
        output = table_cells_detector.predict(table_path, batch_size=1)
        local_cells = []

        for res_idx, res in enumerate(output):
            res.save_to_img(os.path.join(table_crop_dir, f"table_{table_idx+1}_{res_idx+1}.png"))
            res.save_to_json(os.path.join(table_crop_dir, f"table_{table_idx+1}_{res_idx+1}.json"))

            for i, cell in enumerate(res['boxes']):
                x1, y1, x2, y2 = map(int, cell['coordinate'])
                crop_img = image_table[y1:y2, x1:x2]
                crop_path = os.path.join(table_crop_dir, f"cell_{table_idx+1}_{res_idx+1}_{i+1}.png")
                cv2.imwrite(crop_path, crop_img)

                detection_results = text_detector.predict(crop_path, batch_size=4)
                img = Image.open(crop_path).convert("RGB")

                segments = []
                # 1️⃣ Gom tất cả patch vào list
                patches = []
                coords = []  # lưu (min_y, min_x) để sau trả về
                for page in detection_results:
                    for poly, score in zip(page.get("dt_polys", []), page.get("dt_scores", [])):
                        pts = np.array([[int(pt[0]), int(pt[1])] for pt in poly], np.int32)
                        rect = order_points(pts)

                        maxW = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
                        maxH = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))

                        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], np.float32)
                        M = cv2.getPerspectiveTransform(rect, dst)
                        warp = cv2.warpPerspective(np.array(img), M, (maxW, maxH))

                        patch = Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))

                        patches.append(patch)
                        ys, xs = pts[:, 1], pts[:, 0]
                        coords.append((min(ys), min(xs)))
                
                # 2️⃣ OCR batch một lần
                if patches:
                    texts = detector.predict_batch(patches)  # giữ nguyên thứ tự
                    for text, (min_y, min_x) in zip(texts, coords):
                        text = normalize_text(text)
                        if not text:
                            text = '!'
                        text = text.replace('\n', ' ').replace('\r', ' ').strip()
                        segments.append((text, min_y, min_x))

                if segments:
                    segments.sort(key=lambda s: (s[1], s[2]))
                    full_text = " ".join([s[0] for s in segments]).strip()
                else:
                    full_text = ""

                # Chuẩn hóa tên file từ full_text
                safe_text = re.sub(r'[^a-zA-Z0-9_\- ]', '', full_text).strip().replace(' ', '_')
                if not safe_text:
                    safe_text = "empty"
                crop_save_path = os.path.join(table_crop_dir, f"cell_{page_number+1}_{table_idx+1}_{i+1}_{safe_text}.png")
                cv2.imwrite(crop_save_path, crop_img)
                
                cell_box_to_text_map[(page_number+1, table_idx+1, (x1, y1, x2, y2))] = full_text if cell_box_to_text_map is not None else None

                all_cells.append({
                    'page': page_number + 1,
                    'table_idx': table_idx + 1,
                    'box': (x1, y1, x2, y2),
                    'text': full_text
                })
                local_cells.append({
                    'box': (x1, y1, x2, y2),
                    'text': full_text
                })
        
        # Build bảng từ all_cells
        cells_sorted = sorted(local_cells, key=y_center)
        rows = []; thresh_y = 30
        for c in cells_sorted:
            yc = y_center(c); added=False
            for r in rows:
                if abs(yc - y_center(r[0])) < thresh_y:
                    r.append(c); added=True; break
            if not added:
                rows.append([c])

        for r in rows:
            r.sort(key=x_center)

        max_cols = max(len(r) for r in rows) if rows else 0
        table = [[c['text'] for c in r] + [""]*(max_cols-len(r)) for r in rows]
        all_tables.append(table)

    return all_tables, all_cells, cell_box_to_text_map

