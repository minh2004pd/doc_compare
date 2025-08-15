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
from paddleocr import TextDetection, LayoutDetection, TextImageUnwarping, TableCellsDetection
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
import torch
from collections import defaultdict
from PIL import Image
import string
import base64
import uuid
import fitz
import io
import logging
from src.scan_pipeline import run_scan_pdf_pipeline
from src.doc_pipeline import run_doc_pdf_pipeline

def setup_logger(log_path="./logs/compare_pdf.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("compare_pdf_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
    return logger

logger = setup_logger()

def doc_compare(text_file1, text_file2, output_diff_file, label="[-]", all=False, double_check = True):
    """
    So sánh hai file văn bản đã xử lý từ PDF.
    Trả về True nếu nội dung giống nhau, False nếu khác.
    """
    # Đọc nội dung hai file
    with open(text_file1, "r", encoding="utf-8") as f1:
        text1 = f1.read()
    with open(text_file2, "r", encoding="utf-8") as f2:
        text2 = f2.read()
    
    def normalize_lines(text):
        # Chuẩn hóa từng dòng
        return "\n".join([normalize_text(line) for line in text.splitlines() if line.strip()])
    
    if all:
        # Nếu so sánh tất cả, không chuẩn hóa gì cả
        norm_text1 = normalize_lines(text1)
        norm_text2 = normalize_lines(text2)
    else:
        if any(c.isupper() for c in text1):
            norm_text1 = normalize_text(text1)
        else:
            norm_text1 = text1
        
        if any(c.isupper() for c in text2):
            norm_text2 = normalize_text(text2)
        else:
            norm_text2 = text2
    
    norm_text1 = "\n".join([normalize_line(line) for line in norm_text1.splitlines()])
    norm_text2 = "\n".join([normalize_line(line) for line in norm_text2.splitlines()])

        # Lưu lại text đã chuẩn hóa để so sánh/debug
    norm1_path = text_file1.replace(".txt", "_norm.txt")
    norm2_path = text_file2.replace(".txt", "_norm.txt")
    with open(norm1_path, "w", encoding="utf-8") as f:
        f.write(norm_text1)
    with open(norm2_path, "w", encoding="utf-8") as f:
        f.write(norm_text2)

    # Khởi tạo diff_match_patch
    dmp = dmp_module.diff_match_patch()
    diff = dmp.diff_main(norm_text1, norm_text2)
    dmp.diff_cleanupSemantic(diff)  # Làm đẹp diff để dễ xem

    # In kết quả từng đoạn, giữ nguyên xuống dòng
    with open(output_diff_file, "w", encoding="utf-8") as f_out:
        for op, data in diff:
            if op == 0:
                f_out.write(data)
            elif op == -1:
                if data.strip() == "":
                    f_out.write(data)  # giữ nguyên tab/cách xuống dòng nếu chỉ là khoảng trắng
                else:
                    # Giữ nguyên xuống dòng trong đoạn khác biệt
                    lines = data.splitlines(keepends=True)
                    for line in lines:
                        if line.strip() == "":
                            f_out.write(line)
                        else:
                            f_out.write(f"[-]{line}[-]")
            elif op == 1 and double_check:
                if data.strip() == "":
                    continue
                else:
                    f_out.write(f"[+]")

def process_pdf_files(pdf_path, output_dir="./output", max_num_pages=1000,
                      text_detector=None, table_cells_detector=None, detector=None, layout_detector=None):

    scanned, _ = is_scanned_pdf(pdf_path)
    if not scanned:
        # Nếu không phải PDF quét, sử dụng pipeline scan
        result = run_scan_pdf_pipeline(pdf_path, output_dir=output_dir,
                                       max_num_pages=max_num_pages,
                                       text_detector=text_detector,
                                       table_cells_detector=table_cells_detector,
                                       detector=detector,
                                       layout_detector=layout_detector)

    else:
        result = run_doc_pdf_pipeline(pdf_path, output_dir=os.path.join(output_dir, "file1"))
    
    logger.info(f"✅ Đã xử lý xong file PDF.")

    return result

def compare_texts(txt1_path, txt2_path, txt_table_path1, txt_table_path2, output_dir):
    diff1 = os.path.join(output_dir, "diff1.txt")
    diff2 = os.path.join(output_dir, "diff2.txt")
    diff_table1 = os.path.join(output_dir, "diff_table1.txt")
    diff_table2 = os.path.join(output_dir, "diff_table2.txt")

    doc_compare(txt1_path, txt2_path, diff1, label="[-]")
    doc_compare(txt2_path, txt1_path, diff2, label="[-]")
    doc_compare(txt_table_path1, txt_table_path2, diff_table1, label="[-]")
    doc_compare(txt_table_path2, txt_table_path1, diff_table2, label="[-]")

    return diff1, diff2, diff_table1, diff_table2