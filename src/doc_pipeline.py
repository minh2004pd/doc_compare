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
from src.modules.doc_process import process_doc_file

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

def run_doc_pdf_pipeline(pdf_path, output_dir="./output"):
    """
    Pipeline xử lý file doc PDF:
    - Trích xuất text ngoài bảng
    - Trích xuất bảng
    - Lưu ra file kết quả
    """
    t0 = time.time()
    logger.info(f"🚀 Bắt đầu xử lý file doc PDF: {pdf_path}")
    txt_path, text_table_path = process_doc_file(pdf_path, output_dir=output_dir)
    logger.info(f"✅ Đã xử lý xong file doc PDF.")
    logger.info(f"Text file: {txt_path}")
    logger.info(f"Table file: {text_table_path}")
    t1 = time.time()
    logger.info(f"⏱️ Thời gian xử lý: {t1 - t0:.2f} giây")
    return {
        "text_path": txt_path,
        "text_table_path": text_table_path,
        "char_box_mappings": None,
        "cell_map": None,
        "cell_box_to_text_map": None,
        "box_to_text": None,
        "table_map": None
    }

if __name__ == "__main__":
    pdf_path = "./file.pdf"
    output_dir = "./out2"
    result = run_doc_pdf_pipeline(pdf_path, output_dir=output_dir)

