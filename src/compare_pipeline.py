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
from src.modules.compare_process import compare_texts, process_pdf_files

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

def run_compare_pipeline(pdf_file_1, pdf_file_2, max_num_pages=1000, output_dir="./output",
                        detector=None, text_detector=None, layout_detector=None, table_cells_detector=None):
    logger.info(f"📄 So sánh hai file PDF: {pdf_file_1} và {pdf_file_2}")
    os.makedirs(output_dir, exist_ok=True)

    result1 = process_pdf_files(pdf_file_1, output_dir=output_dir, max_num_pages=max_num_pages,
                                detector=detector,
                                text_detector=text_detector,
                                layout_detector=layout_detector,
                                table_cells_detector=table_cells_detector)
    result2 = process_pdf_files(pdf_file_2, output_dir=output_dir, max_num_pages=max_num_pages,
                                detector=detector,
                                text_detector=text_detector,
                                layout_detector=layout_detector,
                                table_cells_detector=table_cells_detector)

    logger.info(f"✅ Đã xử lý xong hai file PDF.")

    # So sánh text
    txt1_path = result1["text_path"]
    txt2_path = result2["text_path"]
    text_table1_path = result1["text_table_path"]
    text_table2_path = result2["text_table_path"]

    # So sánh văn bản
    diff1, diff2, diff_table1, diff_table2 = compare_texts(txt1_path, txt2_path, text_table1_path, text_table2_path, output_dir)

    return {
        "result1": result1,
        "result2": result2,
        "diff1": diff1,
        "diff2": diff2,
        "diff_table1": diff_table1,
        "diff_table2": diff_table2
    }
    
