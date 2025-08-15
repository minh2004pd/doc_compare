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
from src.modules.scan_process import process_scan_file1, build_char_map_list
from src.modules.scan_table_process import build_table_cell_map_from_box_text

def setup_logger(log_path="./logs/scan_pdf.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("scan_pdf_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
    return logger

logger = setup_logger()

def run_scan_pdf_pipeline(pdf_path, output_dir="./output", max_num_pages=1000, text_detector=None, detector=None, layout_detector=None, table_cells_detector=None):
    """
    Pipeline xử lý file scan PDF:
    - Chuyển PDF sang ảnh
    - Phát hiện layout, crop bảng
    - Nhận diện text bằng OCR batch
    - Trích xuất bảng
    - Sinh char map
    - Lưu kết quả
    """

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info(f"🚀 Bắt đầu xử lý file scan PDF: {pdf_path}")

    t0 = time.time()
    # Bước 1: Xử lý scan, nhận diện text, bảng
    txt_path, box_to_text, page_count, cell_box_to_text_map, table_map = process_scan_file1(
        pdf_path,
        output_dir=output_dir,
        text_detector=text_detector,
        detector=detector,
        layout_detector=layout_detector,
        table_cells_detector=table_cells_detector,
        max_num_pages=max_num_pages
    )

    # Bước 2: Trích xuất bảng từ cell_box_to_text_map
    cell_map, txt_table_path = build_table_cell_map_from_box_text(
        cell_box_to_text_map, table_map, output_dir=output_dir, file_name=file_name
    )

    # Bước 3: Sinh char map cho text
    char_box_mappings = build_char_map_list(txt_path, box_to_text, page_count)
    logger.info(f"Số ký tự đã ánh xạ: {len(char_box_mappings)}")

    t1 = time.time()
    logger.info(f"✅ Hoàn thành xử lý file scan PDF trong {t1 - t0:.2f} giây")
    logger.info(f"Text file: {txt_path}")
    logger.info(f"Table file: {txt_table_path}")

    return {
        "text_path": txt_path,
        "text_table_path": txt_table_path,
        "char_box_mappings": char_box_mappings,
        "cell_map": cell_map,
        "cell_box_to_text_map": cell_box_to_text_map,
        "box_to_text": box_to_text,
        "table_map": table_map,
    }

def main():

    text_detector = TextDetection(model_name="PP-OCRv5_server_det", device="gpu:0")
    model_type = "transformer"  # Hoặc "crnn" nếu bạn muốn dùng mô hình CRNN
    config = Cfg.load_config_from_name(f'vgg_{model_type}')
    config['weights'] = f'./weights/{model_type}ocr_v2.pth'  # Đảm bảo bạn đã tải trọng số phù hợp
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = Predictor(config)
    layout_detector = LayoutDetection(model_name="PP-DocLayout_plus-L", device="gpu:0")
    table_cells_detector = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det", device="gpu:0")

    text_to_boxes = {}
    pdf_path = "./doc.pdf"
    output_dir = "./out"

    result = run_scan_pdf_pipeline(
        pdf_path,
        output_dir=output_dir,
        max_num_pages=1000,
        text_detector=text_detector,
        detector=detector,
        layout_detector=layout_detector,
        table_cells_detector=table_cells_detector
    )

    # output_txt_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_result_scan_norm_text.txt")

    # out = build_char_map_list(output_txt_path, result["box_to_text"], page_count=8)
# 
if __name__ == "__main__":
    main()