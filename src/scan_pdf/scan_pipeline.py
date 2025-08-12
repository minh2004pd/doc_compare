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
from concurrent.futures import ThreadPoolExecutor
from src.scan_pdf.table_scan_process import process_tables, build_table_cell_map_from_box_text
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

def should_skip_text(clean_text):
    check_text = normalize_text(clean_text)
    return (
        not clean_text or
        "tham dinh phap ly" in check_text or
        "tham dinh" in check_text or
        clean_text.upper() == "FF"
    )

def process_layout_file(image_path, output_image_path, table_crop_dir="./tables", layout_detector=None, page_number=1, table_map=None):
    """
    Xử lý file ảnh layout, phát hiện văn bản và lưu kết quả vào PDF.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
    
    image = cv2.imread(image_path)
    table_crop_paths = []

    # Phát hiện văn bản từ ảnh
    layout_results = layout_detector.predict(image_path, batch_size=1, layout_nms=True)
    table_idx = 1
    for res in layout_results:
        for block in res['boxes']:
            x1, y1, x2, y2 = map(int, block['coordinate'])
            # Xử lý từng block văn bản
            if block['label'] == 'table':
                crop_img = image[y1:y2, x1:x2]
                crop_path = os.path.join(table_crop_dir, f"page_{page_number+1}_table_{table_idx}.png")
                cv2.imwrite(crop_path, crop_img)
                table_crop_paths.append(crop_path)
                table_map[(page_number+1, table_idx)] = (crop_path, (x1, y1, x2, y2)) if table_map is not None else None
                table_idx += 1

            if block['label'] == "aside_text" or block['label'] == 'seal' or block['label'] == 'image' or block['label'] == 'signature' or block['label'] == 'stamp' or block['label'] == 'table':
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)  # bôi trắng

    cv2.imwrite(output_image_path, image)
    return output_image_path, table_crop_paths, table_map
    

def build_char_map_list(output_txt_path, box_to_text, page_count):
    glob_line_idx = 1
    char_map = {}
    logger.info(f"page_count: {page_count}")
    logger.info(f"📄 Tạo char_map từ {output_txt_path} với {len(box_to_text)} box và {page_count} trang...")
    with open(output_txt_path, "w", encoding="utf-8") as f_out:
        for i in range(page_count):
            # logger.info(f"📄 Trang {i+1}: {output_txt_path}")
            # Bạn cần lấy lại các box và text theo từng trang từ box_to_text
            page_number = i  # Giả sử bạn muốn lấy trang cuối cùng
            # Lọc các box thuộc trang này
            page_boxes = [(box, text) for (pg, box), text in box_to_text.items() if pg == page_number]
            # Sắp xếp lại nếu cần
            boxes = [box for box, text in page_boxes]
            texts = [text for box, text in page_boxes]
            groups = group_boxes_by_line(boxes, texts, line_threshold=30)
            line_idx = 1
            
            for group in groups:
                group_texts = [t[1] for t in group if t[1]]
                if group_texts:
                    # logger.info(f"group_texts: {group_texts}")
                    norm_line = " ".join(group_texts)
                    norm_line = normalize_text(norm_line)
                    if (
                        re.fullmatch(r"[0-9\s\W]+", norm_line.strip())  # chỉ số hoặc số kèm ký tự đặc biệt
                        or any(
                            (
                                (re.search(r"\d+[A-Za-zÀ-ỹ]", token) or re.search(r"[A-Za-zÀ-ỹ]\d+", token))
                                and 2 <= len(token) <= 5
                            )
                            for token in group_texts
                        )
                    ):
                        continue
                    # logger.info(f"page {page_number} - norm_line: {norm_line}")
                    # if norm_line:
                    # print(norm_line)
                    f_out.write(norm_line + "\n")
                    char_idx = 0
                    for word_idx, (box, word) in enumerate(group):
                        if not word:
                            continue
                        for char in word:
                            key = (glob_line_idx, char_idx)
                            char_map[key] = {
                                "page": page_number,
                                "max_line_idx": len(groups),
                                "char": char,
                                "word": word,
                                "box": box,
                                "word_idx": word_idx,
                                "line_idx": line_idx,
                                "glob_line_idx": glob_line_idx
                            }
                            char_idx += 1
                        if word_idx < len(group) - 1:
                            key = (glob_line_idx, char_idx)
                            char_map[key] = {
                                "page": page_number,
                                "max_line_idx": len(groups),
                                "char": " ",
                                "word": " ",
                                "box": box,  # Hoặc None nếu không muốn gán box
                                "word_idx": word_idx,
                                "line_idx": line_idx,
                                "glob_line_idx": glob_line_idx
                            }
                            char_idx += 1
                    
                    line_idx += 1
                    glob_line_idx += 1
        
    return char_map

def process_scan_file1(pdf_path, output_dir="./output", text_detector=None, detector=None, table_cells_detector=None, layout_detector=None, max_num_pages=1000):
    
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=250)
    image_output_dir = os.path.join(output_dir, f"pdf_images_{file_name}")
    os.makedirs(image_output_dir, exist_ok=True)

    image_paths, table_map, cell_box_to_text_map = [], {}, {}
    detection_results_all = []
    output_table_file = os.path.join(output_dir, f"tables_{file_name}.txt")
    output_crop_dir = os.path.join(output_dir, f"output_crops_{file_name}")
    os.makedirs(output_crop_dir, exist_ok=True)
    output_data_path = os.path.join(output_dir, f"{file_name}.txt")
    output_crop_info = os.path.join(output_dir, f"{file_name}_crop_info.csv")
    output_txt_path = os.path.join(output_dir, f"{file_name}_result_scan_text.txt")
    box_to_text, box_set = {}, set()

    # Chuyển PDF thành ảnh và xử lý layout, bảng
    for i, img in enumerate(images[:max_num_pages]):
        image_path = os.path.join(image_output_dir, f"{file_name}_page_{i+1}.png")
        img.save(image_path, "PNG")
        table_crop_dir = os.path.join(image_output_dir, f"tables_{file_name}_page_{i+1}")
        os.makedirs(table_crop_dir, exist_ok=True)
        image2_path, table_crop_paths, table_map = process_layout_file(
            image_path,
            os.path.join(image_output_dir, f"{file_name}_page_{i+1}_layout.png"),
            table_crop_dir=table_crop_dir,
            layout_detector=layout_detector,
            page_number=i,
            table_map=table_map
        )
        _, _, cell_box_to_text_map = process_tables(
            table_crop_paths, table_crop_dir,
            page_number=i, all_cells=[], cell_box_to_text_map=cell_box_to_text_map,
            table_cells_detector=table_cells_detector,
            text_detector=text_detector, detector=detector
        )
        detection_results = text_detector.predict(image2_path, batch_size=4)
        detection_results_all.append(detection_results)
        image_paths.append(image2_path)

    # Xử lý text từ các box đã phát hiện
    all_crop_info, all_data, all_txt_lines = [], [], []
    for i, image_path in enumerate(image_paths):
        logger.info(f"📄 Trang {i+1}: {output_txt_path}")
        img = Image.open(image_path).convert("RGB")
        detection_results = detection_results_all[i]
        dt_polys, dt_scores = [], []
        for page_result in detection_results:
            polys = page_result.get("dt_polys", [])
            scores = page_result.get("dt_scores", [])
            for poly, score in zip(polys, scores):
                pts = [[int(pt[0]), int(pt[1])] for pt in poly]
                dt_polys.append(pts)
                dt_scores.append(score)
        if not dt_polys:
            logger.info("  → Không phát hiện văn bản.")
            continue

        # Gom crops theo batch để OCR nhanh hơn
        crops, crop_boxes, crop_scores = [], [], []
        for j, (box, score) in enumerate(zip(dt_polys, dt_scores)):
            crop = crop_box_from_image(img, box)
            is_color, mask, ratio, _, _ = is_stamp_or_signature(crop)
            if is_color:
                continue
            crops.append(crop)
            crop_boxes.append(box)
            crop_scores.append(score)

        if crops:
            preds = detector.predict_batch(crops)  # dùng batch predict
            for j, (clean_text, box, score) in enumerate(zip(preds, crop_boxes, crop_scores)):
                clean_text = clean_text.strip()
                if should_skip_text(clean_text):
                    continue
                safe_text = re.sub(r'[\\/]', '_', clean_text.replace('\n', ' ').replace('\r', ' '))
                crop_filename = f"crop_{file_name}_{safe_text}.png"
                crop_path = os.path.join(output_crop_dir, crop_filename)
                crops[j].save(crop_path)

                all_crop_info.append(f"{i+1},{j+1},{crop_filename},{score:.4f},{safe_text}\n")
                all_data.append(f"{crop_path}\t{safe_text}\n")
                all_txt_lines.append(safe_text)

                box_key = (i, tuple(map(tuple, box)))
                if box_key not in box_set and safe_text.strip():
                    box_set.add(box_key)
                    box_to_text[box_key] = safe_text

        output_img_path = os.path.join(output_dir, f"{file_name}_page_{i+1}_with_boxes.png")
        draw_polygons_on_image(image_path, dt_polys, color=(0, 255, 0), thickness=2, output_path=output_img_path)
        logger.info(f"Đã lưu ảnh với box: {output_img_path}")

    # Ghi file một lần cuối
    with open(output_crop_info, "w", encoding="utf-8") as f_crop_info:
        f_crop_info.writelines(all_crop_info)
    with open(output_data_path, "w", encoding="utf-8") as f_data:
        f_data.writelines(all_data)
    with open(output_txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("\n".join(all_txt_lines))

    return output_txt_path, box_to_text, len(image_paths), cell_box_to_text_map, table_map

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

    # paddle ocr text detection
    text_detector = TextDetection(model_name="PP-OCRv5_server_det")
    model_type = "transformer"  # Hoặc "crnn" nếu bạn muốn dùng mô hình CRNN
    config = Cfg.load_config_from_name(f'vgg_{model_type}')
    config['weights'] = f'./weights/{model_type}ocr_v2.pth'  # Đảm bảo bạn đã tải trọng số phù hợp
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = Predictor(config)
    layout_detector = LayoutDetection(model_name="PP-DocLayout_plus-L")
    table_cells_detector = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det")

    text_to_boxes = {}
    pdf_path = "./doc_test.pdf"
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