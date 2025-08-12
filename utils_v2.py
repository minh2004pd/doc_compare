import os
import shutil
import tempfile
from pdf2image import convert_from_path
import cv2
import numpy as np
import uuid
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import time
from collections import defaultdict
from src.utils import *
from src.scan_pdf.scan_pipeline import build_char_map_list, run_scan_pdf_pipeline
from src.scan_pdf.table_scan_process import process_tables, build_table_cell_map_from_box_text
from src.compare_pipeline import doc_compare, run_compare_pipeline
import psutil
from PIL import Image, ImageDraw

def concat_images(img1, img2):
    # Resize về cùng chiều cao
    h = max(img1.shape[0], img2.shape[0])
    w1 = img1.shape[1]
    w2 = img2.shape[1]
    if img1.shape[0] != h:
        img1 = cv2.copyMakeBorder(img1, 0, h-img1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    if img2.shape[0] != h:
        img2 = cv2.copyMakeBorder(img2, 0, h-img2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    return cv2.hconcat([img1, img2])

def mark_diff_boxes_on_image(pdf_paths, diff_boxes_list, color=(0,0,255)):
    """
    pdf_paths: list đường dẫn PDF hoặc ảnh (mỗi phần tử là 1 trang)
    diff_boxes_list: list các list box, mỗi phần tử là list box của trang tương ứng
    Trả về: list ảnh đã vẽ box (dạng numpy array)
    """
    marked_images = []
    for pdf_path, diff_boxes in zip(pdf_paths, diff_boxes_list):
        ext = os.path.splitext(pdf_path)[1].lower()
        if ext == ".pdf":
            images = convert_from_path(pdf_path, dpi=250, first_page=1, last_page=1)
            if not images:
                raise ValueError(f"Không thể chuyển PDF {pdf_path} sang ảnh.")
            img_pil = images[0]
        elif ext in [".jpg", ".jpeg", ".png"]:
            img_pil = Image.open(pdf_path).convert("RGB")
        else:
            raise ValueError(f"Định dạng không hỗ trợ: {pdf_path}")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        for box in diff_boxes:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
        marked_images.append(img)
    return marked_images

def double_check_and_draw_cells(
    cell_map,
    cell_box_to_text_map,
    table_map,
    diff_table_path,
    txt_table_path1,
    txt_table_path2,
    img_paths,
    out_dir="./out1",
    file_name="file1"
):
    """
    Double check các cell khác biệt bằng OCR GPT-4, cập nhật lại bảng và vẽ box lên ảnh.
    """
    results = extract_boxes_from_diff_v2(diff_table_path, cell_map, label="[-]", table=True)
    if not results:
        print("Không có cell khác biệt trong file.")
        return cell_map, cell_box_to_text_map

    print(f"Số cell khác biệt file: {len(results)}")
    # Gom thông tin
    pairs = [(d['box'], d['page'], d['table_path'], d['table_box'], d['table_idx']) for d in results]
    boxes, page_idxs, table_paths, table_boxs, table_idxs = zip(*pairs) if pairs else ([], [], [], [], [])

    img_pages = [Image.open(p).convert("RGB") for p in img_paths]
    table_boxes_map = defaultdict(list)
    for box, page_idx, tbl_path, tbl_box, tbl_idx in zip(boxes, page_idxs, table_paths, table_boxs, table_idxs):
        key = (page_idx, tbl_path, tbl_idx, tuple(tbl_box))
        table_boxes_map[key].append(box)

    # Double check OCR cho từng cell
    for (page_idx, tbl_path, tbl_idx, tbl_box), box_list in table_boxes_map.items():
        tbl_img = Image.open(tbl_path).convert("RGB")
        box_set = set()
        for box in box_list:
            if box in box_set:
                continue
            box_set.add(box)
            box_pts = (
                [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
                if isinstance(box[0], (int, float)) else box
            )
            crop = crop_box_from_image(tbl_img, box_pts)
            new_text = normalize_text(ocr_image_with_gpt4(crop))
            k = (page_idx, tbl_idx, box)
            cell_box_to_text_map[k] = new_text

    # Cập nhật lại bảng sau double check
    cell_map, txt_table_path1 = build_table_cell_map_from_box_text(
        cell_box_to_text_map, table_map, output_dir=out_dir, file_name=file_name
    )
    doc_compare(txt_table_path1, txt_table_path2, diff_table_path, label="[-]", double_check=False)
    doc_compare(txt_table_path2, txt_table_path1, diff_table_path.replace("1", "2"), label="[-]", double_check=False)

    # Vẽ lại box lên bảng và paste vào ảnh trang gốc
    results = extract_boxes_from_diff_v2(diff_table_path, cell_map, label="[-]", table=True)
    pairs = [(d['box'], d['page'], d['table_path'], d['table_box']) for d in results]
    boxes, page_idxs, table_paths, table_boxs = zip(*pairs) if pairs else ([], [], [], [])
    new_table_boxes_map = defaultdict(list)
    for box, page_idx, tbl_path, tbl_box in zip(boxes, page_idxs, table_paths, table_boxs):
        key = (page_idx, tbl_path, tuple(tbl_box))
        new_table_boxes_map[key].append(box)

    for (page_idx, tbl_path, tbl_box), box_list in new_table_boxes_map.items():
        tbl_img = Image.open(tbl_path).convert("RGB")
        draw = ImageDraw.Draw(tbl_img)
        for box in box_list:
            if isinstance(box[0], (list, tuple)):
                draw.polygon([tuple(pt) for pt in box], outline="red", width=5)
            else:
                draw.rectangle(box, outline="red", width=5)
        x1, y1, x2, y2 = tbl_box
        tbl_img_resized = tbl_img.resize((x2-x1, y2-y1))
        img_pages[page_idx-1].paste(tbl_img_resized, (x1, y1))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"page_{page_idx}_with_tablebox.png")
        img_pages[page_idx-1].save(out_path)
        img_pages[page_idx-1].save(img_paths[page_idx-1])

    return cell_map, cell_box_to_text_map

def double_check_and_draw(
    img_paths,
    char_box_mappings,
    diff_path,
    box_to_text,
    txt_path,
    txt_path_other,
    page_count,
    color=(0,0,255),
    ocr_func=ocr_image_with_gpt4
):
    """
    Đánh dấu box khác biệt lên ảnh, double check OCR nếu cần, cập nhật lại diff.
    Trả về list ảnh đã đánh dấu.
    """
    if not char_box_mappings:
        return [cv2.imread(img_path) for img_path in img_paths]

    results = extract_boxes_from_diff_v2(diff_path, char_box_mappings, label="[-]")
    if not results:
        return [cv2.imread(img_path) for img_path in img_paths]

    boxes, page_idxs = zip(*[(d['box'], d['page']) for d in results])
    boxes_by_page = defaultdict(list)
    for box, page_idx in zip(boxes, page_idxs):
        boxes_by_page[page_idx].append(box)
    diff_boxes_list = [boxes_by_page[i] for i in range(len(img_paths))]
    imgs = mark_diff_boxes_on_image(img_paths, diff_boxes_list, color=color)

    double_check = False
    for i in range(len(img_paths)):
        img_pil = Image.open(img_paths[i]).convert("RGB")
        page_boxes = [box for box, page_idx in zip(boxes, page_idxs) if page_idx == i]
        if page_boxes:
            double_check = True
            box_set = set()
            for box in page_boxes:
                box_key = (i, tuple(map(tuple, box)))
                if box_key in box_to_text and box_key not in box_set:
                    crop = crop_box_from_image(img_pil, box)
                    new_text = ocr_func(crop)
                    new_text = normalize_text(new_text).replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')
                    box_to_text[box_key] = new_text
                    box_set.add(box_key)

    if double_check:
        char_box_mappings_new = build_char_map_list(txt_path, box_to_text, page_count)
        doc_compare(txt_path, txt_path_other, diff_path, label="[-]", all=True, double_check=False)
        doc_compare(txt_path_other, txt_path, diff_path.replace("1", "2"), label="[-]", all=True, double_check=False)
        new_results = extract_boxes_from_diff_v2(diff_path, char_box_mappings_new, label="[-]")
        pairs = [(d['box'], d['page']) for d in new_results]
        if pairs:
            boxes, page_idxs = zip(*pairs)
        else:
            boxes, page_idxs = [], []
        boxes_by_page = defaultdict(list)
        for box, page_idx in zip(boxes, page_idxs):
            boxes_by_page[page_idx].append(box)
        diff_boxes_list = [boxes_by_page[i] for i in range(len(img_paths))]
        imgs = mark_diff_boxes_on_image(img_paths, diff_boxes_list, color=color)
    return imgs