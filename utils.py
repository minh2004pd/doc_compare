import os
import cv2
import numpy as np
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import fitz  # PyMuPDF
import requests
import json
import base64
import time
from mistralai import Mistral
from dotenv import load_dotenv
import datauri
import os

load_dotenv()
mistral_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=mistral_key)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def ocr_from_image(image_path="./output1/pdf_images_doc/doc_page_1.png"):
    """
    Gửi ảnh base64 lên API Mistral OCR, trả về kết quả OCR.
    """

    # Encode ảnh local thành base64, xác định mime-type phù hợp
    from mimetypes import guess_type
    mime, _ = guess_type(image_path)
    if mime is None:
        mime = "image/png"  # fallback
    base64_image = encode_image_to_base64(image_path)
    data_url = f"data:{mime};base64,{base64_image}"

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": data_url,
        },
        include_image_base64=True
    )
    # Xử lý lấy text markdown từ tất cả các trang (hoặc chỉ trang đầu)
    if hasattr(ocr_response, "dict"):
        ocr_response = ocr_response.model_dump()
    if "pages" in ocr_response and len(ocr_response["pages"]) > 0:
        # Ghép text tất cả các trang (hoặc chỉ lấy trang đầu: ocr_response["pages"][0]["markdown"])
        text = "\n\n".join(page["markdown"] for page in ocr_response["pages"])
        return text
    return ""


def create_voc_xml(filename, width, height, boxes):
    ann = Element("annotation")
    SubElement(ann, "filename").text = filename
    SubElement(ann, "folder").text = ""
    source = SubElement(ann, "source")
    SubElement(source, "sourceImage").text = ""
    SubElement(source, "sourceAnnotation").text = "Datumaro"
    size = SubElement(ann, "imagesize")
    SubElement(size, "nrows").text = str(height)
    SubElement(size, "ncols").text = str(width)

    for idx, (xmin, ymin, xmax, ymax, text) in enumerate(boxes):
        obj = SubElement(ann, "object")
        SubElement(obj, "name").text = "tes"
        SubElement(obj, "deleted").text = "0"
        SubElement(obj, "verified").text = "0"
        SubElement(obj, "occluded").text = "no"
        SubElement(obj, "id").text = str(idx)
        SubElement(obj, "type").text = "bounding_box"
        poly = SubElement(obj, "polygon")
        for x, y in [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]:
            pt = SubElement(poly, "pt")
            SubElement(pt, "x").text = f"{x:.2f}"
            SubElement(pt, "y").text = f"{y:.2f}"
        SubElement(obj, "attributes").text = f"Content={text}, rotation=0.0"

    xml = parseString(tostring(ann)).toprettyxml(indent="  ")
    return xml

def group_boxes_by_line(dt_polys, rec_texts, line_threshold=15):
    """
    Gom các box và text thành dòng theo trung bình tung độ (y),
    sau đó sắp xếp các dòng theo thứ tự từ trên xuống dưới.
    Trả về: list các dòng, mỗi dòng là list tuple (box, text), dòng đầu tiên nằm trên cùng.
    """
    centers = []
    for box, text in zip(dt_polys, rec_texts):
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        y_center = np.mean(ys)
        x_left = min(xs)
        centers.append((box, text, y_center, x_left))

    # Sắp xếp các bounding boxes theo y_center
    centers.sort(key=lambda x: x[2])

    # Gom nhóm theo dòng
    lines = []
    for box, text, y_center, x_left in centers:
        assigned = False
        for line in lines:
            if abs(line['y'] - y_center) < line_threshold:
                line['boxes'].append((box, text, y_center, x_left))
                line['y'] = (line['y'] * (len(line['boxes']) - 1) + y_center) / len(line['boxes'])
                assigned = True
                break
        if not assigned:
            lines.append({'y': y_center, 'boxes': [(box, text, y_center, x_left)]})

    # Sắp xếp mỗi dòng theo trục X (từ trái sang phải)
    sorted_lines = []
    for line in lines:
        sorted_boxes = sorted(line['boxes'], key=lambda b: b[3])
        sorted_lines.append([(b[0], b[1]) for b in sorted_boxes])

    # Cuối cùng, sắp xếp lại danh sách các dòng theo trục Y (từ trên xuống dưới)
    sorted_lines = sorted(sorted_lines, key=lambda line:
        np.mean([pt[1] for pt in line[0][0]]))

    return sorted_lines

def map_chars_to_boxes(grouped_lines):
    """
    Đánh chỉ số cho từng ký tự theo (line_idx, char_idx),
    ánh xạ từng ký tự sang box tương ứng (thuộc về từ đó).
    """
    char_map = {}
    for line_idx, line in enumerate(grouped_lines):
        for word_idx, (box, word) in enumerate(line):
            # # print(f"Mapping word '{word}' at line {line_idx}, word index {word_idx}, box {box}")
            for char_idx, char in enumerate(word):
                if char.strip() == "":
                    continue  # bỏ khoảng trắng
                key = (line_idx, char_idx)
                char_map[key] = {
                    "char": char,
                    "word": word,
                    "box": box,
                    "word_idx": word_idx,
                    "line_idx": line_idx
                }
    return char_map

import re

def extract_boxes_from_diff_v2(diff_file_path, char_map, label="[-]", table=False):
    """
    Duyệt từng dòng, từng ký tự trong diff, chỉ trích ký tự nằm trong [-]...[-],
    và ánh xạ theo char_map (dựa trên (line_idx, char_idx)).
    """
    label1 = "[+]"
    with open(diff_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_label = False
    target_positions = []  # danh sách (line_idx, char_idx) nằm trong [-]...[-]
    glob_line_idx = 1  # dòng bắt đầu từ 1 (giống char_map)

    for line in lines:
        char_idx = 0
        glob_char_idx = 0
        # print(f"[DEBUG] Processing line {glob_line_idx}: {line.strip()}")
        while glob_char_idx < len(line):
            if line[glob_char_idx:glob_char_idx+len(label)] == label:
                in_label = not in_label
                glob_char_idx += len(label)
                continue

            if in_label:
                # if line[glob_char_idx] == '-':
                #     print(f"[DEBUG] Collecting '-' at line {glob_line_idx}, char_idx {char_idx}")
                target_positions.append((glob_line_idx, char_idx))
                # # print(f"Collecting '{line[char_idx]}' at line {line_idx}, char_idx {char_idx}")

            if line[glob_char_idx:glob_char_idx+len(label1)] == label1:
                glob_char_idx += len(label1)

                # Append ký tự bên trái (nếu có)
                if glob_char_idx > 0:
                    target_positions.append((glob_line_idx, char_idx-1))
                # Append ký tự bên phải (nếu có)
                if glob_char_idx < len(line) - len(label1):
                    target_positions.append((glob_line_idx, char_idx+1))
                continue
            
            char_idx += 1
            glob_char_idx += 1
        glob_line_idx += 1

    # Ánh xạ theo char_map
    result = []
    if not table:
        for pos in target_positions:
            if pos in char_map:
                info = char_map[pos]
                result.append({
                    "char": info["char"],
                    "box": info["box"],
                    "line_idx": info["line_idx"],
                    "word": info["word"],
                    "page": info["page"],
                })
                # print(f"[DEBUG] Found position {pos} in char_map, box {info['box']}, char '{info['char']}'")
            else:
                continue  # nếu không tìm thấy trong char_map, bỏ qua
                # print(f"Warning: {pos} not found in char_map")
    else:
        # Nếu là bảng, cần xử lý đặc biệt với các ký tự trong bảng
        for pos in target_positions:
            if pos in char_map:
                info = char_map[pos]
                result.append({
                    'table_path': info.get("table_path", ""),
                    'table_box': info.get("table_box", ()),
                    "box": info["box"],
                    "page": info["page"],
                    'table_idx': info.get("table_idx", -1),
                })
                print(f"[DEBUG] Found position {pos} in char_map, box {info['box']}")
            else:
                continue

    return result

def extract_boxes_from_diff(diff_file_path, char_map, label="[-]"):
    """
    Duyệt từng dòng, từng ký tự trong diff, chỉ trích ký tự nằm trong [-]...[-],
    và ánh xạ theo char_map (dựa trên (line_idx, char_idx)).
    """

    label1 = "[+]"
    with open(diff_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_label = False
    target_positions = []  # danh sách (line_idx, char_idx) nằm trong [-]...[-]
    line_idx = 1  # dòng bắt đầu từ 1 (giống char_map)

    for line in lines:
        char_idx = 0
        char_idx_glob = 0
        while char_idx_glob < len(line):
            if line[char_idx_glob:char_idx_glob+len(label)] == label:
                in_label = not in_label
                char_idx_glob += len(label)
                continue

            if in_label:
                target_positions.append((line_idx, char_idx))
                # # print(f"Collecting '{line[char_idx]}' at line {line_idx}, char_idx {char_idx}")
            
            if line[char_idx_glob:char_idx_glob+len(label1)] == label1:
                char_idx_glob += len(label1)
                target_positions.append((line_idx, char_idx))
                # Append ký tự bên trái (nếu có)
                if char_idx_glob > 0:
                    target_positions.append((line_idx, char_idx-1))
                # Append ký tự bên phải (nếu có)
                if char_idx_glob < len(line) - len(label1):
                    target_positions.append((line_idx, char_idx+1))
                continue
            char_idx += 1
            char_idx_glob += 1
        line_idx += 1

    # Ánh xạ theo char_map
    result = []
    for pos in target_positions:
        if pos in char_map:
            info = char_map[pos]
            result.append({
                "char": info["char"],
                "box": info["box"],
                "line_idx": info["line_idx"],
                "word": info["word"]
            })
        else:
            continue  # nếu không tìm thấy trong char_map, bỏ qua
            # print(f"Warning: {pos} not found in char_map")

    return result


def draw_boxes_on_image(image_path, boxes, output_path, color=(0, 0, 255), thickness=2):
    """
    Vẽ các polygon box lên ảnh và lưu ra file mới.
    """
    image = Image.open(image_path).convert("RGB")
    image_cv = np.array(image)
    for box in boxes:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_cv, [pts], isClosed=True, color=color, thickness=thickness)
    output_image = Image.fromarray(image_cv)
    output_image.save(output_path)

def get_strong_color_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [10, 255, 255]),    # Đỏ đậm
        ([160, 100, 100], [179, 255, 255]), # Đỏ cuối vòng HSV
        ([10, 100, 100], [25, 255, 255]),   # Hồng/Đỏ cam
        ([25, 100, 100], [35, 255, 255]),   # Cam
        ([90, 100, 100], [140, 255, 255])   # Xanh dương đến tím
    ]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return mask

def is_stamp_or_signature(img_pil, red_ratio_thresh=0.000001, area_thresh=1):
    """Xác định nếu ảnh chứa vùng đỏ đáng kể → con dấu/chữ ký."""
    img = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask = get_strong_color_mask(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_area = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > area_thresh)
    total = mask.shape[0] * mask.shape[1]
    ratio = red_area / total
    return ratio >= red_ratio_thresh, mask, ratio, red_area, total

def is_scanned_pdf(path, text_threshold=0.1):
    doc = fitz.open(path)
    total_text_area = 0.0
    total_area = 0.0

    for page in doc:
        rect = page.rect
        total_area += rect.width * rect.height
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:  # block chứa text
                for line in b["lines"]:
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        total_text_area += (x1 - x0) * (y1 - y0)

    text_ratio = total_text_area / total_area
    # print(f"Text ratio: {text_ratio:.4f} (threshold: {text_threshold})")
    return text_ratio > text_threshold, text_ratio

def main():
    # Xử lý tất cả ảnh input
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        pil = Image.open(os.path.join(input_folder, fname))
        ok, mask, ratio, red_area, total = is_stamp_or_signature(pil)

        if ok:
            pil.save(os.path.join(output_folder, "original", fname))
            Image.fromarray(mask).save(os.path.join(output_folder, "mask", fname))
            # # print(f"✔️ {fname}: Phát hiện dấu/chữ ký màu (đỏ) ‑ {ratio:.4%} vùng đỏ")
            # # print(f"red_area: {red_area}")
            # # print(f"total: {total}")
            # # print(f"ratio: {ratio}")
        else:
            # # print(f"❌ {fname}: Chỉ văn bản đen/xám (vùng đỏ chỉ {ratio:.4%})")
            # # print(f"red_area: {red_area}")
            # # print(f"total: {total}")
            # # print(f"ratio: {ratio}")
            continue

if __name__ == "__main__":
    input_path = "./output1/pdf_images_doc/doc_page_1.png"
    text = ocr_from_image(input_path)
    print(text)