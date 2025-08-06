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
from utils import (
    group_boxes_by_line, is_scanned_pdf, is_stamp_or_signature, extract_boxes_from_diff,
    extract_boxes_from_diff_v2,
    draw_boxes_on_image,
    ocr_from_image
)
import pdfplumber
import fitz
import diff_match_patch as dmp_module
import re
from collections import defaultdict
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
import string
import base64
import uuid
import fitz  # PyMuPDF

import io

load_dotenv()

def ocr_image_with_gpt4(image_pil, custom_prompt=None):
    """
    OCR ảnh bằng GPT-4 Vision với prompt tùy chỉnh
    Args:
        image_pil: PIL Image object (RGB format)
        custom_prompt: Custom prompt cho OCR
    """
    # Default prompt cho OCR
    default_prompt = """OCR CHÍNH XÁC 100%:
    - Đọc TỪNG KÝ TỰ như trong ảnh
    - KHÔNG sửa chính tả, KHÔNG đoán từ, không thêm từ, không bỏ từ
    - KHÔNG dịch nghĩa, không giải thích, không thêm chú thích
    - KHÔNG thêm/bớt dấu câu, không thêm/bớt khoảng trắng
    - KHÔNG thêm/bớt chữ hoa/thường
    - KHÔNG thêm/bớt dấu câu, không thêm/bớt khoảng trắng
    - KHÔNG thêm/bớt ký tự
    - KHÔNG sử dụng markdown format (không dùng ```, không dùng dấu nháy)
    - Giữ nguyên khoảng trắng, không xuống dòng, kết quả trả về trong cùng 1 dòng
    - Không có text → "!"
    - Tối đa 100 ký tự
    - CHỈ trả về text thuần, không format, không giải thích

    CHỈ TRẢ VỀ VĂN BẢN:"""
    
    prompt = custom_prompt if custom_prompt else default_prompt
    
    # Initialize OpenAI Client
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Function to encode PIL image
    def encode_pil_image(image_pil):
        import io
        buffered = io.BytesIO()
        # Đảm bảo ảnh ở định dạng RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        image_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    try:
        # Encode image
        base64_image = encode_pil_image(image_pil)
        
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,  # đủ cho 100 ký tự text trả về
            temperature=0.5,  # giữ kết quả ổn định, không ngẫu nhiên
            top_p=0.5,       # loại bỏ các token không chắc
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Lỗi OCR: {e}"

def normalize_line(line):
    # Loại bỏ khoảng trắng dư thừa giữa các trường, giữ lại tab nếu có
    return " ".join(line.strip().split())

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

MAX_LENGTH = 512

# Tạo bảng dịch để loại bỏ dấu câu
translator = str.maketrans('', '', string.punctuation)

# Danh sách các ký tự hoặc chuỗi đặc biệt muốn loại bỏ thêm
special_tokens = [
    "...", "…", ",-", "--", "–", "—", "―", "•", "·", "“", "”", "‘", "’", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "=", "+", "-", "*", "&", "^", "%", "$", "#", "@", "~", "`", "'", '"', "…", "…", "…", "…"
]

# paddle ocr text detection
text_detector = TextDetection(model_name="PP-OCRv5_server_det")
model_type = "transformer"  # Hoặc "crnn" nếu bạn muốn dùng mô hình CRNN
config = Cfg.load_config_from_name(f'vgg_{model_type}')
config['weights'] = f'./weights/{model_type}ocr_v2.pth'  # Đảm bảo bạn đã tải trọng số phù hợp
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = Predictor(config)
layout_detector = LayoutDetection(model_name="PP-DocLayout_plus-L")
unwarp_model = TextImageUnwarping(model_name="UVDoc")
table_cells_detector = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det")
text_to_boxes = {}

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def y_center(c): return (c['box'][1] + c['box'][3])//2
def x_center(c): return (c['box'][0] + c['box'][2])//2

def process_tables(table_paths, table_crop_dir, page_number=1, all_cells=None, cell_box_to_text_map=None):
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
                for page in detection_results:
                    for poly, score in zip(page.get("dt_polys",[]), page.get("dt_scores",[])):
                        pts = np.array([[int(pt[0]), int(pt[1])] for pt in poly], np.int32)
                        rect = order_points(pts)
                        maxW = int(max(np.linalg.norm(rect[2]-rect[3]), np.linalg.norm(rect[1]-rect[0])))
                        maxH = int(max(np.linalg.norm(rect[1]-rect[2]), np.linalg.norm(rect[0]-rect[3])))
                        dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]],np.float32)
                        M = cv2.getPerspectiveTransform(rect, dst)
                        warp = cv2.warpPerspective(np.array(img), M, (maxW, maxH))
                        patch = Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
                        text = detector.predict(patch).strip()
                        text = normalize_text(text)
                        if not text:
                            text = '!'
                            
                        text = text.replace('\n',' ').replace('\r',' ').strip()
                        ys, xs = pts[:,1], pts[:,0]
                        segments.append((text, min(ys), min(xs)))

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
                if full_text.strip() == '-':
                    print(f"[DEBUG] Found '-' in cell text on page {page_number+1}, table {table_idx+1}, box {(x1, y1, x2, y2)}")
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
            # Xử lý từng block văn bản
            if block['label'] == 'table':
                x1, y1, x2, y2 = map(int, block['coordinate'])
                crop_img = image[y1:y2, x1:x2]
                crop_path = os.path.join(table_crop_dir, f"page_{page_number+1}_table_{table_idx}.png")
                cv2.imwrite(crop_path, crop_img)
                print(f"Đã lưu crop bảng: {crop_path}")
                table_crop_paths.append(crop_path)
                table_map[(page_number+1, table_idx)] = (crop_path, (x1, y1, x2, y2)) if table_map is not None else None
                table_idx += 1

            if block['label'] == "aside_text" or block['label'] == 'seal' or block['label'] == 'image' or block['label'] == 'signature' or block['label'] == 'stamp' or block['label'] == 'table':
                x1, y1, x2, y2 = map(int, block['coordinate'])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)  # bôi trắng

    cv2.imwrite(output_image_path, image)
    return output_image_path, table_crop_paths, table_map

def draw_text_overlay(image_path, boxes_text, output_pdf_path):
    img = Image.open(image_path)
    w, h = img.size

    # Tạo canvas tạm
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(w, h))
    pdfmetrics.registerFont(TTFont('Arial', '/root/doc_compare/OCR-Vietnamese-Text-Generator/trdg/fonts/vi/SVN-Arial 3.ttf'))
    c.setFont("Arial", 10)

    for box, text in boxes_text:
        if not text.strip():
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        min_x, min_y = min(xs), min(ys)
        box_h = max(ys) - min_y
        c.drawString(min_x, h - min_y - box_h, text)

    c.save()
    packet.seek(0)

    # Merge ảnh gốc với lớp text
    bg_stream = io.BytesIO()
    img.convert("RGB").save(bg_stream, format="PDF")
    bg_stream.seek(0)

    bg_pdf = PdfReader(bg_stream)
    text_pdf = PdfReader(packet)
    writer = PdfWriter()
    page = bg_pdf.pages[0]
    page.merge_page(text_pdf.pages[0])
    writer.add_page(page)

    with open(output_pdf_path, "wb") as f_out:
        writer.write(f_out)


def draw_text_only_pdf(boxes_text, output_pdf_path, page_size, font_path=None):
    width, height = page_size
    c = canvas.Canvas(output_pdf_path, pagesize=page_size)

    # Font mặc định hoặc tùy chọn
    font_name = "Helvetica"
    if font_path:
        font_name = "CustomFont"
        pdfmetrics.registerFont(TTFont(font_name, font_path))

    for box, text in boxes_text:
        if not text.strip():
            continue

        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(xs), max(ys)

        box_w = max_x - min_x
        box_h = max_y - min_y

        # Bắt đầu với cỡ font cơ sở, scale theo box
        font_size = box_h * 0.8  # scale theo chiều cao

        # Đảm bảo text không dài hơn box
        text_width = pdfmetrics.stringWidth(text, font_name, font_size)
        if text_width > box_w:
            font_size *= (box_w / text_width)

        # Vị trí y cần chuyển đổi từ ảnh sang PDF (0,0 ở góc dưới)
        y_pdf = height - min_y - box_h

        c.setFont(font_name, font_size)
        c.drawString(min_x, y_pdf, text)

    c.save()
    print(f"✅ PDF đã tạo: {output_pdf_path}")

def draw_polygons_on_image(image_path, polygons, color=(0, 255, 0), thickness=2, output_path=None):
    img = cv2.imread(image_path)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    if output_path:
        cv2.imwrite(output_path, img)
    return img

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

            # Gom thành dòng/cột
            def x_center(c): return (c['box'][0] + c['box'][2]) / 2
            def y_center(c): return (c['box'][1] + c['box'][3]) / 2

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
                        # nếu cần debug dấu '-' thì vẫn in được
                        if ch == '-':
                            print(f" - char [{page},{glob_line_idx}, {char_idx}] {ch} → {cell_obj['box']}")
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

def process_scan_file1(pdf_path, output_dir="./output", text_detector=None, detector_model=None, max_num_pages=1000):
    t0 = time.time()
    
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    images = convert_from_path(pdf_path, dpi=250)
    image_output_dir = os.path.join(output_dir, f"pdf_images_{file_name}")
    os.makedirs(image_output_dir, exist_ok=True)

    # Chuyển PDF thành ảnh
    image_paths = []
    table_crop_paths = []
    all_cells = []
    table_map = {}
    cell_map = {}
    cell_box_to_text_map = {}  # (page, table_idx, (x1, y1, x2, y2)) -> text

    detection_results_all = []
    output_table_file = os.path.join(output_dir, f"tables_{file_name}.txt")
    with open(output_table_file, "w", encoding="utf-8") as f_table:
        glob_line_idx=1
        for i, img in enumerate(images):
            if i >= max_num_pages:
                print(f"Đã đạt giới hạn {max_num_pages} trang, dừng lại.")
                break
            image_path = os.path.join(image_output_dir, f"{file_name}_page_{i+1}.png")
            img.save(image_path, "PNG")
            table_crop_dir = os.path.join(image_output_dir, f"tables_{file_name}_page_{i+1}")
            os.makedirs(table_crop_dir, exist_ok=True)
            image2_path, table_crop_paths, table_map = process_layout_file(image_path,
                                            os.path.join(image_output_dir, f"{file_name}_page_{i+1}_layout.png"),
                                            table_crop_dir=table_crop_dir,
                                            layout_detector=layout_detector,
                                            page_number=i,
                                            table_map=table_map
                                            )
            
            all_tables, all_cells, cell_box_to_text_map = process_tables(table_crop_paths, table_crop_dir,
                                                    page_number=i, all_cells=all_cells, cell_box_to_text_map=cell_box_to_text_map)

            # if all_tables:
            #     for table_idx, table in enumerate(all_tables):
            #         for row in table:
            #             norm_row = [cell for cell in row]
            #             f_table.write(" ".join(norm_row) + "\n")
            #             # print(f"Trang {i+1} line {glob_line_idx}: {' | '.join(norm_row)}")
                    
            #             char_idx = 0
            #             for col_idx, cell_text in enumerate(norm_row):
                            
            #                 # Tìm cell object tương ứng (theo text, page, table_idx)
            #                 cell_obj = next(
            #                     (c for c in all_cells if c['text'] == cell_text and c['page'] == i+1 and c['table_idx'] == table_idx + 1),
            #                     None
            #                 )
            #                 if cell_obj:
            #                     for ch in cell_text:
            #                         key = (glob_line_idx, char_idx)
            #                         key_table = (i+1, table_idx+1)
            #                         # Gán ánh xạ từ (glob_line_idx, char_idx) → (page, table_idx, box)
            #                         cell_map[key] = {
            #                             'table_path': table_map.get(key_table, None)[0],
            #                             'table_box': table_map.get(key_table, None)[1],
            #                             'page': i + 1,
            #                             'table_idx': table_idx + 1,
            #                             'col_idx': col_idx,
            #                             'char': ch,
            #                             'box': cell_obj['box']
            #                         }
            #                         # print(f"  [{i+1},{glob_line_idx}] {ch} → {cell_obj['text']}, table path: {table_map.get(key_table, None)[0]}")
            #                         char_idx += 1
            #                     # Thêm ký tự tab giữa các cell (trừ cell cuối cùng)
            #                     if col_idx < len(norm_row) - 1:
            #                         key = (glob_line_idx, char_idx)
            #                         cell_map[key] = {
            #                             'table_path': table_map.get(key_table, None)[0],
            #                             'table_box': table_map.get(key_table, None)[1],
            #                             'page': i + 1,
            #                             'table_idx': table_idx + 1,
            #                             'col_idx': col_idx,
            #                             'char': ' ',
            #                             'box': cell_obj['box']
            #                         }
            #                         char_idx += 1
            #             glob_line_idx += 1

            detection_results = text_detector.predict(
                image2_path,
                batch_size=4
            )

            detection_results_all.append(detection_results)
            image_paths.append(image2_path)

    # Tạo thư mục lưu crop
    output_crop_dir = os.path.join(output_dir, f"output_crops_{file_name}")
    os.makedirs(output_crop_dir, exist_ok=True)

    char_map = {}
    output_data_path = os.path.join(output_dir, f"{file_name}.txt")
    output_crop_info = os.path.join(output_dir, f"{file_name}_crop_info.csv")
    output_txt_path = os.path.join(output_dir, f"{file_name}_result_scan_text.txt")
    glob_line_idx = 1
    box_to_text = {}  # Ánh xạ box (tuple) → word/text
    box_set = set()  # Set để lưu các box đã xử lý

    for i, image_path in enumerate(image_paths):
        if i >= max_num_pages:
            print(f"Đã đạt giới hạn {max_num_pages} trang, dừng lại.")
            break

        print(f"📄 Trang {i+1}: {output_txt_path}")
        with open(output_crop_info, "w", encoding="utf-8") as f_crop_info, \
            open(output_data_path, "w", encoding="utf-8") as f_data:
            
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            print(f"\n📄 Trang {i+1} ({file_name}):")

            dt_polys = []
            dt_scores = []
            detection_results = detection_results_all[i]
            for page_result in detection_results:
                polys = page_result.get("dt_polys", [])
                scores = page_result.get("dt_scores", [])
                for poly, score in zip(polys, scores):
                    pts = [[int(pt[0]), int(pt[1])] for pt in poly]
                    dt_polys.append(pts)
                    dt_scores.append(score)
            if not dt_polys:
                print("  → Không phát hiện văn bản.")
                continue
            
            # if len(table_crop_paths) > 0:
            #     print(f"  → Phát hiện {len(table_crop_paths)} bảng trong ảnh.")
            #     for j, crop_path in enumerate(table_crop_paths):
            #         print(f"  {j+1}. Crop bảng: {crop_path}")
            #         table_txt_path = extract_table_from_scan(crop_path, cell_detector, text_detector, detector, output_dir)
            #         f_data.write(f"{table_txt_path}\n")
            clean_texts = []
            new_dt_polys = []
            new_dt_scores = []
            for j, (box, score) in enumerate(zip(dt_polys, dt_scores)):
                crop = crop_box_from_image(img, box)

                is_color, mask, ratio, _, _ = is_stamp_or_signature(crop)
                if is_color:
                    # print(f"  [{i+1},{j+1}] 🚫 Bỏ vì màu đậm tỉ lệ {ratio:.1%}")
                    continue

                clean_text = detector.predict(crop)
                check_text = normalize_text(clean_text)
                if "tham dinh phap ly" in check_text or "tham dinh" in check_text:
                    print(f"  [{i+1},{j+1}] 🚫 Bỏ qua box chứa 'đã thẩm định bởi'")
                    continue
                # print(f"  [{i+1},{j+1}] Text: {clean_text}")
                # clean_text = ocr_image_with_gpt4(crop)
                # clean_text = ocr_from_image(crop_path)
                # time.sleep(1)  # Thêm delay 1 giây giữa các lần gọi API để tránh bị giới hạn
                clean_text = clean_text.strip()
                if not clean_text:
                    clean_text = ";"  # Nếu không có text, gán là "!"
                    print(f"  [{i+1},{j+1}] 🚫 Không có text, gán là ';'")

                # Thay ký tự / hoặc \ bằng _
                safe_text_before = re.sub(r'[\\/]', '_', clean_text)

                # Định nghĩa crop_filename và các đường dẫn
                crop_filename = f"crop_{file_name}_{safe_text_before}.png"
                data_dir = os.path.join(f"output_crops_{file_name}", crop_filename)
                crop_path = os.path.join(output_crop_dir, crop_filename)
                crop.save(crop_path)
                # Batch prediction
                # predictions = corrector(clean_text, max_length=MAX_LENGTH)
                # clean_text = predictions[0]['generated_text']
                # print(f"  {j+1}. {clean_text}")

                safe_text = clean_text.replace('\n', ' ').replace('\r', ' ')
                f_crop_info.write(f"{i+1},{j+1},{crop_filename},{score:.4f},{safe_text}\n")
                f_data.write(f"{data_dir}\t{safe_text}\n")

                if not clean_text or clean_text.upper() == "FF":
                    clean_text=";"
                else:
                    clean_text = normalize_text(clean_text)

                clean_texts.append(clean_text)
                if clean_text.strip() != "":
                    box_key = (i, tuple(map(tuple, box)))  # Thêm page_number vào key (i+1 nếu muốn bắt đầu từ 1)
                    if box_key not in box_set:  # Chỉ thêm nếu chưa có
                        box_set.add(box_key)
                        box_to_text[box_key] = clean_text
                        # print(f"  [{i+1},{j+1}] 📦 Box mới: {box_key} → {clean_text}")

                    else:
                        print(f"  🚫 Bỏ qua box đã có: {box_key}")
                new_dt_polys.append(box)
                new_dt_scores.append(score)
                
            output_img_path = os.path.join(output_dir, f"{file_name}_page_{i+1}_with_boxes.png")
            draw_polygons_on_image(image_path, new_dt_polys, color=(0, 255, 0), thickness=2, output_path=output_img_path)
            print(f"Đã lưu ảnh với box: {output_img_path}")
    
    return output_txt_path, box_to_text, len(image_paths), cell_box_to_text_map, table_map

def build_char_map_list(output_txt_path, box_to_text, page_count):
    glob_line_idx = 1
    char_map = {}
    print(f"page_count: {page_count}")
    print(f"📄 Tạo char_map từ {output_txt_path} với {len(box_to_text)} box và {page_count} trang...")
    with open(output_txt_path, "w", encoding="utf-8") as f_out:
        for i in range(page_count):
            # print(f"📄 Trang {i+1}: {output_txt_path}")
            # Bạn cần lấy lại các box và text theo từng trang từ box_to_text
            page_number = i  # Giả sử bạn muốn lấy trang cuối cùng
            # Lọc các box thuộc trang này
            page_boxes = [(box, text) for (pg, box), text in box_to_text.items() if pg == page_number]
            # Sắp xếp lại nếu cần
            boxes = [box for box, text in page_boxes]
            texts = [text for box, text in page_boxes]
            groups = group_boxes_by_line(boxes, texts, line_threshold=20)
            line_idx = 1
            
            for group in groups:
                group_texts = [t[1] for t in group if t[1]]
                if group_texts:
                    # print(f"group_texts: {group_texts}")
                    norm_line = " ".join(group_texts)
                    # print(f"page {page_number} - norm_line: {norm_line}")
                    # if norm_line:
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

def crop_box_from_image(img, box):
    """
    Crop vùng box từ ảnh gốc với biến đổi phối cảnh.
    Args:
        img: PIL.Image hoặc numpy array (RGB)
        box: list 4 điểm [[x1, y1], ...] theo thứ tự bất kỳ
    Returns:
        crop: PIL.Image RGB đã được cắt và biến đổi phối cảnh
    """
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    pts = np.array(list(zip(xs, ys)), dtype=np.int32)

    # Sắp xếp lại điểm theo thứ tự: [top-left, top-right, bottom-right, bottom-left]
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]       # top-left
        rect[2] = pts[np.argmax(s)]       # bottom-right
        rect[1] = pts[np.argmin(diff)]    # top-right
        rect[3] = pts[np.argmax(diff)]    # bottom-left
        return rect

    rect_pts = order_points(pts)

    # Tính width và height tối đa của vùng chữ
    widthA = np.linalg.norm(rect_pts[2] - rect_pts[3])
    widthB = np.linalg.norm(rect_pts[1] - rect_pts[0])
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(rect_pts[1] - rect_pts[2])
    heightB = np.linalg.norm(rect_pts[0] - rect_pts[3])
    maxHeight = int(max(heightA, heightB))

    # Điểm đích phẳng, vuông góc
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Biến đổi phối cảnh
    if isinstance(img, Image.Image):
        img_cv = np.array(img)
    else:
        img_cv = img
    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warp = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight))

    # Convert sang ảnh PIL RGB
    crop = Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
    return crop

def process_doc_file(pdf_path, output_dir="./output"):
    """
    Xử lý file PDF, trích xuất text ngoài bảng và dữ liệu bảng ra 2 file riêng.
    Lọc bảng lặp, bảng con, bảng nhỏ.
    """
    import pdfplumber
    from pdfplumber.utils import intersects_bbox
    import os
    from unidecode import unidecode

    t0 = time.time()
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Đang xử lý tệp PDF: {pdf_path}")

    output_txt_path = os.path.join(output_dir, f"{file_name}_result_doc_text.txt")
    output_table_path = os.path.join(output_dir, f"{file_name}_tables.txt")
    seen_tables_global = set()

    with pdfplumber.open(pdf_path) as pdf, \
         open(output_txt_path, "w", encoding="utf-8") as f_text, \
         open(output_table_path, "w", encoding="utf-8") as f_table:

        for i, page in enumerate(pdf.pages):
            tables = page.find_tables()
            valid_bboxes = []
            for ti, table in enumerate(tables):
                raw = table.extract()
                if raw and len(raw) > 1:
                    valid_bboxes.append(table.bbox)
                else:
                    print(f"[DEBUG] skipping empty table bbox on page {i+1}, bbox={table.bbox}")

            # Tách text ngoài bảng (ưu tiên giữ dòng chứa từ khóa)
            lines = page.extract_text_lines(layout=True) or []
            text_blocks = []
            for obj in lines:
                txt = obj.get("text", "").strip()
                if not txt:
                    continue
                inside = any(intersects_bbox([obj], bb) for bb in valid_bboxes)
                
                if not inside:
                    text_blocks.append((obj["top"], txt))
                else:
                    print(f"[DEBUG] dropped as in-table-area: '{txt}' on page {i+1}")

            text_blocks.sort(key=lambda x: x[0])
            for _, line in text_blocks:
                # Làm sạch như cũ
                for token in special_tokens:
                    line = line.replace(token, " ")
                clean_line = line.translate(translator)
                no_accent_line = unidecode(clean_line)
                single_space_line = ' '.join(no_accent_line.split())
                check_text = normalize_text(single_space_line)
                if not single_space_line:
                    continue
                if "tham dinh phap ly" in check_text or "tham dinh" in check_text:
                    print(f"  🚫 Bỏ qua dòng chứa 'tham dinh phap ly'")
                    continue
                f_text.write(single_space_line + "\n")

            # Ghi bảng
            for ti, table in enumerate(tables):
                raw = table.extract()
                if not raw or len(raw) <= 1:
                    continue
                cleaned = []
                prev = None
                for row in raw:
                    cells = [" ".join((c or "").replace("\n", " ").split()) for c in row]
                    non_empty = [j for j,c in enumerate(cells) if c]
                    if prev and len(non_empty)==1 and all(not cells[k] for k in range(len(cells)) if k!=non_empty[0]):
                        prev[non_empty[0]] += ' ' + cells[non_empty[0]]
                    else:
                        if prev:
                            cleaned.append(prev)
                        prev = cells
                if prev:
                    cleaned.append(prev)
                if len(cleaned) < 2 or all(len(r)==1 for r in cleaned):
                    continue
                key = tuple(tuple(r) for r in cleaned)
                if key in seen_tables_global:
                    continue
                seen_tables_global.add(key)
                # f_table.write(f"--- Table {ti+1} - Page {i+1} ---\n")
                for r in cleaned:
                    r = [normalize_text(cell) for cell in r]
                    f_table.write("\t".join(r) + "\n")

    t1 = time.time()
    print(f"⏱️ Thời gian xử lý {t1 - t0:.2f} giây")
    print(f"✅ Đã trích xuất văn bản KHÔNG DẤU từ tệp PDF và lưu vào '{output_txt_path}'")
    print(f"✅ Đã trích xuất dữ liệu bảng từ tệp PDF và lưu vào '{output_table_path}'")
    return output_txt_path, output_table_path

# def is_bbox_inside(inner, outer):
#     # inner, outer: (x0, y0, x1, y1)
#     return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

# def process_doc_file(pdf_path, output_dir="./output"):
#     """
#     Xử lý file PDF, trích xuất text ngoài bảng và dữ liệu bảng ra 2 file riêng.
#     Lọc bảng lặp, bảng con, bảng nhỏ.
#     """
#     t0 = time.time()
#     file_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"📄 Đang xử lý tệp PDF: {pdf_path}")

#     output_txt_path = os.path.join(output_dir, f"{file_name}_result_doc_text.txt")
#     output_table_path = os.path.join(output_dir, f"{file_name}_tables.txt")
#     seen_tables_global = set()

#     with fitz.open(pdf_path) as pdf, \
#          open(output_txt_path, "w", encoding="utf-8") as f_text, \
#          open(output_table_path, "w", encoding="utf-8") as f_table:

#         for i, page in enumerate(pdf):
#             tables = page.find_tables()
#             # Lọc bảng con: chỉ giữ bảng lớn nhất (không nằm hoàn toàn trong bảng khác)
#             bboxes = [table.bbox for table in tables if table.extract() and len(table.extract()) > 1]
#             keep = [True] * len(bboxes)
#             for idx, bb in enumerate(bboxes):
#                 for jdx, obb in enumerate(bboxes):
#                     if idx != jdx and is_bbox_inside(bb, obb):
#                         keep[idx] = False
#                         break
#             valid_tables = [t for t, k in zip(tables, keep) if k]
#             valid_bboxes = [t.bbox for t in valid_tables]

#             # Trích xuất văn bản ngoài bảng
#             text_dict = page.get_text("dict")
#             text_lines = []
#             for block in text_dict["blocks"]:
#                 if block["type"] != 0:
#                     continue
#                 for line in block["lines"]:
#                     line_bbox = fitz.Rect(line["bbox"])
#                     inside = any(line_bbox.intersects(fitz.Rect(bb)) for bb in valid_bboxes)
#                     if not inside:
#                         line_text = " ".join(span["text"] for span in line["spans"])
#                         text_lines.append((line_bbox.y0, line_text))
#                     else:
#                         line_text = " ".join(span["text"] for span in line["spans"])
#                         print(f"[DEBUG] dropped as in-table-area: '{line_text}' on page {i+1}")

#             text_lines.sort(key=lambda x: x[0])
#             for _, line in text_lines:
#                 # Làm sạch như cũ
#                 for token in special_tokens:
#                     line = line.replace(token, " ")
#                 clean_line = line.translate(translator)
#                 no_accent_line = unidecode(clean_line)
#                 single_space_line = ' '.join(no_accent_line.split())
#                 check_text = normalize_text(single_space_line)
#                 if not single_space_line:
#                     continue
#                 if "tham dinh phap ly" in check_text or "tham dinh" in check_text:
#                     print(f"  🚫 Bỏ qua dòng chứa 'tham dinh phap ly'")
#                     continue
#                 f_text.write(single_space_line + "\n")

#             # Xử lý và ghi các bảng
#             for ti, table in enumerate(valid_tables):
#                 raw = table.extract()
#                 if not raw or len(raw) <= 1:
#                     continue
#                 cleaned = []
#                 prev = None
#                 for row in raw:
#                     cells = [" ".join((c or "").replace("\n", " ").split()) for c in row]
#                     non_empty = [j for j, c in enumerate(cells) if c]
#                     if prev and len(non_empty) == 1 and all(not cells[k] for k in range(len(cells)) if k != non_empty[0]):
#                         prev[non_empty[0]] += ' ' + cells[non_empty[0]]
#                     else:
#                         if prev:
#                             cleaned.append(prev)
#                         prev = cells
#                 if prev:
#                     cleaned.append(prev)
#                 if len(cleaned) < 2 or all(len(r) == 1 for r in cleaned):
#                     continue
#                 key = tuple(tuple(r) for r in cleaned)
#                 if key in seen_tables_global:
#                     continue
#                 seen_tables_global.add(key)
#                 for r in cleaned:
#                     r = [normalize_text(cell) for cell in r]
#                     f_table.write("\t".join(r) + "\n")

#     t1 = time.time()
#     print(f"⏱️ Thời gian xử lý {t1 - t0:.2f} giây")
#     print(f"✅ Đã trích xuất văn bản KHÔNG DẤU từ tệp PDF và lưu vào '{output_txt_path}'")
#     print(f"✅ Đã trích xuất dữ liệu bảng từ tệp PDF và lưu vào '{output_table_path}'")
#     return output_txt_path, output_table_path

# def process_doc_file(pdf_path, output_dir="./output"):
#     """
#     Xử lý file PDF, trích xuất text từng dòng bao gồm cả bảng.
#     """
#     import pdfplumber

#     t0 = time.time()
#     file_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"📄 Đang xử lý tệp PDF: {pdf_path}")

#     output_txt_path = os.path.join(output_dir, f"{file_name}_result_doc_text.txt")
    
#     with pdfplumber.open(pdf_path) as pdf, \
#          open(output_txt_path, "w", encoding="utf-8") as f_out:
        
#         for i, page in enumerate(pdf.pages):
#             print(f"📄 Trang {i+1}: {output_txt_path}")
            
#             # Trích xuất toàn bộ text của trang (bao gồm cả bảng)
#             full_text = page.extract_text()
            
#             if full_text:
#                 # Xử lý từng dòng
#                 for line in full_text.split("\n"):
#                     stripped_line = line.strip()
#                     if not stripped_line:
#                         continue
                    
#                     # Làm sạch text
#                     for token in special_tokens:
#                         stripped_line = stripped_line.replace(token, " ")
                    
#                     clean_line = stripped_line.translate(translator)
#                     no_accent_line = unidecode(clean_line)
#                     single_space_line = ' '.join(no_accent_line.split())
#                     check_text = normalize_text(single_space_line)
#                     if not single_space_line:
#                         continue
#                     if "tham dinh phap ly" in check_text or "tham dinh" in check_text:
#                         print(f"  🚫 Bỏ qua dòng chứa 'tham dinh phap ly'")
#                         continue

#                     f_out.write(single_space_line + "\n")
    
#     t1 = time.time()
#     print(f"⏱️ Thời gian xử lý {t1 - t0:.2f} giây")
#     print(f"✅ Đã trích xuất văn bản KHÔNG DẤU từ tệp PDF và lưu vào '{output_txt_path}'")
#     return output_txt_path

def normalize_text(text):

    if not any(c.isalnum() for c in text):
        return text
    """
    Chuẩn hóa text: bỏ dấu, loại lặp, loại từ không cần, chuẩn hóa khoảng trắng, lower, ...
    Mỗi chữ chỉ cách nhau 1 khoảng trắng, không có dòng nào trống.
    """
    # Bỏ đầu/cuối, về lowercase
    text = text.strip().lower()
    # Bỏ dấu tiếng Việt
    text = unidecode(text)
    # Loại bỏ các token đặc biệt
    for token in special_tokens:
        text = text.replace(token, " ")

    # Loại bỏ dòng chỉ toàn chữ cái (không có số/ký tự khác)
    # if text.isalpha():
    #     return ""
    # Loại bỏ từ "scanner"
    if "scanner" in text:
        return ""
    
    # Loại bỏ dấu câu
    text = text.translate(translator)
    # Chuẩn hóa khoảng trắng từng dòng, loại bỏ dòng trống
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = ' '.join(line.split())
        if line:  # Bỏ dòng trống
            clean_lines.append(line)
    return '\n'.join(clean_lines)



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
    
    if all:
        # Nếu so sánh tất cả, không chuẩn hóa gì cả
        norm_text1 = normalize_text(text1)
        norm_text2 = normalize_text(text2)
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
    
    # with open(norm1_path, "r", encoding="utf-8") as f1, open(norm2_path, "r", encoding="utf-8") as f2:
    #     lines1 = [normalize_line(line) for line in f1.readlines()]
    #     lines2 = [normalize_line(line) for line in f2.readlines()]
    
    # with open(norm1_path, "w", encoding="utf-8") as f1, open(norm2_path, "w", encoding="utf-8") as f2:
    #     f1.write("\n".join(lines1))
    #     f2.write("\n".join(lines2))

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
                #print(f"\033[91m[-]{data}\033[0m")  # Xóa (chỉ có ở file 1), màu đỏ
            # elif label == "[+]":
            # elif op == 1:
            #     if data.strip() == "":
            #         continue  # Nếu chỉ là dấu cách, bỏ qua
            #         #print(f"{data}")      # Nếu chỉ là dấu cách, in bình thường
            #     else:
            #         f_out.write(f"[+]{data}[+]")

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

def main():
    input_dir1 = "./input1"
    input_dir2 = "./input2"
    scan_dir = "./scan_data"
    doc_dir = "./doc_data"
    output_dir1 = "./output1"
    output_dir2 = "./output2"
    final_output_dir = "./final_output"

    os.makedirs(input_dir1, exist_ok=True)
    os.makedirs(input_dir2, exist_ok=True)
    os.makedirs(scan_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    basename = ""
    t0 = time.time()

    for file_idx, fname in enumerate(os.listdir(input_dir1)):
        char_box_mappings1 = None
        char_box_mappings2 = None
        txt1_paths = None
        txt2_paths = None

        # for fname in os.listdir(input_dir1):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(input_dir1, fname)
        base = os.path.splitext(fname)[0]
        print(f"file name 1: {base}")
        scanned1, ratio = is_scanned_pdf(fpath)
        if not scanned1:
            dest = os.path.join(scan_dir, fname)
            os.rename(fpath, dest)
            print(f"→ {fname} là SCAN, chuyển vào {scan_dir}")
            txt1_paths, char_box_mappings1, pdf1_paths = process_scan_file(dest, output_dir=output_dir1,  text_detector=text_detector, detector_model=detector)
        else:
            dest = os.path.join(doc_dir, fname)
            os.rename(fpath, dest)
            print(f"→ {fname} là DOC, chuyển vào {doc_dir}")
            txt1_paths = process_doc_file(dest, output_dir=output_dir1)
        # txt1_paths[base] = txt_path
        basename = base  # Lưu tên cơ sở để so sánh sau

        # for fname in os.listdir(input_dir2):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(input_dir2, fname)
        base = os.path.splitext(fname)[0]
        print(f"file name 2: {base}")
        scanned2, ratio = is_scanned_pdf(fpath)
        if not scanned2:
            dest = os.path.join(scan_dir, fname)
            os.rename(fpath, dest)
            print(f"→ {fname} là SCAN, chuyển vào {scan_dir}")
            txt2_paths, char_box_mappings2, pdf2_paths = process_scan_file(dest, output_dir=output_dir2, text_detector=text_detector, detector_model=detector)
        else:
            dest = os.path.join(doc_dir, fname)
            os.rename(fpath, dest)
            print(f"→ {fname} là DOC, chuyển vào {doc_dir}")
            txt2_paths = process_doc_file(dest, output_dir=output_dir2)
        # txt2_paths[base] = txt_path
        basename = base
        print("len(txt1_paths):", len(txt1_paths))
        print("len(txt2_paths):", len(txt2_paths))
        
        # Ghép cặp và so sánh
        # for base in txt1_paths:
        #     if base in txt2_paths:
        for i in range(0, len(txt1_paths)):
            # Tổng hợp danh sách box
            print(f"\n🔎 So sánh: {txt1_paths[i]} <-> {txt2_paths[i]}")
            diff_output1 = os.path.join(final_output_dir, f"page_{i+1}_diff1.txt")
            diff_output2 = os.path.join(final_output_dir, f"page_{i+1}_diff2.txt")

            # Vẽ box vào ảnh scan gốc (ví dụ trang đầu tiên)
            # Bạn cần xác định đúng đường dẫn ảnh gốc tương ứng với trang đang xử lý
            if not scanned1:
                box_list1 = []
                doc_compare(txt1_paths[i], txt2_paths[i], diff_output1, label="[-]")
                deleted_boxes1 = extract_boxes_from_diff(diff_output1, char_box_mappings1[i], label="[-]")
                for item in deleted_boxes1:
                    # print(f"Ký tự '{item['char']}' trong từ '{item['word']}' tại dòng {item['line_idx']} → box: {item['box']}")
                    box_list1.append(item['box'])

                pdf_path = pdf1_paths[i]  # Giả sử txt1_paths[i] là đường dẫn đến ảnh gốc
                print(f"pdf_path: {pdf_path}")
                img_marked = mark_diff_boxes_on_image(pdf_path, box_list1, color=(0,0,255))
                output_img_path = os.path.join(final_output_dir, f"page_{i+1}_marked1.png")
                cv2.imwrite(output_img_path, img_marked)
                print(f"Đã lưu ảnh đánh dấu box khác biệt: {output_img_path}")

                t1 = time.time()
                print(f"\n⏱️ Thời gian tổng cộng: {t1 - t0:.2f} giây")

            if not scanned2:
                box_list2 = []
                doc_compare(txt2_paths[i], txt1_paths[i], diff_output2, label="[-]")
                deleted_boxes2 = extract_boxes_from_diff(diff_output2, char_box_mappings2[i], label="[-]")
                for item in deleted_boxes2:
                    # print(f"Ký tự '{item['char']}' trong từ '{item['word']}' tại dòng {item['line_idx']} → box: {item['box']}")
                    box_list2.append(item['box'])

                pdf_path = pdf2_paths[i]
                img_marked = mark_diff_boxes_on_image(pdf_path, box_list2, color=(0,255,0))
                output_img_path = os.path.join(final_output_dir, f"page_{i+1}_marked2.png")
                cv2.imwrite(output_img_path, img_marked)
                print(f"Đã lưu ảnh đánh dấu box khác biệt: {output_img_path}")
                # Kết thúc vòng lặp, in ra tổng thời gian
                t1 = time.time()
                print(f"\n⏱️ Thời gian tổng cộng: {t1 - t0:.2f} giây")

    print("txt1_paths:", txt1_paths)
    print("txt2_paths:", txt2_paths)
   
    # for item in deleted_boxes:
    #     print(f"Ký tự '{item['char']}' trong từ '{item['word']}' tại dòng {item['line_idx']} → box: {item['box']}")

if __name__ == "__main__":
    # main()
    process_scan_searchable_file("/root/doc_compare/scan_data/3.pdf", output_dir="./output", text_detector=text_detector, detector=detector)
    # process_scan_file('./scan_data/1.pdf', output_dir='./output')
    # process_doc_file('./doc_data/1.pdf', output_dir='./output')


