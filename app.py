# gradio_demo.py
import gradio as gr
import os
import shutil
import tempfile
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from paddleocr import TextDetection, LayoutDetection, TextImageUnwarping, TableCellsDetection
import torch
from pdf2image import convert_from_path
import cv2
import numpy as np
import uuid
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import time
from collections import defaultdict
import psutil
from PIL import Image, ImageDraw
from utils_v2 import *
from src.utils import *
from src.scan_pdf.scan_pipeline import build_char_map_list, run_scan_pdf_pipeline
from src.compare_pipeline import doc_compare, run_compare_pipeline

# Thư mục tạm để lưu ảnh so sánh
CACHE_DIR = os.path.join(os.getcwd(), "tmp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = CACHE_DIR

# paddle ocr text detection
text_detector = TextDetection(model_name="PP-OCRv5_server_det")
model_type = "transformer"  # Hoặc "crnn" nếu bạn muốn dùng mô hình CRNN
config = Cfg.load_config_from_name(f'vgg_{model_type}')
config['weights'] = f'./weights/{model_type}ocr_v2.pth'  # Đảm bảo bạn đã tải trọng số phù hợp
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = Predictor(config)
layout_detector = LayoutDetection(model_name="PP-DocLayout_plus-L")
table_cells_detector = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det")

def preview_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=250)
    preview_paths = []
    for idx, img in enumerate(images):
        out_path = os.path.join(CACHE_DIR, f"preview_{uuid.uuid4().hex}_page_{idx+1}.png")
        img.save(out_path, "PNG")
        preview_paths.append(out_path)
    return preview_paths

def split_pdf_to_images(pdf_path, out_dir, prefix):

    images = convert_from_path(pdf_path, dpi=250)
    img_paths = []
    pdf_paths = []

    # Đọc PDF gốc để tách từng trang
    reader = PdfReader(pdf_path)

    for i, img in enumerate(images):
        # Lưu ảnh
        img_path = os.path.join(out_dir, f"{prefix}_page_{i+1}.png")
        img.save(img_path, "PNG")
        img_paths.append(img_path)

        # Lưu PDF một trang
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        pdf_page_path = os.path.join(out_dir, f"{prefix}_page_{i+1}.pdf")
        with open(pdf_page_path, "wb") as f:
            writer.write(f)
        pdf_paths.append(pdf_page_path)

    return img_paths, pdf_paths

def compare_pdfs_stream(pdf1, pdf2):

    out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    output_dir = './output'

    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as tmpdir:
        pdf1_path = os.path.join(tmpdir, "file1.pdf")
        pdf2_path = os.path.join(tmpdir, "file2.pdf")
        shutil.copy(pdf1, pdf1_path)
        shutil.copy(pdf2, pdf2_path)

        gallery = []
        t0 = time.time()
        # Xử lý từng trang: OCR, so sánh, vẽ box, ghép ảnh
        # Xử lý file 1 (bên phải)
        
        img_paths1, pdf1_paths = split_pdf_to_images(pdf1_path, tmpdir, "pdf1")
        img_paths2, pdf2_paths = split_pdf_to_images(pdf2_path, tmpdir, "pdf2")
        
        max_page = min(len(img_paths1), len(img_paths2))
        # max_page = min(max_page, 2)  # Giới hạn số trang so sánh tối đa là 3
        print(f"Giới hạn số trang so sánh: {max_page}")
        img_paths1 = img_paths1[:max_page]
        pdf1_paths = pdf1_paths[:max_page]
        img_paths2 = img_paths2[:max_page]
        pdf2_paths = pdf2_paths[:max_page]
        print(f"Số trang file 1: {len(img_paths1)}")
        print(f"Số trang file 2: {len(img_paths2)}")

        out = run_compare_pipeline(pdf1_path, pdf2_path, max_num_pages=max_page, output_dir=output_dir,
                                    detector=detector,
                                    text_detector=text_detector,
                                    layout_detector=layout_detector,
                                    table_cells_detector=table_cells_detector)

        # Mặc định ảnh trắng
        img1 = img2 = np.ones((800, 600, 3), np.uint8) * 255

        double_check1 = False  # Biến để kiểm tra xem có box nào cần OCR không
        double_check2 = False  # Biến để kiểm tra xem có box nào cần OCR không
        # Double check bằng OCR GPT-4 cho từng cell khác biệt
        double_check_cell = False
        cell_map1 = out["result1"]["cell_map"]
        cell_map2 = out["result2"]["cell_map"]

        if cell_map1:
            cell_box_to_text_map1 = out["result1"]["cell_box_to_text_map"]
            table_map1 = out["result1"]["table_map"]
            diff_table1 = out["diff_table1"]
            txt_table_path1 = out["result1"]["text_table_path"]
            txt_table_path2 = out["result2"]["text_table_path"]
            cell_map1, cell_box_to_text_map1 = double_check_and_draw_cells(
                cell_map1, cell_box_to_text_map1, table_map1,
                diff_table1, txt_table_path1, txt_table_path2, img_paths1,
                out_dir="./out1", file_name="file1"
            )
        
        if cell_map2:
            cell_box_to_text_map2 = out["result2"]["cell_box_to_text_map"]
            table_map2 = out["result2"]["table_map"]
            diff_table2 = out["diff_table2"]
            txt_table_path1 = out["result1"]["text_table_path"]
            txt_table_path2 = out["result2"]["text_table_path"]
            cell_map2, cell_box_to_text_map2 = double_check_and_draw_cells(
                cell_map2, cell_box_to_text_map2, table_map2,
                diff_table2, txt_table_path1, txt_table_path2, img_paths2,
                out_dir="./out2", file_name="file2"
            )

        txt1_path = out["result1"]["text_path"]
        txt2_path = out["result2"]["text_path"]

        # File 1 (bên trái)
        diff1 = out["diff1"]
        char_box_mappings1 = out["result1"]["char_box_mappings"]
        box_to_text1 = out["result1"]["box_to_text"]
        imgs1 = double_check_and_draw(
            img_paths1, char_box_mappings1, diff1,
            box_to_text1, txt1_path, txt2_path, max_page, color=(0,0,255)
        )

        # File 2 (bên phải)
        diff2 = out["diff2"]
        char_box_mappings2 = out["result2"]["char_box_mappings"]
        box_to_text2 = out["result2"]["box_to_text"]
        imgs2 = double_check_and_draw(
            img_paths2, char_box_mappings2, diff2,
            box_to_text2, txt2_path, txt1_path, max_page, color=(0,255,0)
        )

        # Resize cho dễ nhìn
        def resize_img(img, scale=1.5):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)


        imgs1 = [resize_img(img1, scale=1) for img1 in imgs1]
        imgs2 = [resize_img(img2, scale=1) for img2 in imgs2]

        # Ghép: file 2 bên trái, file 1 bên phải
        num_pages = min(len(img_paths1), len(img_paths2))
        count = 38
        out_dir = './output/compare_contract_' + str(count)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_pages):
            out_img = concat_images(imgs1[i], imgs2[i])
            out_path = os.path.join(out_dir, f"compare_contract_{count}_page_{i+1}.png")
            cv2.imwrite(out_path, out_img)
            t1 = time.time()
            process = psutil.Process(os.getpid())
            ram_used = process.memory_info().rss / (1024 ** 2)  # MB
            caption = f"Trang {i+1} - Thời gian xử lý: {t1-t0:.2f} giây - RAM: {ram_used:.2f} MB"
            print(caption)  # Log ra terminal
            gallery.append((out_path, caption))
            yield gallery  # Trả về toàn bộ gallery đã có

with gr.Blocks() as demo:
    gr.Markdown("### So sánh từng trang PDF (hiện từng trang ngay khi xử lý xong)")

    with gr.Row():
        with gr.Column():
            pdf1 = gr.File(label="PDF 1", file_types=[".pdf"], type="filepath")
            preview1 = gr.Gallery(label="Xem trước PDF 1", show_label=True, columns=1, height=400, allow_preview=True)
        with gr.Column():
            pdf2 = gr.File(label="PDF 2", file_types=[".pdf"], type="filepath")
            preview2 = gr.Gallery(label="Xem trước PDF 2", show_label=True, columns=1, height=400, allow_preview=True)

    img_out = gr.Gallery(label="Kết quả từng trang (trái: PDF 2, phải: PDF 1)", columns=2, height=800)
    btn = gr.Button("So sánh")

    btn.click(compare_pdfs_stream, inputs=[pdf1, pdf2], outputs=img_out)
    pdf1.change(preview_pdf, inputs=pdf1, outputs=preview1)
    pdf2.change(preview_pdf, inputs=pdf2, outputs=preview2)

if __name__ == "__main__":
    demo.launch(allowed_paths=[CACHE_DIR])
