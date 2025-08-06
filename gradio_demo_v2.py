# gradio_demo.py
import gradio as gr
import os
import shutil
import tempfile
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from paddleocr import TextDetection
import torch
from pdf2image import convert_from_path
import cv2
import numpy as np
import uuid
from pdf2image import convert_from_path
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import time
from collections import defaultdict
import psutil
from PIL import Image, ImageDraw

from test2_v2 import normalize_text, build_table_cell_map_from_box_text, ocr_image_with_gpt4, crop_box_from_image, build_char_map_list,  process_scan_file1, process_doc_file, is_scanned_pdf, doc_compare, extract_boxes_from_diff, extract_boxes_from_diff_v2, mark_diff_boxes_on_image, concat_images

# Thư mục tạm để lưu ảnh so sánh
CACHE_DIR = os.path.join(os.getcwd(), "tmp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = CACHE_DIR

# paddle ocr text detection
text_detector = TextDetection(model_name="PP-OCRv5_mobile_det")
model_type = "transformer"  # Hoặc "crnn" nếu bạn muốn dùng mô hình CRNN
config = Cfg.load_config_from_name(f'vgg_{model_type}')
config['weights'] = f'./weights/{model_type}ocr3.pth'  # Đảm bảo bạn đã tải trọng số phù hợp
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = Predictor(config)

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
        scanned1, _ = is_scanned_pdf(pdf1_path)
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

        if not scanned1:
            txt1_path, box_to_text1, page_count1, cell_box_to_text_map1, table_map1 = process_scan_file1(
                pdf1_path, output_dir="./out1", text_detector=text_detector, detector_model=detector, max_num_pages=max_page
            )
            cell_map1, txt_table_path1 = build_table_cell_map_from_box_text(
                cell_box_to_text_map1, table_map1, output_dir="./out1", file_name="file1"
            )
            char_box_mappings1 = build_char_map_list(txt1_path, box_to_text1, page_count1)
            print(f"Số box file 1: {len(char_box_mappings1)}")
        else:
            txt1_path, txt_table_path1 = process_doc_file(pdf1_path, output_dir="./out1")
            char_box_mappings1 = None
           # img_paths1, pdf1_paths = split_pdf_to_images(pdf1_path, tmpdir, "pdf1")

        # Xử lý file 2 (bên trái)
        scanned2, _ = is_scanned_pdf(pdf2_path)
        if not scanned2:
            txt2_path, box_to_text2, page_count2, cell_box_to_text_map2, table_map2 = process_scan_file1(
                pdf2_path, output_dir="./out2", text_detector=text_detector, detector_model=detector
            )
            cell_map2, txt_table_path2 = build_table_cell_map_from_box_text(
                cell_box_to_text_map2, table_map2, output_dir="./out2", file_name="file2"
            )
            char_box_mappings2 = build_char_map_list(txt2_path, box_to_text2, page_count2)
        else:
            print("File 2 là file scan, không cần xử lý OCR")
            txt2_path, txt_table_path2 = process_doc_file(pdf2_path, output_dir="./out2")
            char_box_mappings2 = None
            # img_paths2, pdf2_paths = split_pdf_to_images(pdf2_path, tmpdir, "pdf2")

        out_path1 = os.path.join(output_dir, f"compare1_file1_2.txt")
        out_path2 = os.path.join(output_dir, f"compare2_file2_1.txt")
        diff1 = os.path.join("./output", f"diff1.txt")
        diff2 = os.path.join("./output", f"diff2.txt")

        out_table1 = os.path.join(output_dir, f"compare_table1_file1.txt")
        out_table2 = os.path.join(output_dir, f"compare_table2_file2.txt")
        diff_table1 = os.path.join("./output", f"diff_table1.txt")
        diff_table2 = os.path.join("./output", f"diff_table2.txt")
        doc_compare(txt1_path, txt2_path, diff1, label="[-]")
        doc_compare(txt2_path, txt1_path, diff2, label="[-]")

        doc_compare(txt_table_path1, txt_table_path2, diff_table1, label="[-]")
        doc_compare(txt_table_path2, txt_table_path1, diff_table2, label="[-]")

        # Chuyển nội dung từ diff sang out_path
        # with open(diff1, 'r', encoding='utf-8') as fin, open(out_path1, 'w', encoding='utf-8') as fout:
        #     fout.write(f"Trang {i+1} - PDF 1:\n")
        #     fout.write(fin.read())

        # with open(diff2, 'r', encoding='utf-8') as fin, open(out_path2, 'w', encoding='utf-8') as fout:
        #     fout.write(f"Trang {i+1} - PDF 2:\n")
        #     fout.write(fin.read())

        
        # Mặc định ảnh trắng
        img1 = img2 = np.ones((800, 600, 3), np.uint8) * 255

        double_check1 = False  # Biến để kiểm tra xem có box nào cần OCR không
        double_check2 = False  # Biến để kiểm tra xem có box nào cần OCR không
        # Double check bằng OCR GPT-4 cho từng cell khác biệt
        double_check_cell = False
        if cell_map1:
            results1 = extract_boxes_from_diff_v2(diff_table1, cell_map1, label="[-]", table=True)
            boxes1, page1_idxs, table_paths1, table_boxs1, table_idxs1 = zip(*[(d['box'], d['page'], d['table_path'], d['table_box'], d['table_idx']) for d in results1])
            if boxes1:
                double_check_cell = True
                print(f"Số cell khác biệt file 1: {len(boxes1)}")
            else:
                print("Không có cell khác biệt trong file 1")
            # Đảm bảo img_paths1 là list ảnh gốc từng trang (PIL.Image)
            img_pages = [Image.open(p).convert("RGB") for p in img_paths1]
            # Gom các box theo (page_idx, tbl_path, tbl_box)
            table_boxes_map = defaultdict(list)
            for box, page_idx, tbl_path, tbl_box, tbl_idx in zip(boxes1, page1_idxs, table_paths1, table_boxs1, table_idxs1):
                key = (page_idx, tbl_path, tbl_idx, tuple(tbl_box))
                table_boxes_map[key].append(box)

            # Vẽ tất cả box lên từng bảng, rồi paste vào ảnh trang gốc
            for (page_idx, tbl_path, tbl_idx, tbl_box), box_list in table_boxes_map.items():
                tbl_img = Image.open(tbl_path).convert("RGB")
                box_set = set()
                for box in box_list:
                    if box in box_set:
                        continue
                    box_set.add(box)
                    print(f"Box: {box}, page_idx: {page_idx}, tbl_path: {tbl_path}, tbl_box: {tbl_box}")
                    # Chuẩn hóa box về polygon 4 điểm nếu là bbox
                    if isinstance(box[0], (int, float)):
                        # box dạng (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box
                        box_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    else:
                        box_pts = box
                    crop = crop_box_from_image(tbl_img, box_pts)
                    new_text = ocr_image_with_gpt4(crop)
                    new_text = normalize_text(new_text)
                    print(f"OCR GPT-4 cho cell: {new_text}")
                    # Tìm lại key trong cell_map1 để cập nhật text
                    k = (page_idx, tbl_idx, box)
                    print(f"current text: {cell_box_to_text_map1.get(k, '')}")
                    cell_box_to_text_map1[k] = new_text
                    
            if double_check_cell:    
                print(f"Double check cell")
                cell_map1, txt_table_path1 = build_table_cell_map_from_box_text(
                    cell_box_to_text_map1, table_map1, output_dir="./out1", file_name="file1"
                )

                doc_compare(txt_table_path1, txt_table_path2, diff_table1, label="[-]",double_check=False)
                doc_compare(txt_table_path2, txt_table_path1, diff_table2, label="[-]", double_check=False)

                results1 = extract_boxes_from_diff_v2(diff_table1, cell_map1, label="[-]", table=True)
                boxes1, page1_idxs, table_paths1, table_boxs1 = zip(*[(d['box'], d['page'], d['table_path'], d['table_box']) for d in results1])
                print(f"Số cell khác biệt file 1 sau OCR: {len(boxes1)}")
                # Gom các box theo (page_idx, tbl_path, tbl_box)
                new_table_boxes_map = defaultdict(list)
                for box, page_idx, tbl_path, tbl_box in zip(boxes1, page1_idxs, table_paths1, table_boxs1):
                    key = (page_idx, tbl_path, tuple(tbl_box))
                    new_table_boxes_map[key].append(box)
                # Vẽ tất cả box lên từng bảng, rồi paste vào ảnh trang gốc
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
                    out_dir = "./out"
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"page_{page_idx}_with_tablebox.png")
                    img_pages[page_idx-1].save(out_path)
                    # Lưu lại nếu muốn
                    # Ghi đè vào ảnh gốc trong img_paths1
                    img_pages[page_idx-1].save(img_paths1[page_idx-1])
                    print(f"Đã lưu ảnh kết quả: {out_path}")
        

        # File 1 (bên phải)
        if char_box_mappings1:
            results1 = extract_boxes_from_diff_v2(diff1, char_box_mappings1, label="[-]")
            boxes1, page1_idxs = zip(*[(d['box'], d['page']) for d in results1])
            # Tạo dict: page_idx -> list box
            boxes_by_page = defaultdict(list)
            for box, page_idx in zip(boxes1, page1_idxs):
                boxes_by_page[page_idx].append(box)
            diff_boxes_list = [boxes_by_page[i] for i in range(len(img_paths1))]
            imgs1 = mark_diff_boxes_on_image(img_paths1, diff_boxes_list, color=(0,0,255))
            
            for i in range(len(img_paths1)):
                img_pil1 = Image.open(img_paths1[i]).convert("RGB")  # Đọc ảnh gốc 1 lần ngoài vòng lặp
                page_boxes = [box for box, page_idx in zip(boxes1, page1_idxs) if page_idx == i]
                print(f"Trang {i+1} - Số box file 1: {len(set(page_boxes))}")
                if page_boxes:
                    double_check1 = True  # Có box cần OCR
                    box_set1 = set()
    
                    for box in page_boxes:
                        box_key = (i, tuple(map(tuple, box)))
                        
                        if box_key in box_to_text1 and box_key not in box_set1:
                            # Crop vùng box từ ảnh gốc
                            crop = crop_box_from_image(img_pil1, box)
                            # Nhận diện text bằng GPT-4 OCR
                            new_text = ocr_image_with_gpt4(crop)
                            # Loại bỏ tất cả ký tự xuống dòng, kể cả Unicode line separator
                            new_text = normalize_text(new_text).replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')
                            box_to_text1[box_key] = new_text
                            print(f"OCR file 1 GPT-4: {new_text}")
                            box_set1.add(box_key)
                            # print(f"text: {box_to_text1[box_key]}")
            
            if double_check1:
                char_box_mappings1_new = build_char_map_list(txt1_path, box_to_text1, page_count1)
                doc_compare(txt1_path, txt2_path, diff1, label="[-]", all=True, double_check=False)
                doc_compare(txt2_path, txt1_path, diff2, label="[-]", all=True, double_check=False)
                if char_box_mappings1_new:
                    new_results1 = extract_boxes_from_diff_v2(diff1, char_box_mappings1_new, label="[-]")
                    pairs = [(d['box'], d['page']) for d in new_results1]
                    if pairs:
                        boxes1, page1_idxs = zip(*pairs)
                    else:
                        boxes1, page1_idxs = [], []
                    print(f"Trang {i+1} - Số box_new file 1: {len(set(boxes1))}")
                    # Tạo dict: page_idx -> list box
                    boxes_by_page = defaultdict(list)
                    for box, page_idx in zip(boxes1, page1_idxs):
                        boxes_by_page[page_idx].append(box)
                    diff_boxes_list = [boxes_by_page[i] for i in range(len(img_paths1))]
                    imgs1 = mark_diff_boxes_on_image(img_paths1, diff_boxes_list, color=(0,0,255))
        else:
            imgs1 = [cv2.imread(img_path1) for img_path1 in img_paths1]

        # File 2 (bên trái)
        if char_box_mappings2:
            results2 = extract_boxes_from_diff_v2(diff2, char_box_mappings2, label="[-]")
            boxes2, page2_idxs = zip(*[(d['box'], d['page']) for d in results2])
            selected_img_paths2 = [img_paths2[idx] for idx in page2_idxs]
            # Tạo dict: page_idx -> list box
            boxes_by_page = defaultdict(list)
            for box, page_idx in zip(boxes2, page2_idxs):
                boxes_by_page[page_idx].append(box)
            diff_boxes_list = [boxes_by_page[i] for i in range(len(img_paths2))]
            imgs2 = mark_diff_boxes_on_image(img_paths2, diff_boxes_list, color=(0,255,0))
            
        
        else:
            imgs2 = [cv2.imread(img_path2) for img_path2 in img_paths2]

        # Resize cho dễ nhìn
        def resize_img(img, scale=1.5):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)


        imgs1 = [resize_img(img1, scale=1) for img1 in imgs1]
        imgs2 = [resize_img(img2, scale=1) for img2 in imgs2]

            # Ghép: file 2 bên trái, file 1 bên phải
        num_pages = min(len(img_paths1), len(img_paths2))
        count = 14
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
