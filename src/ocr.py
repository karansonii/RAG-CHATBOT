# # src/ocr_ppt.py
# import io
# from pptx import Presentation
# from PIL import Image
# import numpy as np
# from paddleocr import PaddleOCR
# # Initialize PaddleOCR reader once (English only)
# # Set use_angle_cls=True if you want orientation correction
# ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
# def extract_images_from_slide(slide):
# """
# Extract all images from a PPTX slide as PIL Images.
# """
# images = []
# for shape in slide.shapes:
# if hasattr(shape, "image"):
# image_stream = io.BytesIO(shape.image.blob)
# img = Image.open(image_stream)
# images.append(img)
# return images
# def read_text_from_image(image):
# """
# Perform OCR on a PIL Image using PaddleOCR and return extracted text.
# """
# # Convert PIL Image to numpy array
# img_array = np.array(image)
# # OCR result: list of [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], text, confidence ]
# result = ocr_reader.ocr(img_array, cls=True)
# # Extract only text
# lines = [line[1][0] for page in result for line in page]
# return "\n".join(lines)
# def read_ppt_text(ppt_path):
# """
# Extract text from all slides in a PPT/PPTX file using PaddleOCR.
# Returns a dictionary with slide number as key and extracted text as value.
# """
# prs = Presentation(ppt_path)
# ppt_text = {}
# for i, slide in enumerate(prs.slides, start=1):
# slide_text = []
# # 1️⃣ Extract native text from shapes
# for shape in slide.shapes:
# if hasattr(shape, "text") and shape.text.strip():
# slide_text.append(shape.text.strip())
# # 2️⃣ Extract text from images (OCR)
# images = extract_images_from_slide(slide)
# for img in images:
# try:
# ocr_text = read_text_from_image(img)
# if ocr_text.strip():
# slide_text.append(ocr_text.strip())
# except Exception as e:
# print(f"⚠️ OCR error on slide {i}: {e}")
# # Combine all text for the slide
# ppt_text[f"Slide_{i}"] = "\n".join(slide_text)
# return ppt_text
# if __name__ == "__main__":
# # Example usage
# ppt_file = "example.pptx"
# slides_text = read_ppt_text(ppt_file)
# for slide_num, text in slides_text.items():
# print(f"--- {slide_num} ---")
# print(text)
# print("\n")
