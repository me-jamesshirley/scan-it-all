import os
import pytesseract
import sane
import re
import cv2
import threading
import multiprocessing
import numpy as np
import tkinter as tk
import difflib
import json
from tkinter import simpledialog
from datetime import datetime
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

class ScannedDocument:
    def __init__(self, image, ocr_image, ocr_text, date, entity, classifcation, base_path='~/Documents'):
        self.image = image
        self.ocr_image = ocr_image
        self.ocr_text = ocr_text
        self.date = date
        self.entity = entity
        self.classifcation = classifcation
        self.base_path = os.path.expanduser(base_path)

    def save(self):
        path = self.base_path + '/' + self.classifcation + '/' + str(self.date.date()) + '/' + self.entity + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        filename_prefix = str(self.date)
        # Save the image to a file
        filename_image = filename_prefix + '.png'
        full_filename_image = path + filename_image
        self.image.save(full_filename_image)

        # Save the OCR image to a file
        ocr_filename_image = filename_prefix + '_ocr.png'
        ocr_full_filename_image = path + ocr_filename_image
        self.ocr_image.save(ocr_full_filename_image)

        # Save the organized text to a file
        filename_text = filename_prefix + '.txt'
        full_filename_text = path + filename_text
        with open(full_filename_text, 'w') as f:
            f.write(self.ocr_text)

        print('Saved image: [' + full_filename_image + ']')
        print('Saved ocr image: [' + ocr_full_filename_image + ']')
        print('Saved text: [' + full_filename_text + ']')

class NoItemFound(Exception):
    pass


def get_scan_optimized_image(original_image):
    ocr_image = original_image.convert('L')

    # Adjust brightness and contrast
    ocr_image = ImageEnhance.Brightness(ocr_image).enhance(1.5)
    ocr_image = ImageEnhance.Contrast(ocr_image).enhance(2)

    # Denoise the image
    ocr_image = ocr_image.filter(ImageFilter.MedianFilter(size=3))

    return ocr_image
        
def get_ocr_optimized_image(original_image):
    scan_image = get_scan_optimized_image(original_image)

    # Binarize
    scan_image = scan_image.point( lambda p: 255 if p > 250 else 0 )
    # To mono
    return scan_image.convert('1')


def get_value_from_format(formats, text, fallback):
    for a_format in formats:
        match = re.search(a_format[0], text)
        if match:
            return datetime.strptime(match.group(0), a_format[1]) 

    return fallback


def find_datetime(text):
    date_formats = [(r'(\d{2}/\d{2}/\d{2})', '%m/%d/%y'), (r'(\d{2}-\d{2}-\d{2})', '%m-%d-%y')]
    time_formats = [(r'(\d{2}:\d{2}:\d{2})', '%H:%M:%S'), (r'(\d{2}:\d{2})', '%H:%M')]

    parsed_date = get_value_from_format(date_formats, text, datetime.now()).date()
    parsed_time = get_value_from_format(time_formats, text, datetime.now()).time()

    return datetime.combine(parsed_date, parsed_time)


def find_best_match(text, items_to_test):
    best_match = None
    acceptable_ratio = .6
    for line in text.splitlines():
        lower_line = line.lower()
        lower_split_line = lower_line.split()
        for item in items_to_test:
            lower_item = item.lower()
            line_ratio = difflib.SequenceMatcher(None, lower_item, lower_line).ratio()
            if line_ratio > acceptable_ratio:
                best_match = item
                acceptable_ratio = line_ratio

            # try also the split line
            for word in lower_split_line:
                word_ratio = difflib.SequenceMatcher(None, lower_item, word).ratio()
                if word_ratio > acceptable_ratio:
                    best_match = item
                    acceptable_ratio = word_ratio

    return best_match
    

def find_business(text):
    return find_best_match(text, common_store_names)

common_tax_documents = [
    "W-2", 
    "1099-INT", 
    "1099-MISC", 
    ] 
common_store_names = [
    "Walmart", 
    "Target", 
    "Best Buy", 
    "Home Depot", 
    "Lowe's", 
    "Costco",
    "Safeway",
    "Whole Foods",
    ]
try:
    with open(os.path.expanduser('~/.config/scan-it-all/config')) as config_file:
        config = json.load(config_file)
        common_store_names.extend(config['store_names'])
except FileNotFoundError:
    pass
def find_classification_and_entity(text):
    taxes_match = find_best_match(text, common_tax_documents)
    if taxes_match is not None:
        return ('Taxes', taxes_match)
    store_match = find_best_match(text, common_store_names)
    if store_match is not None:
        return ('Reciepts', store_match)
    raise NotImplemented

def add_padding(img, padding):
    # Convert the image to a numpy array
    arr = np.array(img)

    # Get original array dimensions
    rows, cols, _ = arr.shape

    # New numpy array with added space
    new_rows = rows + 2 * padding
    new_cols = cols + 2 * padding
    new_arr = np.zeros((new_rows, new_cols, 3), dtype=np.uint8)

    # Copy original array into new array
    new_arr[padding:new_rows-padding, padding:new_cols-padding, :] = arr

    # Convert the new array back to an image
    new_img = Image.fromarray(new_arr)

    return new_img


def crop_image(image):
    padded_image = add_padding(image, 64)
    
    optimized_image = get_scan_optimized_image(padded_image)
    np_optimized_image = np.array(optimized_image)
    kernel = np.ones((5,5),np.uint8)
    np_eroded_image = cv2.morphologyEx(np_optimized_image, cv2.MORPH_CLOSE, kernel, iterations= 3)

    edged = cv2.Canny(np_eroded_image, 30, 200)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Get the bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to the contour
    cropped_img = np.array(padded_image)[y:y + h, x:x + w]

    return Image.fromarray(cropped_img)


def scan_and_ocr(device):
    # Scan the image
    image = device.scan()

    # Remove any extra black from the scanned image
    image = crop_image(image)

    # Perform OCR on the image
    ocr_image = get_ocr_optimized_image(image)
    text = pytesseract.image_to_string(ocr_image)

    # Try to find a date/time
    datetime = find_datetime(text)

    # Try to find the classification, and entity
    classifcation, entity = find_classification_and_entity(text)

    # Return the extracted text
    return ScannedDocument(image, ocr_image, text, datetime, entity, classifcation)


def run_once(queue):
    sane.init()
    devices = sane.get_devices(True)
    device_name = devices[0][0]
    device = sane.open(device_name)
    device.mode = 'color'

    # Scan and perform OCR on the image
    scanned_doc = scan_and_ocr(device)

    # Saves both the image and the ocr text
    scanned_doc.save()

    device.close()
    sane.exit()

if __name__ == '__main__':
    user_input = ''
    while user_input != 'q':
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_once, args=(queue,))
        process.start()
        process.join()

        user_input = input('Enter \'q\' to stop: ')


