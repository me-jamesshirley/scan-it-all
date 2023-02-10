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
    def __init__(self, image, ocr_image, ocr_text, date, business, classifcation, base_path='~/Documents'):
        self.image = image
        self.ocr_image = ocr_image
        self.ocr_text = ocr_text
        self.date = date
        self.business = business
        self.classifcation = classifcation
        self.base_path = os.path.expanduser(base_path)

    def save(self):
        path = self.base_path + '/' + self.classifcation + '/' + str(self.date.date()) + '/' + self.business + '/'

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

class NoBusinessFound(Exception):
    pass

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

def spell_check_store_name(word):
    for business_name in common_store_names:
        if difflib.SequenceMatcher(None, business_name.lower(), word.lower()).ratio() > .6:
            return business_name

    return None

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

def find_business(text):
    business = None
    for line in text.splitlines():
        business = spell_check_store_name(line)
        if business is not None:
            return business
    raise NoBusinessFound
    
def find_classification(text):
    return 'Reciepts'

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
    image = add_padding(image, 64)
    
    optimized_image = get_scan_optimized_image(image)
    np_optimized_image = np.array(optimized_image)
    
    edged = cv2.Canny(np_optimized_image, 30, 200)

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
    cropped_img = np.array(image)[y:y + h, x:x + w]

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

    # Try to find a business
    business = find_business(text)

    # Try to find what type of document this is
    classifcation = find_classification(text)

    # Return the extracted text
    return ScannedDocument(image, ocr_image, text, datetime, business, classifcation)


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

