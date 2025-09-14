import cv2
import numpy as np
import pandas as pd
import easyocr
from pathlib import Path
import re
import sys

def find_plot_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("Contour detection failed. No shapes found.")
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, x + w, y + h

def preprocess(crop):
    h, w = crop.shape[:2]
    if h == 0 or w == 0: return None
    upscaled_crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def ocr_axis(crop, reader):
    processed_crop = preprocess(crop)
    if processed_crop is None: return []
    return reader.readtext(processed_crop)

def get_line(folder_name):
    color_palette = [
        (np.array([100, 150, 50]), np.array([140, 255, 255])),  
        (np.array([10, 150, 50]), np.array([25, 255, 255])),   
        (np.array([40, 100, 50]), np.array([80, 255, 255])),   
    ]
    line_names = [name.strip() for name in folder_name.split(' and ')]
    if not line_names: return {}
    colors_to_extract = {}
    for i, name in enumerate(line_names):
        if i < len(color_palette):
            colors_to_extract[name] = color_palette[i]
    return colors_to_extract

def parse_yaxis(results):
    numbers = []
    for (_, text, _) in results:
        try:
            cleaned_text = re.sub(r'[^\d.-]', '', text)
            if cleaned_text: numbers.append(float(cleaned_text))
        except ValueError: continue
    if not numbers: raise ValueError("OCR could not find any valid numbers on the Y-axis.")
    return min(numbers), max(numbers)

def parse_xaxis(results):
    years = []
    for (_, text, _) in results:
        match = re.search(r'(\b\d{4}\b)(-\d{2})?', text)
        if match:
            years.append(int(match.group(1)))
    if not years: raise ValueError("OCR could not find a year on the X-axis.")
    return max(set(years), key=years.count)


def extract(image_path, ocr_reader, colors_to_extract):
    try:
        image = cv2.imread(str(image_path))
        if image is None: raise FileNotFoundError(f"Image not loaded: {image_path}")
        px1, py1, px2, py2 = find_plot_area(image)

        y_axis_crop = image[py1:py2, max(0, px1 - 120):px1]
        x_axis_crop = image[py2:py2 + 80, px1:px2]

        y_min_val, y_max_val = parse_yaxis(ocr_axis(y_axis_crop, ocr_reader))
        year_ocr = parse_xaxis(ocr_axis(x_axis_crop, ocr_reader))
        start_date = pd.to_datetime(f'{year_ocr}-01-01')
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        all_results = {}
        num_days = 366 if pd.Timestamp(year=year_ocr, month=1, day=1).is_leap_year else 365
        daily_x_pixels = np.linspace(px1, px2, num_days)

        for line_name, (lower_bound, upper_bound) in colors_to_extract.items():
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            plot_area_mask = mask[py1:py2, px1:px2]
            line_pixels = np.where(plot_area_mask == 255)
            
            if len(line_pixels[0]) < 2: continue
                
            raw_pixel_y, raw_pixel_x = line_pixels[0] + py1, line_pixels[1] + px1
            sorted_indices = np.argsort(raw_pixel_x)
            pixel_x, pixel_y = raw_pixel_x[sorted_indices], raw_pixel_y[sorted_indices]
            daily_y_pixels = np.interp(daily_x_pixels, pixel_x, pixel_y)
            
            y_pixel_range = py2 - py1
            y_data_range = y_max_val - y_min_val
            final_y_values = y_min_val + ((py2 - daily_y_pixels) * y_data_range / y_pixel_range)
            all_results[line_name] = final_y_values

        if not all_results: return None
            
        dates = pd.date_range(start=start_date, periods=num_days)
        df = pd.DataFrame({'Date': dates})
        for line_name, data in all_results.items():
            df[line_name] = data
        return df

    except Exception as e:
        print(f"ERROR processing {image_path.name}: {e}")
        return None

def main():
    project_root = Path(__file__).parent.parent
    root_path = project_root / 'Data'
    if not root_path.is_dir():
        print(f"Error: Directory '{root_path}' not found.")
        sys.exit()
    
    print("Initializing OCR engine...")
    try:
        ocr_reader = easyocr.Reader(['en'])
        print("OCR engine loaded successfully.\n")
    except Exception as e:
        print(f"Fatal Error during initialization: {e}")
        return

    country_folders = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('C')]
    for country_folder in country_folders:
        print(f"--- Processing Country: {country_folder.name} ---")
        feature_folders = [d for d in country_folder.iterdir() if d.is_dir() and any(d.glob('*.png'))]
        for feature_folder in feature_folders:
            print(f"Processing Feature: {feature_folder.name}")
            colors_to_extract = get_line(feature_folder.name)
            if not colors_to_extract: continue
            
            image_files = sorted(list(feature_folder.glob('*.png')))
            all_years_data = []
            
            for image_file in image_files:
                print(f"    - Extracting from {image_file.name}...")
                yearly_df = extract(image_file, ocr_reader, colors_to_extract)
                if yearly_df is not None:
                    all_years_data.append(yearly_df)

            if all_years_data:
                combined_df = pd.concat(all_years_data, ignore_index=True)
                combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.strftime('%Y-%m-%d')
                output_filename = feature_folder / f"_extracted_data_2000-2024.csv"
                combined_df.to_csv(output_filename, index=False)
                print(f"SUCCESS: Saved combined data to {output_filename}\n")
            else:
                print(f"FAILED: No data could be extracted for feature {feature_folder.name}.\n")

    print("All folders processed.")

if __name__ == '__main__':
    main()