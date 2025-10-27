import cv2
import numpy as np
import os
from collections import defaultdict # For grouping by class
import math

# --- Configuration Parameters ---
ROOT_DIR = r"C:\Users\c0077\Downloads\COMP4423_GP"
SEMANTIC_FOLDER_NAME = 'camera_images_semantic_front'
REAL_FOLDER_NAME = 'camera_images_real_front'
LABEL_OUTPUT_FOLDER_NAME = 'yolo_labels_front'
VIZ_OUTPUT_FOLDER_NAME = 'visualized_bbox_front'

# Minimum area threshold
# Only final merged bounding boxes with a pixel area greater than or equal to this value will be kept.
MIN_AREA_THRESHOLD = 0.003 # Unit: normalized area (you can adjust this value as needed)

BACKGROUND_COLORS_BGR = {
    # Add more of your background colors here...
    # Carefully check your segmented images to ensure common background colors are listed.
    # (192, 192, 192), # Possibly road/pavement
    (50, 234, 157), # Road sign
    (128, 64, 128), # Road
    (80, 90, 55), # Mountain/Hill
    (180, 130, 70), # Sky
    (232, 35, 244), # Sidewalk/Pedestrian path
    (152, 251, 152), # Grass/Vegetation
    (81, 0, 81), # Gutter/Drain
}

color_to_class_id = {}
next_class_id = 0

# --- Helper Functions ---
# (find_unique_colors, get_or_assign_class_id, denormalize_yolo,
#  calculate_overlap, merge_boxes, merge_overlapping_boxes functions remain unchanged in logic,
#  only their comments/prints will be translated if they had any originally in Chinese)

def find_unique_colors(image):
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return [tuple(color) for color in unique_colors]

def get_or_assign_class_id(color_bgr):
    global next_class_id, color_to_class_id, BACKGROUND_COLORS_BGR
    if color_bgr in BACKGROUND_COLORS_BGR:
        return None
    if color_bgr not in color_to_class_id:
        color_to_class_id[color_bgr] = next_class_id
        print(f"New object color found: {color_bgr} -> Assigned Class ID: {next_class_id}")
        next_class_id += 1
    return color_to_class_id[color_bgr]

def denormalize_yolo(x_center, y_center, w, h, img_width, img_height):
    abs_w = w * img_width
    abs_h = h * img_height
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height
    x_min = int(abs_x_center - abs_w / 2)
    y_min = int(abs_y_center - abs_h / 2)
    x_max = int(abs_x_center + abs_w / 2)
    y_max = int(abs_y_center + abs_h / 2)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width - 1, x_max)
    y_max = min(img_height - 1, y_max)
    return x_min, y_min, x_max, y_max

def calculate_overlap(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return False
    return True

def merge_boxes(boxes_to_merge):
    if not boxes_to_merge: return None
    min_x = min(b[0] for b in boxes_to_merge)
    min_y = min(b[1] for b in boxes_to_merge)
    max_x = max(b[2] for b in boxes_to_merge)
    max_y = max(b[3] for b in boxes_to_merge)
    return (min_x, min_y, max_x, max_y)

def merge_overlapping_boxes(initial_boxes):
    if not initial_boxes: return []
    boxes_by_class = defaultdict(list)
    for class_id, xmin, ymin, xmax, ymax in initial_boxes:
        boxes_by_class[class_id].append((xmin, ymin, xmax, ymax))

    final_merged_boxes = []
    for class_id, boxes in boxes_by_class.items():
        if not boxes: continue # Added check just in case
        if len(boxes) <= 1:
            final_merged_boxes.append((class_id, *boxes[0]))
            continue

        num_boxes = len(boxes)
        merged = [False] * num_boxes
        for i in range(num_boxes):
            if merged[i]: continue
            component_indices = {i}
            queue = [i]
            merged[i] = True
            while queue:
                current_idx = queue.pop(0)
                for j in range(num_boxes):
                    if not merged[j] and i != j:
                        if calculate_overlap(boxes[current_idx], boxes[j]):
                            merged[j] = True
                            component_indices.add(j)
                            queue.append(j)
            boxes_in_component = [boxes[k] for k in component_indices]
            merged_box = merge_boxes(boxes_in_component)
            if merged_box:
                final_merged_boxes.append((class_id, *merged_box))
    return final_merged_boxes


# --- Main Processing Logic ---
print(f"Starting to process directory: {ROOT_DIR}")
# ... (Previous directory traversal and folder check logic remains unchanged) ...
for path_name in os.listdir(ROOT_DIR):
    path_dir = os.path.join(ROOT_DIR, path_name)
    if not os.path.isdir(path_dir): continue
    print(f"\nProcessing path: {path_name}")

    semantic_dir = os.path.join(path_dir, SEMANTIC_FOLDER_NAME)
    real_dir = os.path.join(path_dir, REAL_FOLDER_NAME)
    label_output_dir = os.path.join(path_dir, LABEL_OUTPUT_FOLDER_NAME)
    viz_output_dir = os.path.join(path_dir, VIZ_OUTPUT_FOLDER_NAME)

    if not os.path.exists(semantic_dir): continue # Simply skip
    # Ensure output directories exist
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)

    for filename in os.listdir(semantic_dir):
        if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
            continue

        semantic_img_path = os.path.join(semantic_dir, filename)
        base_name = os.path.splitext(filename)[0]
        label_txt_path = os.path.join(label_output_dir, f"{base_name}.txt")
        real_img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_real_path = os.path.join(real_dir, base_name + ext)
            if os.path.exists(potential_real_path):
                real_img_path = potential_real_path
                break
        if real_img_path is None and os.path.exists(os.path.join(real_dir, filename)):
             real_img_path = os.path.join(real_dir, filename) # Fallback

        viz_img_path = os.path.join(viz_output_dir, base_name + '.png')

        print(f"  Processing image: {filename}")

        semantic_image = cv2.imread(semantic_img_path)
        if semantic_image is None: continue

        img_height, img_width = semantic_image.shape[:2]
        initial_boxes_pixel = [] # Store initial (class_id, xmin, ymin, xmax, ymax)

        unique_colors_in_image = find_unique_colors(semantic_image)

        for color_bgr in unique_colors_in_image:
            class_id = get_or_assign_class_id(color_bgr)
            if class_id is None: continue

            lower_bound = np.array(color_bgr, dtype=np.uint8)
            upper_bound = np.array(color_bgr, dtype=np.uint8)
            mask = cv2.inRange(semantic_image, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Area filtering can also be added here to reduce the number of initial boxes
                contour_area = cv2.contourArea(contour)
                if contour_area < 10: continue # Filter out very small noise contours

                x, y, w_box, h_box = cv2.boundingRect(contour) # Renamed w, h to w_box, h_box
                if w_box <= 0 or h_box <= 0: continue
                # Record initial box in pixel coordinates
                initial_boxes_pixel.append((class_id, x, y, x + w_box, y + h_box))

        # ******** Perform bounding box merging here ********
        print(f"    Initially detected {len(initial_boxes_pixel)} boxes, starting merge...")
        merged_boxes_pixel = merge_overlapping_boxes(initial_boxes_pixel)
        print(f"    {len(merged_boxes_pixel)} boxes remaining after merge.")
        # ****************************************************

        # Convert final merged boxes to YOLO format and perform area filtering
        yolo_labels_for_image = []
        valid_merged_boxes_for_viz = [] # Store boxes that pass area filtering, for visualization
        skipped_small_boxes = 0

        for class_id, xmin, ymin, xmax, ymax in merged_boxes_pixel:
            w_box = xmax - xmin # Renamed w, h to w_box, h_box
            h_box = ymax - ymin

            center_x = (xmin + w_box / 2) / img_width
            center_y = (ymin + h_box / 2) / img_height
            norm_w = w_box / img_width
            norm_h = h_box / img_height

            # ******** Perform area filtering here ********
            area = norm_w * norm_h
            if area < MIN_AREA_THRESHOLD:
                skipped_small_boxes += 1
                continue # Skip boxes that are too small
            # ******************************************

            # Add validity check
            if norm_w > 0 and norm_h > 0 and 0 <= center_x <= 1 and 0 <= center_y <= 1:
                 yolo_label = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
                 yolo_labels_for_image.append(yolo_label)
                 # Record boxes that pass all checks for later visualization
                 valid_merged_boxes_for_viz.append((class_id, xmin, ymin, xmax, ymax))
            else:
                 print(f"    Warning: Invalid YOLO box generated after merge (ID: {class_id}, cx:{center_x:.2f}, cy:{center_y:.2f}, w:{norm_w:.2f}, h:{norm_h:.2f}), skipped.")

        if skipped_small_boxes > 0:
            print(f"    Skipped {skipped_small_boxes} boxes based on minimum area threshold ({MIN_AREA_THRESHOLD} normalized area).")

        # Write YOLO label file
        if yolo_labels_for_image:
            with open(label_txt_path, 'w') as f:
                for label in yolo_labels_for_image:
                    f.write(label + "\n")
        else:
            # Create an empty file even if the list is empty due to area filtering
            open(label_txt_path, 'w').close()
            if len(merged_boxes_pixel) > 0 and skipped_small_boxes == len(merged_boxes_pixel):
                 print(f"    All merged boxes were skipped due to being smaller than the area threshold.")


        # Visualization (now only draws boxes that passed area filtering)
        if real_img_path and os.path.exists(real_img_path):
            real_image = cv2.imread(real_img_path)
            if real_image is not None:
                # Ensure class_id_to_color is up-to-date
                class_id_to_color = {v: k for k, v in color_to_class_id.items()}
                try:
                    r_h, r_w = real_image.shape[:2]
                    # ******** Only iterate through boxes that passed area filtering ********
                    for viz_idx, (class_id_read, x_min, y_min, x_max, y_max) in enumerate(valid_merged_boxes_for_viz):

                        # (Coordinate validity check here can be omitted as YOLO conversion implicitly checks w>0, h>0)
                        # if x_min >= x_max or y_min >= y_max: continue

                        box_color = class_id_to_color.get(class_id_read)
                        if box_color is None:
                             # print(f"    Error (Visualization): Could not find color for ID {class_id_read}!")
                             box_color = (0, 255, 255) # Yellow as fallback
                        elif not isinstance(box_color, tuple) or len(box_color) != 3:
                             # print(f"    Error (Visualization): Color '{box_color}' for ID {class_id_read} is invalid!")
                             box_color = (0, 0, 255) # Red as fallback

                        draw_color = tuple(map(int, box_color))

                        # Draw rectangle
                        cv2.rectangle(real_image, (x_min, y_min), (x_max, y_max), draw_color, 2)

                        # Draw text
                        label_text = f"ID: {class_id_read}"
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        bg_y_min = max(y_min - text_height - baseline, 0)
                        bg_x_max = min(x_min + text_width, r_w - 1)
                        text_color = (0,0,0) if np.mean(draw_color) > 127 else (255,255,255) # Black or white text for contrast
                        cv2.rectangle(real_image, (x_min, bg_y_min), (bg_x_max, y_min), draw_color, -1) # Filled background for text
                        cv2.putText(real_image, label_text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                    cv2.imwrite(viz_img_path, real_image)

                except Exception as e:
                    print(f"  Error: An error occurred during visualization for {filename}: {e}")
            else:
                 print(f"  Warning: Could not read real image {real_img_path} for visualization.")
        # elif yolo_labels_for_image: # Changed to check if there are valid boxes for visualization
        elif valid_merged_boxes_for_viz:
             print(f"  Warning: Real image {real_img_path} not found or could not be read, skipping visualization.")


# --- Code for creating color legend and saving mapping file ---
# (create_color_legend function remains unchanged in logic)
def create_color_legend(color_map, filename="color_legend.png", square_size=50, padding=10):
    if not color_map:
        print("Color map is empty, cannot create legend.")
        return
    class_id_to_color = {v: k for k, v in color_map.items()}
    sorted_ids = sorted(class_id_to_color.keys())
    num_colors = len(sorted_ids)
    if num_colors == 0:
        print("No class IDs found, cannot create legend.") # Changed from "return" to print a message
        return
    cols = int(math.ceil(math.sqrt(num_colors)))
    rows = int(math.ceil(num_colors / cols))
    img_width = cols * square_size + (cols + 1) * padding
    img_height = rows * square_size + (rows + 1) * padding
    legend_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    print(f"\nCreating color legend image ({rows}x{cols} grid): {filename}")
    for i, class_id in enumerate(sorted_ids):
        color_bgr = class_id_to_color[class_id]
        draw_color = tuple(map(int, color_bgr))
        row_idx = i // cols
        col_idx = i % cols
        x1 = padding + col_idx * (square_size + padding)
        y1 = padding + row_idx * (square_size + padding)
        x2 = x1 + square_size
        y2 = y1 + square_size
        cv2.rectangle(legend_image, (x1, y1), (x2, y2), draw_color, -1)
        label_text = f"ID: {class_id}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (0, 0, 0) if np.mean(draw_color) > 127 else (255, 255, 255)
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font_face, font_scale, thickness)
        text_x = x1 + (square_size - text_width) // 2
        text_y = y1 + (square_size + text_height) // 2
        cv2.putText(legend_image, label_text, (text_x, text_y), font_face, font_scale, text_color, thickness)
    try:
        cv2.imwrite(filename, legend_image)
        print(f"Color legend saved to: {filename}")
    except Exception as e:
        print(f"Error: Could not save color legend image {filename}: {e}")


print("\n--- Processing complete ---")
print("Final Color to Class ID Mapping:")
mapping_txt_filename = "color_class_id_mapping.txt"
if color_to_class_id:
    class_id_to_color_final = {v: k for k, v in color_to_class_id.items()}
    sorted_ids = sorted(class_id_to_color_final.keys())
    print(f"Saving color mapping to file: {mapping_txt_filename}")
    try:
        with open(mapping_txt_filename, 'w', encoding='utf-8') as f:
            f.write("Color Class ID Mapping\n")
            f.write("========================\n")
            for i in sorted_ids:
                color = class_id_to_color_final.get(i, "Unknown Color")
                if isinstance(color, tuple):
                     printable_color = tuple(map(int, color))
                     output_line = f"Class ID: {i} -> BGR Color: {printable_color}"
                else:
                     printable_color = color
                     output_line = f"Class ID: {i} -> {printable_color}"
                print(f"  {output_line}")
                f.write(output_line + "\n")
        print(f"Color mapping successfully saved to: {mapping_txt_filename}")
    except IOError as e:
        print(f"Error: Could not write color mapping file {mapping_txt_filename}: {e}")
        print("Please check file permissions or path.")
    legend_filename = "color_class_id_legend.png"
    create_color_legend(color_to_class_id, legend_filename)
else:
    print("  No Class IDs were assigned.")
    try:
        with open(mapping_txt_filename, 'w', encoding='utf-8') as f:
            f.write("No Class IDs were assigned.\n")
        print(f"Empty mapping file created: {mapping_txt_filename}")
    except IOError as e:
         print(f"Error: Could not write empty mapping file {mapping_txt_filename}: {e}")