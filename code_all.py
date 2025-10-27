import cv2
import numpy as np
import os
from collections import defaultdict # For grouping by class
import math

# --- Configuration Parameters ---
ROOT_DIR = r"C:\Users\c0077\Downloads\reference_images(2)\reference_images"
SEMANTIC_FOLDER_NAME = 'camera_images_semantic_front'
REAL_FOLDER_NAME = 'camera_images_real_front'
LABEL_OUTPUT_FOLDER_NAME = 'yolo_labels_front'
VIZ_OUTPUT_FOLDER_NAME = 'visualized_bbox_front'

BACKGROUND_COLORS_BGR = {
    # Add more of your background colors here...
    # Carefully check your segmented images to ensure common background colors are listed
    #(192, 192, 192), # Potentially road/pavement
}

color_to_class_id = {}
next_class_id = 0

# --- Helper Functions ---

def find_unique_colors(image):
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return [tuple(color) for color in unique_colors]

def get_or_assign_class_id(color_bgr):
    global next_class_id, color_to_class_id, BACKGROUND_COLORS_BGR
    if color_bgr in BACKGROUND_COLORS_BGR: return None
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
    """Checks if two boxes (xmin, ymin, xmax, ymax) overlap."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # Condition for no overlap
    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return False
    return True

def merge_boxes(boxes_to_merge):
    """Merges a list of boxes into a single bounding box."""
    if not boxes_to_merge:
        return None
    min_x = min(b[0] for b in boxes_to_merge)
    min_y = min(b[1] for b in boxes_to_merge)
    max_x = max(b[2] for b in boxes_to_merge)
    max_y = max(b[3] for b in boxes_to_merge)
    return (min_x, min_y, max_x, max_y)

def merge_overlapping_boxes(initial_boxes):
    """
    Merges overlapping bounding boxes of the same class.
    Args:
        initial_boxes: List, where each element is (class_id, xmin, ymin, xmax, ymax).
    Returns:
        List, containing merged (class_id, xmin, ymin, xmax, ymax).
    """
    if not initial_boxes:
        return []

    # 1. Group by class_id
    boxes_by_class = defaultdict(list)
    for class_id, xmin, ymin, xmax, ymax in initial_boxes:
        boxes_by_class[class_id].append((xmin, ymin, xmax, ymax))

    final_merged_boxes = []

    # 2. Process each class
    for class_id, boxes in boxes_by_class.items():
        if len(boxes) <= 1:
            # If only one box or no boxes, add directly to the final list
            if boxes:
                final_merged_boxes.append((class_id, *boxes[0]))
            continue

        num_boxes = len(boxes)
        merged = [False] * num_boxes # Mark if a box has been merged into a group
        current_merged_for_class = []

        # 3. Merge using a connected components-like approach
        for i in range(num_boxes):
            if merged[i]:
                continue # This box has already been merged

            # Start a new merge group
            component_indices = {i}
            queue = [i] # Used for BFS to find overlapping boxes
            merged[i] = True

            while queue:
                current_idx = queue.pop(0)
                for j in range(num_boxes):
                    # Only check unmerged boxes that are not the box itself
                    if not merged[j] and i != j:
                        # Check if box current_idx and box j overlap
                        if calculate_overlap(boxes[current_idx], boxes[j]):
                            merged[j] = True
                            component_indices.add(j)
                            queue.append(j) # Add newly merged box to queue to continue search

            # Merge all boxes in this connected component
            boxes_in_component = [boxes[k] for k in component_indices]
            merged_box = merge_boxes(boxes_in_component)
            if merged_box:
                current_merged_for_class.append(merged_box)

        # Add merged boxes for this class to the final list
        for mb in current_merged_for_class:
            final_merged_boxes.append((class_id, *mb))

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
                if cv2.contourArea(contour) < 10: continue # Filter out very small noise contours
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0: continue
                # Record initial box in pixel coordinates
                initial_boxes_pixel.append((class_id, x, y, x + w, y + h))

        # ******** Perform bounding box merging here ********
        print(f"    Initially detected {len(initial_boxes_pixel)} boxes, starting merge...")
        merged_boxes_pixel = merge_overlapping_boxes(initial_boxes_pixel)
        print(f"    {len(merged_boxes_pixel)} boxes remaining after merge.")
        # ****************************************************

        # Convert final merged boxes to YOLO format
        yolo_labels_for_image = []
        for class_id, xmin, ymin, xmax, ymax in merged_boxes_pixel:
            w_box = xmax - xmin # Renamed to avoid conflict with img_width
            h_box = ymax - ymin # Renamed to avoid conflict with img_height
            center_x = (xmin + w_box / 2) / img_width
            center_y = (ymin + h_box / 2) / img_height
            norm_w = w_box / img_width
            norm_h = h_box / img_height

            # Add validity check
            if norm_w > 0 and norm_h > 0 and 0 <= center_x <= 1 and 0 <= center_y <= 1:
                 yolo_label = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
                 yolo_labels_for_image.append(yolo_label)
            else:
                 print(f"    Warning: Invalid YOLO box generated after merge (ID: {class_id}, cx:{center_x:.2f}, cy:{center_y:.2f}, w:{norm_w:.2f}, h:{norm_h:.2f}), skipped.")

        # Write YOLO label file
        if yolo_labels_for_image:
            with open(label_txt_path, 'w') as f:
                for label in yolo_labels_for_image:
                    f.write(label + "\n")
        else:
            # Create an empty file even if no labels were generated (e.g., all filtered out)
            open(label_txt_path, 'w').close()


        # Visualization (now uses merged_boxes_pixel directly for drawing)
        if real_img_path and os.path.exists(real_img_path):
            real_image = cv2.imread(real_img_path)
            if real_image is not None:
                class_id_to_color = {v: k for k, v in color_to_class_id.items()} # Ensure it's up-to-date
                try:
                    # Draw directly using merged_boxes_pixel to avoid file I/O and normalization/denormalization errors for viz
                    r_h, r_w = real_image.shape[:2]
                    for viz_idx, (class_id_read, x_min, y_min, x_max, y_max) in enumerate(merged_boxes_pixel):

                        # Check coordinate validity (though merge_boxes should handle it, just in case)
                        if x_min >= x_max or y_min >= y_max:
                            print(f"    Warning: Invalid merged box coordinates found during visualization (ID: {class_id_read}, Box: {(x_min, y_min, x_max, y_max)}), skipping draw.")
                            continue

                        box_color = class_id_to_color.get(class_id_read)
                        if box_color is None:
                             print(f"    Error (Visualization): Could not find color for ID {class_id_read}!")
                             box_color = (0, 255, 255) # Yellow as fallback
                        elif not isinstance(box_color, tuple) or len(box_color) != 3:
                             print(f"    Error (Visualization): Color '{box_color}' for ID {class_id_read} is invalid!")
                             box_color = (0, 0, 255) # Red as fallback

                        draw_color = tuple(map(int, box_color))

                        # Draw rectangle
                        cv2.rectangle(real_image, (x_min, y_min), (x_max, y_max), draw_color, 2)

                        # Draw text
                        label_text = f"ID: {class_id_read}"
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        bg_y_min = max(y_min - text_height - baseline, 0)
                        bg_x_max = min(x_min + text_width, r_w - 1)
                        # Determine text color (black or white) for contrast
                        text_color = (0,0,0) if np.mean(draw_color) > 127 else (255,255,255)
                        cv2.rectangle(real_image, (x_min, bg_y_min), (bg_x_max, y_min), draw_color, -1) # Filled background for text
                        cv2.putText(real_image, label_text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                    cv2.imwrite(viz_img_path, real_image)

                except Exception as e:
                    print(f"  Error: An error occurred during visualization for {filename}: {e}")
            else:
                 print(f"  Warning: Could not read real image {real_img_path} for visualization.")
        elif yolo_labels_for_image: # Only prompt if labels were generated
             print(f"  Warning: Real image {real_img_path} not found or could not be read, skipping visualization.")


# --- New function: Create Color Legend ---
def create_color_legend(color_map, filename="color_legend.png", square_size=50, padding=10):
    """
    Creates an image file showing color squares and their corresponding Class IDs.

    Args:
        color_map (dict): Dictionary in the format { (B, G, R)_tuple: class_id_int }.
        filename (str): Filename to save the legend image.
        square_size (int): Side length of each color square (pixels).
        padding (int): Spacing between squares and at the edges.
    """
    if not color_map:
        print("Color map is empty, cannot create legend.")
        return

    # Reverse the map and sort by Class ID
    class_id_to_color = {v: k for k, v in color_map.items()}
    sorted_ids = sorted(class_id_to_color.keys())
    num_colors = len(sorted_ids)

    if num_colors == 0:
        print("No valid class colors found, cannot create legend.")
        return

    # Calculate grid layout
    cols = int(math.ceil(math.sqrt(num_colors)))
    rows = int(math.ceil(num_colors / cols))

    # Calculate image dimensions
    img_width = cols * square_size + (cols + 1) * padding
    img_height = rows * square_size + (rows + 1) * padding

    # Create a white background image
    legend_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    print(f"\nCreating color legend image ({rows}x{cols} grid): {filename}")

    # Draw each color square and label
    for i, class_id in enumerate(sorted_ids):
        color_bgr = class_id_to_color[class_id]
        # Ensure color is a Python int tuple
        draw_color = tuple(map(int, color_bgr))

        # Calculate square position
        row_idx = i // cols
        col_idx = i % cols
        x1 = padding + col_idx * (square_size + padding)
        y1 = padding + row_idx * (square_size + padding)
        x2 = x1 + square_size
        y2 = y1 + square_size

        # Draw color square
        cv2.rectangle(legend_image, (x1, y1), (x2, y2), draw_color, -1) # -1 for filled

        # Prepare label text
        label_text = f"ID: {class_id}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Determine text color (black or white) for contrast
        text_color = (0, 0, 0) if np.mean(draw_color) > 127 else (255, 255, 255)

        # Calculate text size for centering (or appropriate placement)
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font_face, font_scale, thickness)

        # Position text near the center of the square
        text_x = x1 + (square_size - text_width) // 2
        text_y = y1 + (square_size + text_height) // 2 # Adjust slightly down for vertical centering

        # Draw text
        cv2.putText(legend_image, label_text, (text_x, text_y), font_face, font_scale, text_color, thickness)

    # Save the image
    try:
        cv2.imwrite(filename, legend_image)
        print(f"Color legend saved to: {filename}")
    except Exception as e:
        print(f"Error: Could not save color legend image {filename}: {e}")


# ... (All previous code) ...

print("\n--- Processing complete ---")
print("Final Color to Class ID Mapping:")

# Define the filename for the mapping
mapping_txt_filename = "color_class_id_mapping.txt"

if color_to_class_id:
    class_id_to_color_final = {v: k for k, v in color_to_class_id.items()}
    sorted_ids = sorted(class_id_to_color_final.keys())

    print(f"Saving color mapping to file: {mapping_txt_filename}")
    try:
        # Use 'with open' to automatically manage file closing
        # Specifying encoding='utf-8' is good practice for potential non-ASCII characters
        with open(mapping_txt_filename, 'w', encoding='utf-8') as f:
            # (Optional) Write a header line
            f.write("Color Class ID Mapping\n")
            f.write("========================\n")

            # Iterate through sorted IDs
            for i in sorted_ids:
                color = class_id_to_color_final.get(i, "Unknown Color")

                # Prepare color string for printing and writing
                if isinstance(color, tuple):
                     # Convert np.uint8 to regular int for printing
                     printable_color = tuple(map(int, color))
                     # Format output string
                     output_line = f"Class ID: {i} -> BGR Color: {printable_color}"
                else:
                     # If color is not a tuple (e.g., "Unknown Color" string)
                     printable_color = color
                     output_line = f"Class ID: {i} -> {printable_color}" # Or other desired format

                # 1. Print to console (maintaining original behavior)
                print(f"  {output_line}")

                # 2. Write to file (adding newline character '\n')
                f.write(output_line + "\n")

        print(f"Color mapping successfully saved to: {mapping_txt_filename}")

    except IOError as e:
        # Print error message if file writing fails
        print(f"Error: Could not write color mapping file {mapping_txt_filename}: {e}")
        print("Please check file permissions or path.")

    # --- Call the function to create the legend (remains unchanged) ---
    legend_filename = "color_class_id_legend.png"
    # Ensure the color_to_class_id dictionary is passed correctly
    create_color_legend(color_to_class_id, legend_filename)
    # --- End of legend function call ---

else:
    print("  No Class IDs were assigned.")
    # If no IDs were assigned, you can choose to create an empty mapping file or do nothing
    try:
        with open(mapping_txt_filename, 'w', encoding='utf-8') as f:
            f.write("No Class IDs were assigned.\n")
        print(f"Empty mapping file created: {mapping_txt_filename}")
    except IOError as e:
         print(f"Error: Could not write empty mapping file {mapping_txt_filename}: {e}")