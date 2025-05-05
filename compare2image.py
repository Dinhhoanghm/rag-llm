import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pytesseract


def compare_images(image1_path, image2_path, labels=("Original Image", "Modified Image")):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        raise ValueError("Error loading images. Check file paths.")

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    (score, diff) = ssim(gray1, gray2, full=True)

    diff = (diff * 255).astype(np.uint8)

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diff_image = image1.copy().astype(np.uint8)

    abs_diff = cv2.absdiff(image1, image2)

    mask = np.any(abs_diff > 30, axis=2).astype(np.uint8) * 255

    diff_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    difference_regions = []

    composite = (image1.copy() * 0.5).astype(np.uint8)

    try:
        text1 = pytesseract.image_to_data(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB),
                                          output_type=pytesseract.Output.DICT)
        text2 = pytesseract.image_to_data(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB),
                                          output_type=pytesseract.Output.DICT)
        has_text_detection = True
    except:
        has_text_detection = False
        print("Warning: Text detection not available. Install pytesseract and configure it properly.")

    for i, contour in enumerate(diff_contours):
        x, y, w, h = cv2.boundingRect(contour)

        if w * h < 25:
            continue

        center_x, center_y = x + w // 2, y + h // 2

        region1 = image1[y:y + h, x:x + w].copy()
        region2 = image2[y:y + h, x:x + w].copy()

        avg_color1_bgr = np.mean(region1, axis=(0, 1)).astype(int)
        avg_color2_bgr = np.mean(region2, axis=(0, 1)).astype(int)

        color1 = get_color_name(avg_color1_bgr)
        color2 = get_color_name(avg_color2_bgr)

        region_text1 = ""
        region_text2 = ""

        if has_text_detection:
            region_text1 = extract_text_in_region(text1, x, y, w, h)
            region_text2 = extract_text_in_region(text2, x, y, w, h)

        cv2.rectangle(diff_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(diff_image, f"#{i + 1}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        color_changed = color1 != color2

        text_changed = region_text1 != region_text2 and (region_text1 != "" or region_text2 != "")

        region_diff = cv2.absdiff(region1, region2)
        mean_diff = np.mean(region_diff)

        region_mask = np.any(region_diff > 30, axis=2)

        is_old_brighter = (np.mean(region1, axis=2) > np.mean(region2, axis=2))

        region_composite = (region1.copy() * 0.5).astype(np.uint8)

        for ry in range(h):
            for rx in range(w):
                if region_mask[ry, rx]:
                    if is_old_brighter[ry, rx]:
                        region_composite[ry, rx] = [0, 0, 255]
                    else:
                        region_composite[ry, rx] = [0, 255, 0]

        composite[y:y + h, x:x + w] = region_composite

        if np.mean(avg_color1_bgr) > np.mean(avg_color2_bgr):
            diff_type = "Removed"
        else:
            diff_type = "Added"

        if text_changed:
            change_type = "Text Change"
            change_description = get_text_change_description(region_text1, region_text2)
        elif color_changed:
            change_type = "Color Change"
            change_description = f"Color changed from {color1} to {color2}"
        else:
            change_type = "Content Change"
            change_description = f"{diff_type} content in this region"

        difference_regions.append({
            'id': i + 1,
            'position': (x, y, w, h),
            'area': w * h,
            'change_type': change_type,
            'change_description': change_description,
            'old_color': color1,
            'new_color': color2,
            'old_text': region_text1,
            'new_text': region_text2,
            'difference_type': diff_type
        })

    total_pixels = image1.shape[0] * image1.shape[1]
    diff_pixels = np.sum(mask > 0)
    diff_percentage = (diff_pixels / total_pixels) * 100

    rgb_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    rgb_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    rgb_diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)
    rgb_composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    rgb_abs_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2RGB)

    return {
        'image1': rgb_image1,
        'image2': rgb_image2,
        'diff_image': rgb_diff_image,
        'composite': rgb_composite,
        'abs_diff': rgb_abs_diff,
        'ssim_score': score,
        'diff_pixels': diff_pixels,
        'diff_percentage': diff_percentage,
        'difference_regions': difference_regions,
        'labels': labels,
        'has_text_detection': has_text_detection
    }


def get_color_name(bgr_color):
    r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]

    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Gray": (128, 128, 128),
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Brown": (165, 42, 42),
        "Pink": (255, 192, 203),
        "Lime": (50, 205, 50),
        "Navy": (0, 0, 128),
        "Teal": (0, 128, 128)
    }

    min_distance = float('inf')
    nearest_color = "Unknown"

    for color_name, (rc, gc, bc) in colors.items():
        distance = (r - rc) ** 2 + (g - gc) ** 2 + (b - bc) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest_color = color_name

    if abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20:
        if r < 30:
            return "Black"
        elif r > 220:
            return "White"
        elif r < 85:
            return "Dark Gray"
        elif r < 170:
            return "Gray"
        else:
            return "Light Gray"

    return nearest_color


def extract_text_in_region(tesseract_data, x, y, w, h):
    text = []
    n_boxes = len(tesseract_data['text'])

    for i in range(n_boxes):
        if int(tesseract_data['conf'][i]) < 0 or not tesseract_data['text'][i].strip():
            continue

        x1 = tesseract_data['left'][i]
        y1 = tesseract_data['top'][i]
        w1 = tesseract_data['width'][i]
        h1 = tesseract_data['height'][i]

        if (x1 >= x - 5 and y1 >= y - 5 and
                x1 + w1 <= x + w + 5 and y1 + h1 <= y + h + 5):
            text.append(tesseract_data['text'][i])

    return " ".join(text).strip()


def get_text_change_description(old_text, new_text):
    if not old_text and new_text:
        return f"Text added: '{new_text}'"
    elif old_text and not new_text:
        return f"Text removed: '{old_text}'"
    else:
        return f"Text changed from '{old_text}' to '{new_text}'"


def display_results(results):
    plt.figure(figsize=(18, 14))

    plt.subplot(2, 3, 1)
    plt.title(results['labels'][0])
    plt.imshow(results['image1'])
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(results['labels'][1])
    plt.imshow(results['image2'])
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Differences Overview')
    plt.imshow(results['abs_diff'])
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Changes Identified (with ID)')
    plt.imshow(results['diff_image'])
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Old vs New (Red=Removed, Green=Added)')
    plt.imshow(results['composite'])
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Change Details')
    plt.axis('off')

    y_pos = 0.95
    plt.text(0.05, y_pos, f"Overall Similarity: {results['ssim_score']:.2f} (0-1 scale)", fontsize=10)
    y_pos -= 0.04
    plt.text(0.05, y_pos, f"Changed Area: {results['diff_percentage']:.1f}% of image", fontsize=10)
    y_pos -= 0.05

    if not results.get('has_text_detection', False):
        plt.text(0.05, y_pos, "Note: Text detection not available.", fontsize=9, style='italic')
        plt.text(0.05, y_pos - 0.03, "Install pytesseract for text comparison.", fontsize=9, style='italic')
        y_pos -= 0.08

    if len(results['difference_regions']) > 0:
        plt.text(0.05, y_pos, f"Changes Found: {len(results['difference_regions'])}", fontsize=10, weight='bold')
        y_pos -= 0.04

        for region in sorted(results['difference_regions'], key=lambda x: x['id']):
            plt.text(0.05, y_pos, f"Change #{region['id']} - {region['change_type']}", fontsize=9, weight='bold')
            y_pos -= 0.03

            plt.text(0.1, y_pos, f"{region['change_description']}", fontsize=8)
            y_pos -= 0.03

            if region['change_type'] == "Text Change" and (region['old_text'] or region['new_text']):
                if region['old_text'] and region['new_text']:
                    plt.text(0.1, y_pos, f"From: '{region['old_text']}'", fontsize=8)
                    y_pos -= 0.025
                    plt.text(0.1, y_pos, f"To: '{region['new_text']}'", fontsize=8)
                    y_pos -= 0.025

            if y_pos < 0.1:
                plt.text(0.05, y_pos, "... more changes not shown ...", fontsize=8, style='italic')
                break
    else:
        plt.text(0.05, y_pos, "No significant changes detected", fontsize=10)

    plt.tight_layout()
    plt.show()


def generate_report(results, output_file="change_report.txt"):
    with open(output_file, 'w') as f:
        f.write("IMAGE CHANGE REPORT\n")
        f.write("==================\n\n")

        f.write(f"Comparing:\n")
        f.write(f"  - {results['labels'][0]}\n")
        f.write(f"  - {results['labels'][1]}\n\n")

        f.write(f"Overall Similarity: {results['ssim_score']:.2f} (0-1 scale, higher means more similar)\n")
        f.write(f"Changed Area: {results['diff_percentage']:.1f}% of the image\n\n")

        if len(results['difference_regions']) > 0:
            f.write(f"CONTENT CHANGES DETECTED ({len(results['difference_regions'])} total)\n")
            f.write("------------------------\n\n")

            text_changes = []
            color_changes = []
            content_changes = []

            for region in results['difference_regions']:
                if region['change_type'] == "Text Change":
                    text_changes.append(region)
                elif region['change_type'] == "Color Change":
                    color_changes.append(region)
                else:
                    content_changes.append(region)

            if text_changes:
                f.write(f"TEXT CHANGES ({len(text_changes)} found):\n")
                for i, change in enumerate(text_changes, 1):
                    f.write(f"  {i}. {change['change_description']}\n")
                    x, y, w, h = change['position']
                    f.write(f"     Location: x={x}, y={y}, width={w}, height={h}\n")
                f.write("\n")

            if color_changes:
                f.write(f"COLOR CHANGES ({len(color_changes)} found):\n")
                for i, change in enumerate(color_changes, 1):
                    f.write(f"  {i}. {change['change_description']}\n")
                    x, y, w, h = change['position']
                    f.write(f"     Location: x={x}, y={y}, width={w}, height={h}\n")
                f.write("\n")

            if content_changes:
                f.write(f"OTHER CONTENT CHANGES ({len(content_changes)} found):\n")
                for i, change in enumerate(content_changes, 1):
                    f.write(f"  {i}. {change['change_description']}\n")
                    x, y, w, h = change['position']
                    f.write(f"     Location: x={x}, y={y}, width={w}, height={h}\n")
                f.write("\n")
        else:
            f.write("No significant content changes detected.\n")

    print(f"Report saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare two images and identify content changes')
    parser.add_argument('image1', help='Path to the first image')
    parser.add_argument('image2', help='Path to the second image')
    parser.add_argument('--old-label', default='Original Image', help='Label for the first image')
    parser.add_argument('--new-label', default='Modified Image', help='Label for the second image')
    parser.add_argument('--report', default='change_report.txt', help='Output file for the text report')
    parser.add_argument('--save-images', action='store_true', help='Save output images')
    parser.add_argument('--output-dir', default='.', help='Directory to save output images')
    parser.add_argument('--threshold', type=int, default=30,
                        help='Threshold for detecting differences (0-255, higher = less sensitive)')

    args = parser.parse_args()

    try:
        results = compare_images(args.image1, args.image2, (args.old_label, args.new_label))

        display_results(results)

        generate_report(results, args.report)

        print(f"\nImage Comparison Summary:")
        print(f"Overall Similarity: {results['ssim_score']:.2f} (0-1 scale)")
        print(f"Changed Area: {results['diff_percentage']:.1f}% of the image")
        print(f"Number of distinct changes: {len(results['difference_regions'])}")

        if len(results['difference_regions']) > 0:
            print("\nChanges overview:")
            for region in sorted(results['difference_regions'], key=lambda x: x['id']):
                print(f"Change #{region['id']} - {region['change_type']}: {region['change_description']}")

        if args.save_images:
            import os
            os.makedirs(args.output_dir, exist_ok=True)

            cv2.imwrite(os.path.join(args.output_dir, 'changes_highlighted.jpg'),
                        cv2.cvtColor(results['diff_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.output_dir, 'old_vs_new.jpg'),
                        cv2.cvtColor(results['composite'], cv2.COLOR_RGB2BGR))

            print(f"\nOutput images saved to {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()