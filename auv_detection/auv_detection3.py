import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def remove_specular_reflection(image):
    """Removes specular reflection and plots intermediate result."""
    image = image.astype(np.float32)
    Imin = np.min(image, axis=2)
    mean_val = np.mean(Imin)
    std_val = np.std(Imin)
    T = mean_val + 0.5 * std_val
    tau = np.where(Imin > T, Imin, 0)
    beta_hat = np.maximum(Imin - tau, 0)

    fsf = np.zeros_like(image)
    for i in range(3):
        fsf[:, :, i] = image[:, :, i] - beta_hat

    fsf = np.clip(fsf, 0, 255).astype(np.uint8)

    # Show after specular removal
    plt.imshow(cv2.cvtColor(fsf, cv2.COLOR_BGR2RGB))
    plt.title("After Specular Removal")
    plt.show()

    return fsf


def detect_rectangles(image, min_aspect=2.5, min_area=300, debug=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.show()

    plt.imshow(blurred, cmap='gray')
    plt.title("Blurred")
    plt.show()

    edges = cv2.Canny(blurred, 50, 150)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edges")
    plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    plt.imshow(edges, cmap='gray')
    plt.title("After Morphology")
    plt.show()

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        if aspect_ratio >= min_aspect:
            rectangles.append(rect)

    if debug:
        debug_img = image.copy()
        for rect in rectangles:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Rectangles Detected")
        plt.show()

    return rectangles


def detect_rectangles2(image, min_aspect=2.5, min_area=300, debug=True):
    # 1) Grayscale + blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if debug:
        plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.show()
        plt.imshow(blurred, cmap='gray'); plt.title("Blurred"); plt.show()

    # 2) Brightness filter: keep top 10% as-is, rest → 0
    thresh_val = np.percentile(blurred, 98.5)  # Top 1.5% as threshold
    bright_mask = blurred.copy()
    bright_mask[blurred < thresh_val] = 0

    if debug:
        plt.imshow(bright_mask, cmap='gray')
        plt.title(f"Top 10% Brightness Kept (thresh={thresh_val:.0f})")
        plt.show()

    # 2.1) Remove specular reflections from the bright mask
    bm_color = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
    bm_spec_free = remove_specular_reflection(bm_color)
    bright_mask = cv2.cvtColor(bm_spec_free, cv2.COLOR_BGR2GRAY)

    if debug:
        plt.imshow(bright_mask, cmap='gray')
        plt.title("After Specular Removal on Bright Mask")
        plt.show()

    # 2.2) Flatten noise by averaging neighbors (5×5 mean filter)
    bright_mask = cv2.blur(bright_mask, (5, 5))
    if debug:
        plt.imshow(bright_mask, cmap='gray')
        plt.title("After 5×5 Mean Filter")
        plt.show()

        
    # 3) Edge detection on that binary mask
    edges = cv2.Canny(bright_mask, 50, 150)
    if debug:
        plt.imshow(edges, cmap='gray'); plt.title("Canny on Bright Mask"); plt.show()

    # 4) Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if debug:
        plt.imshow(closed, cmap='gray'); plt.title("After Morphology"); plt.show()

    # 5) Find contours & filter by aspect ratio + area
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect >= min_aspect:
            rectangles.append(rect)

    # 6) Debug draw
    if debug:
        debug_img = image.copy()
        for rect in rectangles:
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Rectangles Detected")
        plt.show()

    return rectangles


def detect_auv(image, debug=True):
    specular_free = remove_specular_reflection(image)

    rectangles = detect_rectangles2(specular_free, min_aspect=3.0, min_area=500, debug=debug)

    output = image.copy()
    best_rect = None
    largest_area = 0

    for rect in rectangles:
        width, height = rect[1]
        area = width * height
        if area > largest_area:
            largest_area = area
            best_rect = rect

    if best_rect:
        box = cv2.boxPoints(best_rect)
        box = np.int0(box)
        cv2.drawContours(output, [box], 0, (0, 255, 0), 3)
        if debug:
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title("Detected AUV")
            plt.show()
        return output, box

    if debug:
        print("No suitable AUV detected.")
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("No Detection")
        plt.show()

    return output, None


def process_folder(input_folder, output_folder="output"):
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')  # Fixed indentation
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(supported_ext):
            img_path = os.path.join(input_folder, fname)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {fname}")
                continue
            output, bbox = detect_auv(image, debug=True)
            if bbox is not None:
                print(f"[✓] AUV detected in {fname} at {bbox}")
            else:
                print(f"[✗] No AUV detected in {fname}")
            

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input_images2")
    process_folder(input_folder)
