import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def remove_specular_reflection(image):
    # Convert to float32
    image = image.astype(np.float32)
    # Compute per-pixel min across channels
    Imin = np.min(image, axis=2)
    
    # Threshold based on image stats
    mean_val = np.mean(Imin)
    std_val = np.std(Imin)
    T = mean_val + 0.5 * std_val  # η = 0.5

    # Offset map τ
    tau = np.where(Imin > T, Imin, 0)

    # Specular component β̂
    beta_hat = Imin - tau
    beta_hat = np.maximum(beta_hat, 0)

    # Subtract from each channel and merge
    fsf = np.zeros_like(image)
    for i in range(3):
        fsf[:, :, i] = image[:, :, i] - beta_hat

    fsf = np.clip(fsf, 0, 255).astype(np.uint8)
    return fsf

def detect_auv(image):
    # Step 1: Remove specular reflection
    specular_free = remove_specular_reflection(image)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(specular_free, cv2.COLOR_BGR2GRAY)

    # Step 3: Denoise (Gaussian blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Step 5: Morphology to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Step 6: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None

    # Step 7: Select largest contour
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 500:  # ignore too small
        return image, None

    # Step 8: Draw bounding box
    x, y, w, h = cv2.boundingRect(max_contour)
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output, (x, y, w, h)

# -------------- Run on folder of images ------------------

def process_folder(folder_path, output_path="output"):
    os.makedirs(output_path, exist_ok=True)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, fname)
            image = cv2.imread(img_path)

            output, bbox = detect_auv(image)
            if bbox:
                print(f"[✓] AUV detected in {fname} at {bbox}")
            else:
                print(f"[✗] No AUV detected in {fname}")
            cv2.imwrite(os.path.join(output_path, fname), output)

# Example: process_folder("images")
def process_folder(input_folder, output_folder):
    """
    Process all images in input_folder, detect AUVs, draw bounding boxes,
    and save results in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(supported_ext):
            img_path = os.path.join(input_folder, fname)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load {fname}")
                continue

            output, bbox = detect_auv(image)

            if bbox:
                print(f"[✓] AUV detected in {fname} at {bbox}")
            else:
                print(f"[✗] No AUV detected in {fname}")

            # Save image with bounding box
            cv2.imwrite(os.path.join(output_folder, fname), output)

if __name__ == "__main__":
    # Set your folders here
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input_images2")
    output_folder = os.path.join(current_dir, "output_images2")
    process_folder(input_folder, output_folder)