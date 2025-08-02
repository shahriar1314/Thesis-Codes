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


def detect_rectangular_objects(image: np.ndarray, mask_size: int = 30,
                               min_aspect: float = 3.0, max_aspect: float = 5.0,
                               show_plot: bool = False) -> np.ndarray:
    """
    Remove wave patterns and detect rectangular objects with specific aspect ratio.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        mask_size (int): FFT low-pass filter size.
        min_aspect (float): Minimum aspect ratio (width / height).
        max_aspect (float): Maximum aspect ratio.
        show_plot (bool): Whether to show plot of results.

    Returns:
        np.ndarray: Image with detected rectangles drawn.
    """

    # --- STEP 1: Convert to grayscale ---
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- STEP 2: Remove wave patterns with FFT ---
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - mask_size:crow + mask_size, ccol - mask_size:ccol + mask_size] = 1
    f_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(f_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift)).astype(np.uint8)

    # --- STEP 3: Threshold and find contours ---
    _, thresh = cv2.threshold(img_back, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- STEP 4: Filter by rectangular aspect ratio ---
    outputx = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        if min_aspect <= aspect_ratio <= max_aspect:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- STEP 5: Optionally show results ---
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title("After FFT Filter")
        plt.imshow(img_back, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title("Thresholded")
        plt.imshow(thresh, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Detected Rectangles")
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.show()

    return outputx



def detect_auv(image):
    # Step 1: Remove specular reflection
    specular_free = remove_specular_reflection(image)
    plt.imshow(specular_free)
    plt.title("Specular removal")
    plt.show()

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(specular_free, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.title("Gray Scale Image")
    plt.show()

    # Step 3: Denoise (Gaussian blur)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = detect_rectangular_objects(gray, mask_size=20, show_plot=True)
    plt.imshow(blurred)
    plt.title("After object detection")
    plt.show()

    # Step 4: Edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    plt.imshow(edges)
    plt.title("Canny Edge Detection")
    plt.show()

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

def process_folder(input_folder):
    """
    Process all images in input_folder, detect AUVs, draw bounding boxes,
    and show results using matplotlib.
    """
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    images_to_plot = []
    titles = []

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

            # Convert BGR to RGB for matplotlib display
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            images_to_plot.append(output_rgb)
            titles.append(fname)

    # Plotting
    return
    num_images = len(images_to_plot)
    cols = 3
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))
    for i, img in enumerate(images_to_plot):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(titles[i], fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input_images2")
    process_folder(input_folder)
