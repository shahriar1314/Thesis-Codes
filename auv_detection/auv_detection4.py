import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import radon


def remove_specular_reflection(image):
    image = image.astype(np.float32)
    Imin = np.min(image, axis=2)
    mean_val, std_val = np.mean(Imin), np.std(Imin)
    T = mean_val + 0.5 * std_val
    tau = np.where(Imin > T, Imin, 0)
    beta_hat = np.maximum(Imin - tau, 0)
    fsf = np.clip(image - beta_hat[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return fsf


def adaptive_directional_fft_filter(gray, width=15):
    # FFT and shift
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Radon transform to find dominant angle
    spectrum_norm = (magnitude_spectrum - magnitude_spectrum.min()) / magnitude_spectrum.ptp()
    theta = np.linspace(0., 180., max(gray.shape), endpoint=False)
    sinogram = radon(spectrum_norm, theta=theta, circle=False)
    dominant_angle = theta[np.argmax(np.sum(sinogram, axis=0))]

    # Directional mask creation
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    for u in range(rows):
        for v in range(cols):
            angle_uv = np.degrees(np.arctan2(u - crow, v - ccol)) % 180
            if dominant_angle - width < angle_uv < dominant_angle + width:
                mask[u, v] = 0

    # Apply mask and inverse FFT
    f_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered))).astype(np.uint8)

    return img_back


def detect_auv(image):
    specular_free = remove_specular_reflection(image)
    plt.imshow(cv2.cvtColor(specular_free, cv2.COLOR_BGR2RGB))
    plt.title("Specular Removal")
    plt.show()

    gray = cv2.cvtColor(specular_free, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.show()

    filtered = adaptive_directional_fft_filter(gray)
    plt.imshow(filtered, cmap='gray')
    plt.title("Adaptive FFT Directional Filter")
    plt.show()

    # Thresholding
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholded")
    plt.show()

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None

    # Largest contour (assuming AUV)
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 500:
        return image, None

    # Bounding box
    x, y, w, h = cv2.boundingRect(max_contour)
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Detected AUV")
    plt.show()

    return output, (x, y, w, h)


def process_folder(input_folder):
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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input_images2")
    process_folder(input_folder)
