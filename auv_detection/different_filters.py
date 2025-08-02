import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
# img_path = os.path.join(current_dir, 'input_images2', '2.png')
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

folder_path = os.path.join(current_dir, 'input_images2')
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for file in image_files:
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {file}")
        continue

    # 1. SOBEL
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    # 2. PREWITT (using custom kernels)
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewitt_x = cv2.filter2D(img, -1, kernelx)
    prewitt_y = cv2.filter2D(img, -1, kernely)
    prewitt = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))

    # 3. ROBERTS (using scipy with 2x2 kernels)
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    roberts_x_img = ndimage.convolve(img, roberts_x)
    roberts_y_img = ndimage.convolve(img, roberts_y)
    roberts = np.sqrt(roberts_x_img**2 + roberts_y_img**2)

    # 4. CANNY
    canny = cv2.Canny(img, 100, 200)

    # 5. LAPLACIAN OF GAUSSIAN (LoG)
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    # # Plot results
    # titles = ['Original', 'Sobel', 'Prewitt', 'Roberts', 'Canny', 'LoG']
    # images = [img, sobel, prewitt, roberts, canny, log]

    # plt.figure(figsize=(12, 6))
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.title(titles[i])
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()



    # Plot results
    titles = ['Original', 'Sobel', 'Prewitt', 'Roberts', 'Canny', 'LoG']
    images = [img, sobel, prewitt, roberts, canny, log]

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle(file)
    plt.tight_layout()
    plt.show()
