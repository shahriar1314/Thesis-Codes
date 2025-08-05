#!/usr/bin/env python3
import os
import glob
import argparse
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

# import your parameters
import auv_detector.params_detector_2 as P


def remove_specular_reflection(image: np.ndarray) -> np.ndarray:
    """Remove specular highlights from a BGR image."""
    img = image.astype(np.float32)
    Imin = np.min(img, axis=2)
    T = Imin.mean() + 0.5 * Imin.std()
    tau = np.where(Imin > T, Imin, 0)
    beta = np.maximum(Imin - tau, 0)
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = img[:, :, c] - beta
    return np.clip(out, 0, 255).astype(np.uint8)


def get_color_pct(mask, hsv_image, rgb_color, tol_hsv):
    """
    Return percentage of pixels in `mask` matching the given RGB color
    (converted to HSV and widened by tol_hsv).
    """
    hsv_val = cv2.cvtColor(
        np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
    lower = np.clip(hsv_val - tol_hsv, 0, 255)
    upper = np.clip(hsv_val + tol_hsv, 0, 255)
    sub = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    m = cv2.inRange(sub, lower, upper)
    return 100.0 * np.count_nonzero(m) / (np.count_nonzero(mask) + 1e-6)


def process_image(img_bgr: np.ndarray, backSub, debug: bool):
    """
    Run the full detection pipeline on one BGR image using your P.* params.
    Returns the annotated BGR output.
    """
    # --- 1) Specular removal ---
    spec_free = remove_specular_reflection(img_bgr)

    # --- 2) Convert to HSV for color tests ---
    hsv = cv2.cvtColor(spec_free, cv2.COLOR_BGR2HSV)

    # --- 3) KNN foreground (optional) ---
    fg_mask = backSub.apply(spec_free)

    # --- 4) Buoy detection: HSV threshold + color % check ---
    lower_o, upper_o = np.array([26, 190, 0]), np.array([36, 231, 245])
    mask_buoy = cv2.inRange(hsv, lower_o, upper_o)
    contours_b, _ = cv2.findContours(
        mask_buoy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img_bgr.copy()
    buoy_ctr = None
    if contours_b:
        c = max(contours_b, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            buoy_ctr = (cx, cy)
            cv2.circle(output, buoy_ctr, 6, (0,165,255), -1)

    # --- 5) AUV detection: HSV threshold + color % + area + aspect ratio ---
    lower_y, upper_y = np.array([25, 0, 169]), np.array([46, 103, 221])
    mask_auv = cv2.inRange(hsv, lower_y, upper_y)
    contours_a, _ = cv2.findContours(
        mask_auv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    auv_ctr = None
    if contours_a:
        c = max(contours_a, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > P.MIN_AREA_SAM:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                # check color %
                y_pct = get_color_pct(
                    mask=c[:, :, None].max(axis=2) * 255,
                    hsv_image=hsv,
                    rgb_color=P.SAM_COLOR,
                    tol_hsv=(5, 25, 25)
                )
                # aspect ratio
                w, h = cv2.minAreaRect(c)[1]
                ar = min(w, h)/max(w, h) if w*h > 0 else 0
                if y_pct > P.SAM_THRESHOLD and ar < P.ASPECT_UPPER_BOUND:
                    auv_ctr = (cx, cy)
                    cv2.circle(output, auv_ctr, 6, (0,255,255), -1)

    # --- 6) Rope reconstruction (multi-frame + polynomial fit) ---
    lower_r, upper_r = np.array([6, 61, 165]), np.array([22, 120, 187])
    mask_rop = cv2.inRange(hsv, lower_r, upper_r)
    rope_color = cv2.bitwise_and(spec_free, spec_free, mask=mask_rop)

    process_image.rope_buf.append(rope_color)
    acc = np.zeros_like(rope_color)
    for im in process_image.rope_buf:
        acc = cv2.add(acc, im)
    gray_acc = cv2.cvtColor(acc, cv2.COLOR_BGR2GRAY)
    _, bin_acc = cv2.threshold(gray_acc, 1, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(bin_acc == 255)
    if len(xs) > 3:
        coeffs = np.polyfit(xs, ys, P.BEST_FIT_DEGREE)
        f = np.poly1d(coeffs)
        xs_f = np.linspace(min(xs), max(xs), 100).astype(int)
        ys_f = f(xs_f).astype(int)
        for x, y in zip(xs_f, ys_f):
            cv2.circle(output, (x, y), 1, (0,255,0), -1)

    # --- 7) Midpoint & heading arrow between buoy & AUV ---
    if auv_ctr and buoy_ctr:
        mx = (auv_ctr[0] + buoy_ctr[0]) // 2
        my = (auv_ctr[1] + buoy_ctr[1]) // 2
        cv2.circle(output, (mx, my), 5, (255,0,255), -1)
        vec = np.array(auv_ctr) - np.array(buoy_ctr)
        perp = np.array([ vec[1], -vec[0] ])
        # normalize to 0.2*pixel-length
        length = np.linalg.norm(perp)
        if length > 0:
            perp = (perp/length) * 20  # 20 px ~ 0.2m scaled
        pt2 = (int(mx + perp[0]), int(my + perp[1]))
        cv2.arrowedLine(output, (mx,my), pt2, (255,255,0), 2)

    # --- Debug plots ---
    if debug:
        titles = [
            "Orig", "SpecFree", "FG Mask", "Buoy Mask",
            "AUV Mask", "Rope Acc", "Final"
        ]
        imgs = [
            img_bgr, spec_free, fg_mask, mask_buoy,
            mask_auv, bin_acc, output
        ]
        plt.figure(figsize=(14,6))
        for i,(im,t) in enumerate(zip(imgs,titles)):
            plt.subplot(2,4,i+1)
            cmap = 'gray' if im.ndim==2 else None
            plt.imshow(im, cmap=cmap); plt.title(t); plt.axis('off')
        plt.tight_layout(); plt.show()

    return output

# persistent buffer for rope
process_image.rope_buf = deque(maxlen=5)


def process_folder(input_folder, output_folder, debug=False):
    """Loop over input_folder, process each image, save to output_folder."""
    os.makedirs(output_folder, exist_ok=True)

    backSub = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=P.KNN_LOWERBOUND,
        detectShadows=False
    )

    for img_path in sorted(glob.glob(os.path.join(input_folder, "*.*"))):
        if not img_path.lower().endswith(('.png','.jpg','jpeg','bmp')):
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Cannot read {img_path}")
            continue

        out = process_image(img, backSub, debug=debug)
        fname = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, fname), out)
        print(f"✅ Processed {fname}")


if __name__ == "__main__":
    import os
    # locate the script’s folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # default folders
    input_folder  = os.path.join(current_dir, "input_images2")
    output_folder = os.path.join(current_dir, "output")
    # run the pipeline (set debug=True if you want the plots)
    process_folder(input_folder, output_folder, debug=True)
