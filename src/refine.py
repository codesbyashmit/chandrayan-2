import cv2
import numpy as np

def filter_uniform_distribution(pts0, pts1, conf, img_shape, grid_size=(4, 4), max_per_cell=50):
    h, w = img_shape
    gh, gw = grid_size
    cell_h, cell_w = h / gh, w / gw
    selected_idx = []
    grid_buckets = {}
    sorted_indices = np.argsort(-conf)

    for idx in sorted_indices:
        x, y = pts0[idx]
        cell_x = min(int(x // cell_w), gw - 1)
        cell_y = min(int(y // cell_h), gh - 1)
        bucket = (cell_y, cell_x)
        if bucket not in grid_buckets:
            grid_buckets[bucket] = 0
        if grid_buckets[bucket] < max_per_cell:
            selected_idx.append(idx)
            grid_buckets[bucket] += 1
    selected_idx = np.array(selected_idx)
    return pts0[selected_idx], pts1[selected_idx]

def refine_and_estimate(pts0, pts1, img0_gray, reproj_threshold=3.0):
    if len(pts0) < 4:
        return None, None, None, 0.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    pts0_refined = pts0.copy().astype(np.float32)
    pts0_refined = cv2.cornerSubPix(
        img0_gray, pts0_refined, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
    )
    H, inlier_mask = cv2.findHomography(
        pts0_refined, pts1, method=cv2.USAC_MAGSAC, ransacReprojThreshold=reproj_threshold
    )
    inlier_mask = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.zeros(len(pts0), dtype=bool)
    inlier_ratio = float(np.sum(inlier_mask)) / float(len(pts0)) if len(pts0) > 0 else 0.0
    return H, pts0_refined[inlier_mask], pts1[inlier_mask], inlier_ratio