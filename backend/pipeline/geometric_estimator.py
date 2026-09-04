import cv2
import numpy as np

class GeometricEstimator:

    def __init__(self, config):
        self.model = config.get('model', 'homography')
        self.ransac_threshold = config.get('ransac_threshold', 5.0)
        self.max_iterations = config.get('max_iterations', 10000)
        self.confidence = config.get('confidence', 0.999)

    def estimate(self, src_pts, ref_pts):
        if len(src_pts) < 4:
            raise ValueError(f'Need at least 4 point pairs, got {len(src_pts)}')
        src_pts = np.float64(src_pts).reshape(-1, 1, 2)
        ref_pts = np.float64(ref_pts).reshape(-1, 1, 2)
        if self.model == 'homography':
            return self._estimate_homography(src_pts, ref_pts)
        elif self.model == 'affine':
            return self._estimate_affine(src_pts, ref_pts)
        else:
            return self._estimate_homography(src_pts, ref_pts)

    def _estimate_homography(self, src_pts, ref_pts):
        try:
            H, mask = cv2.findHomography(src_pts, ref_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=self.max_iterations, confidence=self.confidence)
        except (cv2.error, AttributeError):
            H, mask = cv2.findHomography(src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=self.max_iterations, confidence=self.confidence)
        if H is None:
            H = np.eye(3)
            mask = np.zeros(len(src_pts), dtype=np.uint8)
        inlier_mask = mask.ravel().astype(bool)
        return (H, inlier_mask)

    def _estimate_affine(self, src_pts, ref_pts):
        if len(src_pts) < 3:
            raise ValueError(f'Need at least 3 point pairs for affine, got {len(src_pts)}')
        A, mask = cv2.estimateAffine2D(src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=self.max_iterations, confidence=self.confidence)
        if A is None:
            A = np.eye(2, 3, dtype=np.float64)
            mask = np.zeros(len(src_pts), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = A
        inlier_mask = mask.ravel().astype(bool)
        return (H, inlier_mask)

    def compute_reprojection_errors(self, src_pts, ref_pts, H, model='homography'):
        src_pts = np.float64(src_pts).reshape(-1, 2)
        ref_pts = np.float64(ref_pts).reshape(-1, 2)
        if model == 'affine':
            ones = np.ones((len(src_pts), 1))
            src_h = np.hstack([src_pts, ones])
            A = H[:2, :]
            ref_pred = (A @ src_h.T).T
        else:
            ones = np.ones((len(src_pts), 1))
            src_h = np.hstack([src_pts, ones])
            ref_pred_h = (H @ src_h.T).T
            w = ref_pred_h[:, 2:3]
            w[w == 0] = 1e-10
            ref_pred = ref_pred_h[:, :2] / w
        errors = np.sqrt(np.sum((ref_pts - ref_pred) ** 2, axis=1))
        return errors