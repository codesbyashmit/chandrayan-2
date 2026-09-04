import cv2
import numpy as np

class Evaluator:

    def compute_all(self, src_pts, ref_pts, inlier_mask, transform_matrix, warped_image, reference_image, final_src_pts, final_ref_pts, source_shape, subpixel_residuals, model='homography'):
        metrics = {}
        metrics['total_matches'] = int(len(src_pts))
        metrics['inlier_count'] = int(np.sum(inlier_mask))
        metrics['outlier_count'] = int(np.sum(~inlier_mask))
        metrics['inlier_ratio'] = float(np.mean(inlier_mask)) if len(inlier_mask) > 0 else 0.0
        metrics['final_match_count'] = int(len(final_src_pts))
        if len(final_src_pts) > 0 and transform_matrix is not None:
            reproj_errors = self._compute_reprojection_errors(final_src_pts, final_ref_pts, transform_matrix, model)
            metrics['rmse'] = float(np.sqrt(np.mean(reproj_errors ** 2)))
            metrics['mae'] = float(np.mean(reproj_errors))
            metrics['median_error'] = float(np.median(reproj_errors))
            metrics['max_error'] = float(np.max(reproj_errors))
            metrics['min_error'] = float(np.min(reproj_errors))
            metrics['std_error'] = float(np.std(reproj_errors))
            metrics['reprojection_errors'] = reproj_errors.tolist()
            metrics['sub_pixel_percentage'] = float(np.mean(reproj_errors < 1.0) * 100)
        else:
            metrics['rmse'] = float('inf')
            metrics['mae'] = float('inf')
            metrics['median_error'] = float('inf')
            metrics['max_error'] = float('inf')
            metrics['min_error'] = float('inf')
            metrics['std_error'] = float('inf')
            metrics['reprojection_errors'] = []
            metrics['sub_pixel_percentage'] = 0.0
        if subpixel_residuals is not None and len(subpixel_residuals) > 0:
            nonzero = subpixel_residuals[subpixel_residuals > 0]
            if len(nonzero) > 0:
                metrics['subpixel_mean_shift'] = float(np.mean(nonzero))
                metrics['subpixel_max_shift'] = float(np.max(nonzero))
            else:
                metrics['subpixel_mean_shift'] = 0.0
                metrics['subpixel_max_shift'] = 0.0
        if warped_image is not None and reference_image is not None:
            ncc, ssim = self._compute_image_similarity(warped_image, reference_image)
            metrics['ncc'] = float(ncc)
            metrics['ssim'] = float(ssim)
        else:
            metrics['ncc'] = 0.0
            metrics['ssim'] = 0.0
        return metrics

    def _compute_reprojection_errors(self, src_pts, ref_pts, H, model='homography'):
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

    def _compute_image_similarity(self, warped, reference):
        if len(warped.shape) == 3:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference
        H, W = ref_gray.shape[:2]
        if warped_gray.shape[:2] != (H, W):
            warped_gray = cv2.resize(warped_gray, (W, H))
        valid_mask = warped_gray > 0
        if np.sum(valid_mask) < 100:
            return (0.0, 0.0)
        ncc = self._ncc(warped_gray, ref_gray, valid_mask)
        ssim = self._ssim(warped_gray, ref_gray, valid_mask)
        return (ncc, ssim)

    def _ncc(self, img1, img2, mask):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        v1 = img1[mask]
        v2 = img2[mask]
        mean1 = np.mean(v1)
        mean2 = np.mean(v2)
        v1_centered = v1 - mean1
        v2_centered = v2 - mean2
        numerator = np.sum(v1_centered * v2_centered)
        denominator = np.sqrt(np.sum(v1_centered ** 2) * np.sum(v2_centered ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _ssim(self, img1, img2, mask, C1=6.5025, C2=58.5225):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel_size = 11
        sigma = 1.5
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        window = kernel @ kernel.T
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        valid_ssim = ssim_map[mask]
        if len(valid_ssim) == 0:
            return 0.0
        return float(np.mean(valid_ssim))