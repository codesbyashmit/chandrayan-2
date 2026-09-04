import cv2
import numpy as np
from scipy.ndimage import shift as ndi_shift

class SubPixelRefiner:

    def __init__(self, config):
        self.method = config.get('method', 'phase_correlation')
        self.window_size = config.get('window_size', 21)

    def refine(self, source_image, reference_image, src_pts, ref_pts):
        if self.method == 'phase_correlation':
            return self._phase_correlation_refine(source_image, reference_image, src_pts, ref_pts)
        elif self.method == 'lsm':
            return self._lsm_refine(source_image, reference_image, src_pts, ref_pts)
        else:
            return (src_pts, ref_pts, np.zeros(len(src_pts)))

    def _phase_correlation_refine(self, source_image, reference_image, src_pts, ref_pts):
        half_w = self.window_size // 2
        src_float = source_image.astype(np.float32)
        ref_float = reference_image.astype(np.float32)
        refined_src = []
        refined_ref = []
        residuals = []
        for i in range(len(src_pts)):
            sx, sy = (int(round(src_pts[i][0])), int(round(src_pts[i][1])))
            rx, ry = (int(round(ref_pts[i][0])), int(round(ref_pts[i][1])))
            if sy - half_w < 0 or sy + half_w >= source_image.shape[0] or sx - half_w < 0 or (sx + half_w >= source_image.shape[1]) or (ry - half_w < 0) or (ry + half_w >= reference_image.shape[0]) or (rx - half_w < 0) or (rx + half_w >= reference_image.shape[1]):
                refined_src.append(src_pts[i])
                refined_ref.append(ref_pts[i])
                residuals.append(0.0)
                continue
            src_patch = src_float[sy - half_w:sy + half_w + 1, sx - half_w:sx + half_w + 1]
            ref_patch = ref_float[ry - half_w:ry + half_w + 1, rx - half_w:rx + half_w + 1]
            hann = cv2.createHanningWindow((src_patch.shape[1], src_patch.shape[0]), cv2.CV_32F)
            src_patch = src_patch * hann
            ref_patch = ref_patch * hann
            try:
                shift, response = cv2.phaseCorrelate(src_patch, ref_patch, hann)
                dx, dy = shift
                if abs(dx) < 2.0 and abs(dy) < 2.0:
                    refined_src.append([src_pts[i][0] + dx / 2, src_pts[i][1] + dy / 2])
                    refined_ref.append([ref_pts[i][0] - dx / 2, ref_pts[i][1] - dy / 2])
                    residuals.append(np.sqrt(dx ** 2 + dy ** 2))
                else:
                    refined_src.append(src_pts[i])
                    refined_ref.append(ref_pts[i])
                    residuals.append(0.0)
            except cv2.error:
                refined_src.append(src_pts[i])
                refined_ref.append(ref_pts[i])
                residuals.append(0.0)
        return (np.array(refined_src), np.array(refined_ref), np.array(residuals))

    def _lsm_refine(self, source_image, reference_image, src_pts, ref_pts, max_iterations=20, convergence_threshold=0.001):
        half_w = self.window_size // 2
        src_float = source_image.astype(np.float64)
        ref_float = reference_image.astype(np.float64)
        refined_src = []
        refined_ref = []
        residuals = []
        for i in range(len(src_pts)):
            sx, sy = (src_pts[i][0], src_pts[i][1])
            rx, ry = (ref_pts[i][0], ref_pts[i][1])
            isx, isy = (int(round(sx)), int(round(sy)))
            irx, iry = (int(round(rx)), int(round(ry)))
            if isy - half_w < 0 or isy + half_w >= source_image.shape[0] or isx - half_w < 0 or (isx + half_w >= source_image.shape[1]) or (iry - half_w < 0) or (iry + half_w >= reference_image.shape[0]) or (irx - half_w < 0) or (irx + half_w >= reference_image.shape[1]):
                refined_src.append(src_pts[i])
                refined_ref.append(ref_pts[i])
                residuals.append(0.0)
                continue
            template = ref_float[iry - half_w:iry + half_w + 1, irx - half_w:irx + half_w + 1]
            params = np.array([sx - isx, 1.0, 0.0, sy - isy, 0.0, 1.0, 0.0, 1.0])
            converged = False
            for iteration in range(max_iterations):
                A = []
                l = []
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        x_prime = params[0] + params[1] * dx + params[2] * dy + isx
                        y_prime = params[3] + params[4] * dx + params[5] * dy + isy
                        ix = int(np.floor(x_prime))
                        iy = int(np.floor(y_prime))
                        fx = x_prime - ix
                        fy = y_prime - iy
                        if iy < 0 or iy + 1 >= source_image.shape[0] or ix < 0 or (ix + 1 >= source_image.shape[1]):
                            continue
                        val = (1 - fx) * (1 - fy) * src_float[iy, ix] + fx * (1 - fy) * src_float[iy, ix + 1] + (1 - fx) * fy * src_float[iy + 1, ix] + fx * fy * src_float[iy + 1, ix + 1]
                        if iy < 1 or iy + 1 >= source_image.shape[0] - 1 or ix < 1 or (ix + 1 >= source_image.shape[1] - 1):
                            continue
                        gx = (src_float[iy, ix + 1] - src_float[iy, ix - 1]) / 2.0
                        gy = (src_float[iy + 1, ix] - src_float[iy - 1, ix]) / 2.0
                        ref_val = template[dy + half_w, dx + half_w]
                        predicted = params[6] + params[7] * val
                        row = np.array([params[7] * gx, params[7] * gx * dx, params[7] * gx * dy, params[7] * gy, params[7] * gy * dx, params[7] * gy * dy, 1.0, val])
                        A.append(row)
                        l.append(ref_val - predicted)
                if len(A) < 8:
                    break
                A = np.array(A)
                l = np.array(l)
                try:
                    delta = np.linalg.lstsq(A, l, rcond=None)[0]
                except np.linalg.LinAlgError:
                    break
                params += delta
                if np.max(np.abs(delta[:6])) < convergence_threshold:
                    converged = True
                    break
            refined_sx = isx + params[0]
            refined_sy = isy + params[3]
            residual = np.sqrt((refined_sx - sx) ** 2 + (refined_sy - sy) ** 2)
            if residual < 3.0:
                refined_src.append([refined_sx, refined_sy])
                refined_ref.append(ref_pts[i])
                residuals.append(residual)
            else:
                refined_src.append(src_pts[i])
                refined_ref.append(ref_pts[i])
                residuals.append(0.0)
        return (np.array(refined_src), np.array(refined_ref), np.array(residuals))