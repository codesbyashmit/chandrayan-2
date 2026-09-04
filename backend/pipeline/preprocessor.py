import cv2
import numpy as np

class Preprocessor:

    def __init__(self, config):
        self.method = config.get('method', 'auto')
        self.clahe_clip_limit = config.get('clahe_clip_limit', 3.0)
        self.clahe_grid_size = config.get('clahe_grid_size', 8)
        self.retinex_scales = config.get('retinex_scales', [15, 80, 250])

    def process(self, image):
        if image is None or image.size == 0:
            raise ValueError('Input image is empty or None')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.method == 'clahe':
            return self._apply_clahe(image)
        elif self.method == 'retinex':
            return self._apply_retinex(image)
        elif self.method == 'wallis':
            return self._apply_wallis(image)
        elif self.method == 'auto':
            return self._apply_auto(image)
        else:
            return image

    def _apply_clahe(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(self.clahe_grid_size, self.clahe_grid_size))
        return clahe.apply(image)

    def _apply_retinex(self, image):
        img_float = image.astype(np.float64) + 1.0
        retinex = np.zeros_like(img_float)
        for sigma in self.retinex_scales:
            blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
            blurred = np.maximum(blurred, 1.0)
            retinex += np.log10(img_float) - np.log10(blurred)
        retinex /= len(self.retinex_scales)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-08)
        retinex = (retinex * 255).astype(np.uint8)
        return retinex

    def _apply_wallis(self, image, target_mean=127, target_std=50, brightness_factor=0.8, contrast_factor=0.9, kernel_size=51):
        img_float = image.astype(np.float64)
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0) + 1e-08)
        r1 = contrast_factor * target_std / (local_std + 1e-08)
        r0 = brightness_factor * target_mean + (1 - brightness_factor) * local_mean
        gain = r1 / (1 + r1)
        offset = r0 - gain * local_mean
        result = gain * img_float + offset
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def _apply_auto(self, image):
        retinex = self._apply_retinex(image)
        result = self._apply_clahe(retinex)
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        return result