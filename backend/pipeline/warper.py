import cv2
import numpy as np

class ImageWarper:

    def warp(self, source_image, transform_matrix, target_shape, model='homography'):
        H, W = target_shape[:2]
        if model == 'affine':
            A = transform_matrix[:2, :]
            warped = cv2.warpAffine(source_image, A, (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            warped = cv2.warpPerspective(source_image, transform_matrix, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped

    def create_checkerboard_overlay(self, warped_image, reference_image, block_size=64):
        warped = self._ensure_3channel(warped_image)
        ref = self._ensure_3channel(reference_image)
        H, W = ref.shape[:2]
        warped_resized = cv2.resize(warped, (W, H)) if warped.shape[:2] != (H, W) else warped
        mask = np.zeros((H, W), dtype=np.uint8)
        for y in range(0, H, block_size):
            for x in range(0, W, block_size):
                ry = y // block_size % 2
                rx = x // block_size % 2
                if ry == rx:
                    mask[y:y + block_size, x:x + block_size] = 255
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch > 0, warped_resized, ref)
        return result

    def create_difference_image(self, warped_image, reference_image):
        if len(warped_image.shape) == 3:
            warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped_image
        if len(reference_image.shape) == 3:
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_image
        H, W = ref_gray.shape[:2]
        warped_resized = cv2.resize(warped_gray, (W, H)) if warped_gray.shape[:2] != (H, W) else warped_gray
        diff = cv2.absdiff(warped_resized, ref_gray)
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        return diff_colored

    def create_blend_overlay(self, warped_image, reference_image, alpha=0.5):
        warped = self._ensure_3channel(warped_image)
        ref = self._ensure_3channel(reference_image)
        H, W = ref.shape[:2]
        warped_resized = cv2.resize(warped, (W, H)) if warped.shape[:2] != (H, W) else warped
        blend = cv2.addWeighted(warped_resized, alpha, ref, 1.0 - alpha, 0)
        return blend

    def _ensure_3channel(self, image):
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image