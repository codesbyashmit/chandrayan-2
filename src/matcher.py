import torch
import kornia as K
import numpy as np

class LunarFeatureMatcher:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = K.feature.LoFTR(pretrained="outdoor").to(self.device).eval()
    def match(self, img1_gray: np.ndarray, img2_gray: np.ndarray):
        t_img1 = K.image_to_tensor(img1_gray, False).float() / 255.0
        t_img2 = K.image_to_tensor(img2_gray, False).float() / 255.0
        t_img1 = t_img1.to(self.device)
        t_img2 = t_img2.to(self.device)
        input_dict = {"image0": t_img1, "image1": t_img2}
        with torch.no_grad():
            correspondences = self.matcher(input_dict)
        pts0 = correspondences["keypoints0"].cpu().numpy()
        pts1 = correspondences["keypoints1"].cpu().numpy()
        confidence = correspondences["confidence"].cpu().numpy()
        return pts0, pts1, confidence