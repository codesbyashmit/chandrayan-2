import cv2
import numpy as np

class FeatureMatcher:

    def __init__(self, config):
        self.method = config.get('method', 'flann')
        self.ratio_threshold = config.get('ratio_threshold', 0.75)
        self.cross_check = config.get('cross_check', True)

    def match(self, src_keypoints, src_descriptors, ref_keypoints, ref_descriptors):
        if src_descriptors is None or ref_descriptors is None:
            return ([], np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
        if len(src_descriptors) < 2 or len(ref_descriptors) < 2:
            return ([], np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
        is_binary = src_descriptors.dtype == np.uint8
        if self.method == 'bf':
            matches = self._bf_match(src_descriptors, ref_descriptors, is_binary)
        elif self.method == 'flann':
            matches = self._flann_match(src_descriptors, ref_descriptors, is_binary)
        else:
            matches = self._flann_match(src_descriptors, ref_descriptors, is_binary)
        if len(matches) == 0:
            return ([], np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
        src_pts = np.float64([src_keypoints[m.queryIdx].pt if isinstance(src_keypoints[m.queryIdx], cv2.KeyPoint) else src_keypoints[m.queryIdx] for m in matches])
        ref_pts = np.float64([ref_keypoints[m.trainIdx].pt if isinstance(ref_keypoints[m.trainIdx], cv2.KeyPoint) else ref_keypoints[m.trainIdx] for m in matches])
        return (matches, src_pts, ref_pts)

    def _bf_match(self, src_desc, ref_desc, is_binary=False):
        norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type)
        raw_matches = bf.knnMatch(src_desc, ref_desc, k=2)
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        if self.cross_check and len(good_matches) > 0:
            good_matches = self._cross_check_filter(good_matches, src_desc, ref_desc, norm_type)
        return good_matches

    def _flann_match(self, src_desc, ref_desc, is_binary=False):
        if is_binary:
            index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        else:
            index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            raw_matches = flann.knnMatch(src_desc, ref_desc, k=2)
        except cv2.error:
            return self._bf_match(src_desc, ref_desc, is_binary)
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        return good_matches

    def _cross_check_filter(self, matches, src_desc, ref_desc, norm_type):
        bf_reverse = cv2.BFMatcher(norm_type)
        reverse_matches = bf_reverse.match(ref_desc, src_desc)
        reverse_map = {}
        for m in reverse_matches:
            reverse_map[m.queryIdx] = m.trainIdx
        verified = []
        for m in matches:
            if m.trainIdx in reverse_map and reverse_map[m.trainIdx] == m.queryIdx:
                verified.append(m)
        return verified

    def match_lightglue(self, source_image, reference_image, detector):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            src_kp, src_desc = detector.detect(source_image)
            ref_kp, ref_desc = detector.detect(reference_image)
            return self.match(src_kp, src_desc, ref_kp, ref_desc)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_kp, src_desc = detector.detect(source_image)
        ref_kp, ref_desc = detector.detect(reference_image)
        if len(src_kp) == 0 or len(ref_kp) == 0:
            return ([], np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
        src_coords = detector.get_keypoint_coords(src_kp)
        ref_coords = detector.get_keypoint_coords(ref_kp)
        src_desc_t = torch.from_numpy(src_desc).float().to(device)
        ref_desc_t = torch.from_numpy(ref_desc).float().to(device)
        src_desc_t = F.normalize(src_desc_t, p=2, dim=1)
        ref_desc_t = F.normalize(ref_desc_t, p=2, dim=1)
        sim_matrix = torch.mm(src_desc_t, ref_desc_t.t())
        temperature = 0.1
        prob_src = F.softmax(sim_matrix / temperature, dim=1)
        prob_ref = F.softmax(sim_matrix / temperature, dim=0)
        mutual_score = prob_src * prob_ref
        max_scores_src, max_idx_src = mutual_score.max(dim=1)
        max_scores_ref, max_idx_ref = mutual_score.max(dim=0)
        match_threshold = 0.01
        matched_pairs = []
        for i in range(len(src_coords)):
            j = max_idx_src[i].item()
            if max_idx_ref[j].item() == i and max_scores_src[i].item() > match_threshold:
                matched_pairs.append((i, j))
        if len(matched_pairs) == 0:
            return ([], np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
        src_indices, ref_indices = zip(*matched_pairs)
        src_matched_pts = src_coords[list(src_indices)]
        ref_matched_pts = ref_coords[list(ref_indices)]
        return (matched_pairs, src_matched_pts, ref_matched_pts)