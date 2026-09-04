import cv2
import numpy as np

class FeatureDetector:

    def __init__(self, config):
        self.method = config.get('method', 'sift')
        self.max_keypoints = config.get('max_keypoints', 5000)
        self.superpoint_threshold = config.get('superpoint_threshold', 0.005)
        self._superpoint_model = None
        self._device = None

    def detect(self, image):
        if self.method == 'sift':
            return self._detect_sift(image)
        elif self.method == 'orb':
            return self._detect_orb(image)
        elif self.method == 'akaze':
            return self._detect_akaze(image)
        elif self.method == 'superpoint':
            return self._detect_superpoint(image)
        else:
            raise ValueError(f'Unknown detection method: {self.method}')

    def _detect_sift(self, image):
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            return ([], np.array([]))
        return (keypoints, descriptors)

    def _detect_orb(self, image):
        orb = cv2.ORB_create(nfeatures=self.max_keypoints)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None:
            return ([], np.array([]))
        return (keypoints, descriptors)

    def _detect_akaze(self, image):
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(image, None)
        if descriptors is None:
            return ([], np.array([]))
        if len(keypoints) > self.max_keypoints:
            indices = np.argsort([-kp.response for kp in keypoints])[:self.max_keypoints]
            keypoints = [keypoints[i] for i in indices]
            descriptors = descriptors[indices]
        return (keypoints, descriptors)

    def _detect_superpoint(self, image):
        try:
            import torch
        except ImportError:
            print('WARNING: PyTorch not available. Falling back to SIFT.')
            return self._detect_sift(image)
        if self._superpoint_model is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._superpoint_model = self._load_superpoint()
        with torch.no_grad():
            img_tensor = self._image_to_tensor(image).to(self._device)
            output = self._superpoint_model({'image': img_tensor})
            keypoints_np = output['keypoints'][0].cpu().numpy()
            scores = output['scores'][0].cpu().numpy()
            descriptors_np = output['descriptors'][0].cpu().numpy()
            mask = scores > self.superpoint_threshold
            keypoints_np = keypoints_np[mask]
            scores = scores[mask]
            descriptors_np = descriptors_np[:, mask].T
            if len(keypoints_np) > self.max_keypoints:
                top_indices = np.argsort(-scores)[:self.max_keypoints]
                keypoints_np = keypoints_np[top_indices]
                descriptors_np = descriptors_np[top_indices]
            keypoints = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=8.0) for pt in keypoints_np]
            return (keypoints, descriptors_np.astype(np.float32))

    def _load_superpoint(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SuperPointNet(nn.Module):

            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                c1, c2, c3, c4, c5 = (64, 64, 128, 128, 256)
                self.conv1a = nn.Conv2d(1, c1, 3, padding=1)
                self.conv1b = nn.Conv2d(c1, c1, 3, padding=1)
                self.conv2a = nn.Conv2d(c1, c2, 3, padding=1)
                self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
                self.conv3a = nn.Conv2d(c2, c3, 3, padding=1)
                self.conv3b = nn.Conv2d(c3, c3, 3, padding=1)
                self.conv4a = nn.Conv2d(c3, c4, 3, padding=1)
                self.conv4b = nn.Conv2d(c4, c4, 3, padding=1)
                self.convPa = nn.Conv2d(c4, c5, 3, padding=1)
                self.convPb = nn.Conv2d(c5, 65, 1)
                self.convDa = nn.Conv2d(c4, c5, 3, padding=1)
                self.convDb = nn.Conv2d(c5, 256, 1)

            def forward(self, data):
                x = data['image']
                x = self.relu(self.conv1a(x))
                x = self.relu(self.conv1b(x))
                x = self.pool(x)
                x = self.relu(self.conv2a(x))
                x = self.relu(self.conv2b(x))
                x = self.pool(x)
                x = self.relu(self.conv3a(x))
                x = self.relu(self.conv3b(x))
                x = self.pool(x)
                x = self.relu(self.conv4a(x))
                x = self.relu(self.conv4b(x))
                cP = self.relu(self.convPa(x))
                semi = self.convPb(cP)
                cD = self.relu(self.convDa(x))
                desc = self.convDb(cD)
                dn = F.normalize(desc, p=2, dim=1)
                scores = self._decode_scores(semi)
                keypoints, kp_scores = self._extract_keypoints(scores, data['image'].shape)
                kp_desc = self._sample_descriptors(dn, keypoints, data['image'].shape)
                return {'keypoints': keypoints, 'scores': kp_scores, 'descriptors': kp_desc}

            def _decode_scores(self, semi):
                semi = semi.softmax(dim=1)
                semi = semi[:, :-1, :, :]
                b, c, h, w = semi.shape
                semi = semi.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
                semi = semi.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
                return semi

            def _extract_keypoints(self, scores, img_shape):
                b = scores.shape[0]
                H, W = (img_shape[2], img_shape[3])
                scores = scores[:, :H, :W]
                kernel = 5
                pad = kernel // 2
                max_pool = F.max_pool2d(scores.unsqueeze(1), kernel, stride=1, padding=pad).squeeze(1)
                nms_mask = scores == max_pool
                scores = scores * nms_mask.float()
                keypoints_list = []
                scores_list = []
                for i in range(b):
                    s = scores[i]
                    ys, xs = torch.where(s > 0.005)
                    sc = s[ys, xs]
                    sort_idx = torch.argsort(-sc)
                    xs = xs[sort_idx].float()
                    ys = ys[sort_idx].float()
                    sc = sc[sort_idx]
                    kps = torch.stack([xs, ys], dim=1)
                    keypoints_list.append(kps)
                    scores_list.append(sc)
                return (keypoints_list, scores_list)

            def _sample_descriptors(self, descriptors, keypoints, img_shape):
                import torch
                b, c, h, w = descriptors.shape
                H, W = (img_shape[2], img_shape[3])
                desc_list = []
                for i in range(b):
                    kps = keypoints[i]
                    if len(kps) == 0:
                        desc_list.append(torch.zeros(c, 0, device=descriptors.device))
                        continue
                    grid_x = 2.0 * kps[:, 0] / W - 1.0
                    grid_y = 2.0 * kps[:, 1] / H - 1.0
                    grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).unsqueeze(2)
                    d = F.grid_sample(descriptors[i:i + 1], grid, align_corners=True, mode='bilinear')
                    d = d.squeeze(0).squeeze(2)
                    d = F.normalize(d, p=2, dim=0)
                    desc_list.append(d)
                return desc_list
        model = SuperPointNet().to(self._device)
        model.eval()
        import os
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'superpoint_v1.pth')
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self._device)
            model.load_state_dict(state)
            print(f'[SuperPoint] Loaded weights from {weights_path}')
        else:
            print(f'[SuperPoint] No pretrained weights found at {weights_path}')
            print('[SuperPoint] Using random initialization — download weights for best results.')
            print('[SuperPoint] See: https://github.com/magicleap/SuperGluePretrainedNetwork')
        return model

    def _image_to_tensor(self, image):
        import torch
        img = image.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    def get_keypoint_coords(self, keypoints):
        if len(keypoints) == 0:
            return np.array([]).reshape(0, 2)
        if isinstance(keypoints[0], cv2.KeyPoint):
            return np.array([kp.pt for kp in keypoints], dtype=np.float64)
        return np.array(keypoints, dtype=np.float64)