import numpy as np

class UniformDistributor:

    def __init__(self, config):
        self.grid_size = config.get('grid_size', 8)
        self.min_matches_per_cell = config.get('min_matches_per_cell', 2)

    def enforce(self, src_pts, ref_pts, image_shape):
        if len(src_pts) == 0:
            return (src_pts, ref_pts, self._empty_stats())
        H, W = image_shape[:2]
        cell_h = H / self.grid_size
        cell_w = W / self.grid_size
        cell_indices = np.zeros(len(src_pts), dtype=int)
        cell_row = np.clip((src_pts[:, 1] / cell_h).astype(int), 0, self.grid_size - 1)
        cell_col = np.clip((src_pts[:, 0] / cell_w).astype(int), 0, self.grid_size - 1)
        cell_indices = cell_row * self.grid_size + cell_col
        total_cells = self.grid_size * self.grid_size
        cell_matches = {i: [] for i in range(total_cells)}
        for idx in range(len(src_pts)):
            cell_matches[cell_indices[idx]].append(idx)
        selected_indices = []
        for cell_id in range(total_cells):
            indices = cell_matches[cell_id]
            if len(indices) == 0:
                continue
            center_y = (cell_id // self.grid_size + 0.5) * cell_h
            center_x = (cell_id % self.grid_size + 0.5) * cell_w
            distances = []
            for idx in indices:
                dx = src_pts[idx][0] - center_x
                dy = src_pts[idx][1] - center_y
                distances.append(np.sqrt(dx ** 2 + dy ** 2))
            sorted_by_dist = [indices[j] for j in np.argsort(distances)]
            max_per_cell = max(self.min_matches_per_cell, len(indices) // 2)
            selected_indices.extend(sorted_by_dist[:max_per_cell])
        selected_indices = list(set(selected_indices))
        if len(selected_indices) == 0:
            return (src_pts, ref_pts, self._empty_stats())
        filtered_src = src_pts[selected_indices]
        filtered_ref = ref_pts[selected_indices]
        stats = self._compute_stats(filtered_src, image_shape)
        return (filtered_src, filtered_ref, stats)

    def _compute_stats(self, pts, image_shape):
        H, W = image_shape[:2]
        cell_h = H / self.grid_size
        cell_w = W / self.grid_size
        total_cells = self.grid_size * self.grid_size
        cell_counts = np.zeros(total_cells, dtype=int)
        for pt in pts:
            row = min(int(pt[1] / cell_h), self.grid_size - 1)
            col = min(int(pt[0] / cell_w), self.grid_size - 1)
            cell_counts[row * self.grid_size + col] += 1
        occupied_cells = np.sum(cell_counts > 0)
        coverage = occupied_cells / total_cells
        probs = cell_counts[cell_counts > 0] / cell_counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(total_cells)
        uniformity = entropy / max_entropy if max_entropy > 0 else 0
        grid_counts = cell_counts.reshape(self.grid_size, self.grid_size).tolist()
        return {'coverage': float(coverage), 'occupied_cells': int(occupied_cells), 'total_cells': int(total_cells), 'uniformity_score': float(uniformity), 'entropy': float(entropy), 'max_entropy': float(max_entropy), 'grid_counts': grid_counts, 'mean_per_cell': float(cell_counts[cell_counts > 0].mean()) if occupied_cells > 0 else 0, 'std_per_cell': float(cell_counts[cell_counts > 0].std()) if occupied_cells > 0 else 0}

    def _empty_stats(self):
        return {'coverage': 0.0, 'occupied_cells': 0, 'total_cells': self.grid_size * self.grid_size, 'uniformity_score': 0.0, 'entropy': 0.0, 'max_entropy': 0.0, 'grid_counts': [[0] * self.grid_size for _ in range(self.grid_size)], 'mean_per_cell': 0.0, 'std_per_cell': 0.0}