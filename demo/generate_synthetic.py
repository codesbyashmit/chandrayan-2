import os
import sys
import cv2
import numpy as np
from pathlib import Path


def generate_lunar_surface(width=1024, height=1024, num_craters=80, seed=42):
    rng = np.random.RandomState(seed)
    surface = np.zeros((height, width), dtype=np.float64)
    for octave in range(5):
        scale = 2 ** octave
        freq = max(4, width // (scale * 4))
        noise = rng.randn(freq, freq)
        noise_resized = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        surface += noise_resized * 0.5 ** octave
    surface = (surface - surface.min()) / (surface.max() - surface.min() + 1e-08)
    for _ in range(num_craters):
        cx = rng.randint(50, width - 50)
        cy = rng.randint(50, height - 50)
        radius = rng.randint(8, min(width, height) // 8)
        depth = rng.uniform(0.1, 0.4)
        y, x = np.ogrid[-cy:height - cy, -cx:width - cx]
        r = np.sqrt(x * x + y * y)
        crater_mask = r < radius
        inner_profile = 1.0 - (r[crater_mask] / radius) ** 2
        surface[crater_mask] -= depth * inner_profile
        rim_mask = (r >= radius * 0.85) & (r <= radius * 1.2)
        rim_profile = np.exp(-((r[rim_mask] - radius) / (radius * 0.15)) ** 2)
        surface[rim_mask] += depth * 0.3 * rim_profile
        shadow_mask = (r < radius * 1.1) & (x < 0)
        shadow_region = surface[shadow_mask]
        surface[shadow_mask] = shadow_region * 0.7
    for _ in range(num_craters * 3):
        bx = rng.randint(5, width - 5)
        by = rng.randint(5, height - 5)
        br = rng.randint(2, 6)
        brightness = rng.uniform(-0.15, 0.15)
        cv2.circle(surface, (bx, by), br, float(surface[by, bx] + brightness), -1)
    surface = np.clip(surface, 0, 1)
    surface = (surface * 255).astype(np.uint8)
    surface = cv2.GaussianBlur(surface, (3, 3), 0.8)
    return surface

def apply_illumination_change(image, gamma=1.5, direction_angle=45, seed=None):
    rng = np.random.RandomState(seed)
    img_float = image.astype(np.float64) / 255.0
    img_float = np.power(img_float, gamma)
    h, w = image.shape[:2]
    angle_rad = np.radians(direction_angle)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    gradient = 0.5 + 0.3 * (X * np.cos(angle_rad) + Y * np.sin(angle_rad))
    img_float = img_float * gradient
    noise = rng.randn(h, w) * 0.02
    img_float += noise
    img_float = np.clip(img_float, 0, 1)
    return (img_float * 255).astype(np.uint8)

def apply_geometric_transform(image, rotation_deg=15, scale=0.9, tx=30, ty=20, perspective_strength=0.0005, seed=None):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M_rot = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M_rot
    if perspective_strength > 0:
        rng = np.random.RandomState(seed)
        H[2, 0] = rng.uniform(-perspective_strength, perspective_strength)
        H[2, 1] = rng.uniform(-perspective_strength, perspective_strength)
    transformed = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return (transformed, H)

def generate_ground_truth_matches(H, image_shape, num_points=200, seed=42):
    rng = np.random.RandomState(seed)
    h, w = image_shape[:2]
    grid_size = int(np.sqrt(num_points))
    xs = np.linspace(w * 0.1, w * 0.9, grid_size)
    ys = np.linspace(h * 0.1, h * 0.9, grid_size)
    src_points = []
    ref_points = []
    for x in xs:
        for y in ys:
            jx = x + rng.uniform(-10, 10)
            jy = y + rng.uniform(-10, 10)
            pt = np.array([jx, jy, 1.0])
            pt_transformed = H @ pt
            pt_transformed = pt_transformed[:2] / pt_transformed[2]
            tx, ty_val = pt_transformed
            if 10 < tx < w - 10 and 10 < ty_val < h - 10:
                src_points.append([jx, jy])
                ref_points.append([tx, ty_val])
    return (np.array(src_points), np.array(ref_points))

def generate_test_pairs(output_dir, num_pairs=3):
    import json
    os.makedirs(output_dir, exist_ok=True)
    print('Generating synthetic lunar surface...')
    reference = generate_lunar_surface(1024, 1024, num_craters=100, seed=42)
    ref_path = os.path.join(output_dir, 'reference.png')
    cv2.imwrite(ref_path, reference)
    print(f'  Saved reference: {ref_path}')
    test_configs = [{'name': 'mild_rotation_scale', 'rotation': 5, 'scale': 0.95, 'tx': 15, 'ty': 10, 'perspective': 0.0001, 'gamma': 1.2, 'sun_angle': 30, 'description': 'Mild rotation (5°) + slight scale change + minor illumination shift'}, {'name': 'moderate_transform', 'rotation': 15, 'scale': 0.85, 'tx': 40, 'ty': 30, 'perspective': 0.0003, 'gamma': 1.8, 'sun_angle': 120, 'description': 'Moderate rotation (15°) + scale (0.85) + significant illumination change'}, {'name': 'challenging_transform', 'rotation': 30, 'scale': 0.7, 'tx': 60, 'ty': 50, 'perspective': 0.0005, 'gamma': 2.2, 'sun_angle': 210, 'description': 'Challenging: large rotation (30°) + scale (0.7) + extreme illumination'}]
    for i, cfg in enumerate(test_configs[:num_pairs]):
        print(f"\nGenerating test pair {i + 1}: {cfg['name']}")
        illuminated = apply_illumination_change(reference, gamma=cfg['gamma'], direction_angle=cfg['sun_angle'], seed=i + 100)
        source, H = apply_geometric_transform(illuminated, rotation_deg=cfg['rotation'], scale=cfg['scale'], tx=cfg['tx'], ty=cfg['ty'], perspective_strength=cfg['perspective'], seed=i + 200)
        gt_src_pts, gt_ref_pts = generate_ground_truth_matches(H, reference.shape, num_points=400, seed=i + 300)
        src_path = os.path.join(output_dir, f"source_{i + 1}_{cfg['name']}.png")
        cv2.imwrite(src_path, source)
        print(f'  Saved source: {src_path}')
        gt_data = {'description': cfg['description'], 'transform_matrix': H.tolist(), 'parameters': {'rotation_deg': cfg['rotation'], 'scale': cfg['scale'], 'translation_x': cfg['tx'], 'translation_y': cfg['ty'], 'perspective': cfg['perspective'], 'gamma': cfg['gamma'], 'sun_angle': cfg['sun_angle']}, 'ground_truth_matches': {'source_pts': gt_src_pts.tolist(), 'reference_pts': gt_ref_pts.tolist(), 'count': len(gt_src_pts)}}
        gt_path = os.path.join(output_dir, f"ground_truth_{i + 1}_{cfg['name']}.json")
        with open(gt_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        print(f'  Saved ground truth: {gt_path}')
        print(f"  Transform: rot={cfg['rotation']}° scale={cfg['scale']} gamma={cfg['gamma']} sun={cfg['sun_angle']}°")
        print(f'  Ground truth matches: {len(gt_src_pts)}')
    print(f'\n[OK] Generated {min(num_pairs, len(test_configs))} test pairs in {output_dir}')
    return output_dir
if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'sample_images')
    generate_test_pairs(output_dir, num_pairs=3)