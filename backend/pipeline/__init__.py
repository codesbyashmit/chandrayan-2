from .preprocessor import Preprocessor
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .geometric_estimator import GeometricEstimator
from .subpixel_refiner import SubPixelRefiner
from .uniform_distribution import UniformDistributor
from .warper import ImageWarper
from .evaluator import Evaluator

def run_pipeline(source_path, reference_path, config=None):
    import cv2
    import numpy as np
    import time
    default_config = {'preprocessing': {'method': 'clahe_only', 'clahe_clip_limit': 2.0, 'clahe_grid_size': 8, 'retinex_scales': [15, 80, 250]}, 'feature_detection': {'method': 'superpoint', 'max_keypoints': 2000, 'superpoint_threshold': 0.005}, 'matching': {'method': 'flann', 'ratio_threshold': 0.75, 'cross_check': True}, 'geometric': {'model': 'homography', 'ransac_threshold': 5.0, 'max_iterations': 10000, 'confidence': 0.999}, 'subpixel': {'enabled': True, 'method': 'phase_correlation', 'window_size': 21}, 'uniform_distribution': {'enabled': True, 'grid_size': 8, 'min_matches_per_cell': 2}}
    if config:
        for key in config:
            if key in default_config:
                default_config[key].update(config[key])
    cfg = default_config
    results = {'timings': {}, 'config': cfg}
    t0 = time.time()
    source_img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    reference_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        raise ValueError(f'Could not load source image: {source_path}')
    if reference_img is None:
        raise ValueError(f'Could not load reference image: {reference_path}')
    source_color = cv2.imread(source_path)
    reference_color = cv2.imread(reference_path)
    results['source_shape'] = source_img.shape
    results['reference_shape'] = reference_img.shape
    results['timings']['load'] = time.time() - t0
    t0 = time.time()
    preprocessor = Preprocessor(cfg['preprocessing'])
    source_proc = preprocessor.process(source_img)
    reference_proc = preprocessor.process(reference_img)
    results['timings']['preprocessing'] = time.time() - t0
    t0 = time.time()
    detector = FeatureDetector(cfg['feature_detection'])
    src_kp, src_desc = detector.detect(source_proc)
    ref_kp, ref_desc = detector.detect(reference_proc)
    results['source_keypoints'] = len(src_kp)
    results['reference_keypoints'] = len(ref_kp)
    results['timings']['detection'] = time.time() - t0
    t0 = time.time()
    matcher = FeatureMatcher(cfg['matching'])
    use_lightglue = cfg['matching']['method'] == 'lightglue'
    if use_lightglue and cfg['feature_detection']['method'] == 'superpoint':
        raw_matches, src_matched_pts, ref_matched_pts = matcher.match_lightglue(source_proc, reference_proc, detector)
    else:
        raw_matches, src_matched_pts, ref_matched_pts = matcher.match(src_kp, src_desc, ref_kp, ref_desc)
    results['raw_match_count'] = len(src_matched_pts)
    results['timings']['matching'] = time.time() - t0
    if len(src_matched_pts) < 4:
        results['error'] = 'Insufficient matches found (need at least 4)'
        results['metrics'] = {}
        return results
    t0 = time.time()
    estimator = GeometricEstimator(cfg['geometric'])
    transform_matrix, inlier_mask = estimator.estimate(src_matched_pts, ref_matched_pts)
    inlier_src_pts = src_matched_pts[inlier_mask]
    inlier_ref_pts = ref_matched_pts[inlier_mask]
    results['inlier_count'] = int(np.sum(inlier_mask))
    results['transform_matrix'] = transform_matrix.tolist()
    results['timings']['geometric'] = time.time() - t0
    t0 = time.time()
    if cfg['subpixel']['enabled'] and len(inlier_src_pts) > 0:
        refiner = SubPixelRefiner(cfg['subpixel'])
        refined_src_pts, refined_ref_pts, subpixel_residuals = refiner.refine(source_proc, reference_proc, inlier_src_pts, inlier_ref_pts)
    else:
        refined_src_pts = inlier_src_pts
        refined_ref_pts = inlier_ref_pts
        subpixel_residuals = np.zeros(len(inlier_src_pts))
    results['timings']['subpixel'] = time.time() - t0
    t0 = time.time()
    if cfg['uniform_distribution']['enabled']:
        distributor = UniformDistributor(cfg['uniform_distribution'])
        dist_src_pts, dist_ref_pts, distribution_stats = distributor.enforce(refined_src_pts, refined_ref_pts, source_img.shape)
    else:
        dist_src_pts = refined_src_pts
        dist_ref_pts = refined_ref_pts
        distribution_stats = {}
    results['distribution_stats'] = distribution_stats
    results['final_match_count'] = len(dist_src_pts)
    results['timings']['distribution'] = time.time() - t0
    if len(dist_src_pts) >= 4:
        final_transform, final_mask = estimator.estimate(dist_src_pts, dist_ref_pts)
    else:
        final_transform = transform_matrix
        final_mask = np.ones(len(dist_src_pts), dtype=bool)
    t0 = time.time()
    warper = ImageWarper()
    warped_image = warper.warp(source_color if source_color is not None else source_img, final_transform, reference_img.shape, model=cfg['geometric']['model'])
    checkerboard = warper.create_checkerboard_overlay(warped_image, reference_color if reference_color is not None else reference_img)
    difference = warper.create_difference_image(warped_image, reference_color if reference_color is not None else reference_img)
    results['timings']['warping'] = time.time() - t0
    t0 = time.time()
    evaluator = Evaluator()
    metrics = evaluator.compute_all(src_matched_pts, ref_matched_pts, inlier_mask, final_transform, warped_image, reference_img, dist_src_pts, dist_ref_pts, source_img.shape, subpixel_residuals, model=cfg['geometric']['model'])
    results['metrics'] = metrics
    results['timings']['evaluation'] = time.time() - t0
    results['all_matches'] = {'source_pts': src_matched_pts.tolist(), 'reference_pts': ref_matched_pts.tolist(), 'inlier_mask': inlier_mask.tolist()}
    results['final_matches'] = {'source_pts': dist_src_pts.tolist(), 'reference_pts': dist_ref_pts.tolist()}
    results['_warped_image'] = warped_image
    results['_checkerboard'] = checkerboard
    results['_difference'] = difference
    results['_source_proc'] = source_proc
    results['_reference_proc'] = reference_proc
    total_time = sum(results['timings'].values())
    results['timings']['total'] = total_time
    return results
