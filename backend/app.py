import os
import sys
import json
import uuid
import time
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_pipeline
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'), static_url_path='')
CORS(app)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def image_to_base64(image):
    if image is None:
        return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/health', methods=['GET'])
def health():
    gpu_available = False
    gpu_name = 'N/A'
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return jsonify({'status': 'ok', 'gpu_available': gpu_available, 'gpu_name': gpu_name, 'opencv_version': cv2.__version__})

@app.route('/api/register', methods=['POST'])
def register_images():
    try:
        if 'source' not in request.files or 'reference' not in request.files:
            return (jsonify({'error': 'Both source and reference images are required'}), 400)
        source_file = request.files['source']
        reference_file = request.files['reference']
        if source_file.filename == '' or reference_file.filename == '':
            return (jsonify({'error': 'No file selected'}), 400)
        session_id = str(uuid.uuid4())[:8]
        session_dir = os.path.join(RESULTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        source_path = os.path.join(UPLOAD_DIR, f'{session_id}_source_{source_file.filename}')
        reference_path = os.path.join(UPLOAD_DIR, f'{session_id}_reference_{reference_file.filename}')
        source_file.save(source_path)
        reference_file.save(reference_path)
        config = None
        if 'config' in request.form:
            try:
                config = json.loads(request.form['config'])
            except json.JSONDecodeError:
                pass
        print(f"\n{'=' * 60}")
        print(f'[Session {session_id}] Starting registration pipeline...')
        print(f'  Source: {source_file.filename}')
        print(f'  Reference: {reference_file.filename}')
        if config:
            print(f'  Config: {json.dumps(config, indent=2)}')
        print(f"{'=' * 60}\n")
        results = run_pipeline(source_path, reference_path, config)
        if 'error' in results:
            return (jsonify({'session_id': session_id, 'error': results['error'], 'metrics': results.get('metrics', {})}), 200)
        warped_path = os.path.join(session_dir, 'warped.png')
        checker_path = os.path.join(session_dir, 'checkerboard.png')
        diff_path = os.path.join(session_dir, 'difference.png')
        cv2.imwrite(warped_path, results['_warped_image'])
        cv2.imwrite(checker_path, results['_checkerboard'])
        cv2.imwrite(diff_path, results['_difference'])
        source_img = cv2.imread(source_path)
        reference_img = cv2.imread(reference_path)
        match_viz = _draw_matches(source_img, reference_img, results)
        match_viz_path = os.path.join(session_dir, 'matches.png')
        cv2.imwrite(match_viz_path, match_viz)
        response = {'session_id': session_id, 'metrics': results['metrics'], 'timings': results['timings'], 'source_shape': list(results['source_shape']), 'reference_shape': list(results['reference_shape']), 'source_keypoints': results['source_keypoints'], 'reference_keypoints': results['reference_keypoints'], 'raw_match_count': results['raw_match_count'], 'inlier_count': results['inlier_count'], 'final_match_count': results['final_match_count'], 'transform_matrix': results['transform_matrix'], 'distribution_stats': results.get('distribution_stats', {}), 'config': results['config'], 'all_matches': results['all_matches'], 'final_matches': results['final_matches'], 'images': {'warped': image_to_base64(results['_warped_image']), 'checkerboard': image_to_base64(results['_checkerboard']), 'difference': image_to_base64(results['_difference']), 'match_visualization': image_to_base64(match_viz), 'source_preprocessed': image_to_base64(results['_source_proc']), 'reference_preprocessed': image_to_base64(results['_reference_proc'])}}
        print(f'\n[Session {session_id}] Pipeline complete!')
        print(f"  RMSE: {results['metrics'].get('rmse', 'N/A'):.4f}")
        print(f"  Inlier Ratio: {results['metrics'].get('inlier_ratio', 'N/A'):.2%}")
        print(f"  Final Matches: {results['final_match_count']}")
        print(f"  Total Time: {results['timings'].get('total', 0):.2f}s\n")
        return (jsonify(response), 200)
    except Exception as e:
        traceback.print_exc()
        return (jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500)

@app.route('/api/results/<session_id>/<filename>', methods=['GET'])
def get_result_image(session_id, filename):
    session_dir = os.path.join(RESULTS_DIR, session_id)
    if os.path.exists(os.path.join(session_dir, filename)):
        return send_file(os.path.join(session_dir, filename))
    return (jsonify({'error': 'File not found'}), 404)

def _draw_matches(source_img, reference_img, results):
    h1, w1 = source_img.shape[:2]
    h2, w2 = reference_img.shape[:2]
    max_h = max(h1, h2)
    canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = source_img
    canvas[:h2, w1:w1 + w2] = reference_img
    all_matches = results.get('all_matches', {})
    src_pts = all_matches.get('source_pts', [])
    ref_pts = all_matches.get('reference_pts', [])
    inlier_mask = all_matches.get('inlier_mask', [])
    for i in range(len(src_pts)):
        sx, sy = (int(src_pts[i][0]), int(src_pts[i][1]))
        rx, ry = (int(ref_pts[i][0]) + w1, int(ref_pts[i][1]))
        is_inlier = inlier_mask[i] if i < len(inlier_mask) else False
        if is_inlier:
            color = (0, 255, 0)
            thickness = 1
        else:
            color = (0, 0, 255)
            thickness = 1
        cv2.line(canvas, (sx, sy), (rx, ry), color, thickness)
        cv2.circle(canvas, (sx, sy), 3, color, -1)
        cv2.circle(canvas, (rx, ry), 3, color, -1)
    final_matches = results.get('final_matches', {})
    final_src = final_matches.get('source_pts', [])
    final_ref = final_matches.get('reference_pts', [])
    for i in range(len(final_src)):
        sx, sy = (int(final_src[i][0]), int(final_src[i][1]))
        rx, ry = (int(final_ref[i][0]) + w1, int(final_ref[i][1]))
        cv2.line(canvas, (sx, sy), (rx, ry), (255, 200, 0), 2)
        cv2.circle(canvas, (sx, sy), 4, (255, 200, 0), -1)
        cv2.circle(canvas, (rx, ry), 4, (255, 200, 0), -1)
    return canvas
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  LUNAR IMAGE REGISTRATION SYSTEM')
    print('  SIH 2026 — Problem Statement 26166')
    print('  ISRO — Chandrayaan-2 Image Correspondence')
    print('=' * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f'\n  GPU: {torch.cuda.get_device_name(0)}')
            print(f'  CUDA: {torch.version.cuda}')
        else:
            print('\n  GPU: Not available (CPU mode)')
    except ImportError:
        print('\n  PyTorch: Not installed (classical pipeline only)')
    print(f'\n  OpenCV: {cv2.__version__}')
    print(f'\n  Server starting on http://localhost:5000')
    print('=' * 60 + '\n')
    app.run(host='0.0.0.0', port=5000, debug=True)