"""
NeRF-UAVeL: Create rotating 3D visualization GIFs/MP4s
showing predicted bounding boxes overlaid on NeRF scene geometry.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import imageio


def get_obb_corners(box):
    """Get 8 corners of an oriented bounding box [x, y, z, w, h, d, rot]."""
    x, y, z, w, h, d, rot = box
    hw, hh, hd = w / 2, h / 2, d / 2
    corners = np.array([
        [-hw, -hh, -hd], [hw, -hh, -hd], [hw, hh, -hd], [-hw, hh, -hd],
        [-hw, -hh,  hd], [hw, -hh,  hd], [hw, hh,  hd], [-hw, hh,  hd]
    ])
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    R = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    corners = (R @ corners.T).T + np.array([x, y, z])
    return corners


def draw_obb_wireframe(ax, box, color='red', linewidth=1.5, alpha=0.9):
    """Draw wireframe OBB on matplotlib 3D axes."""
    corners = get_obb_corners(box)
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    for e in edges:
        pts = corners[e]
        ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                  color=color, linewidth=linewidth, alpha=alpha)


def draw_obb_faces(ax, box, color='red', alpha=0.06):
    """Draw semi-transparent faces of OBB."""
    corners = get_obb_corners(box)
    faces = [
        [corners[j] for j in [0,1,2,3]],
        [corners[j] for j in [4,5,6,7]],
        [corners[j] for j in [0,1,5,4]],
        [corners[j] for j in [2,3,7,6]],
        [corners[j] for j in [0,3,7,4]],
        [corners[j] for j in [1,2,6,5]],
    ]
    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.5)
    ax.add_collection3d(collection)


def load_scene(features_path, proposals_path, scene_name, top_k=10, density_threshold=1.0, subsample=3):
    """Load scene geometry and proposals."""
    feat_file = os.path.join(features_path, f'{scene_name}.npz')
    prop_file = os.path.join(proposals_path, f'{scene_name}.npz')

    if not os.path.exists(feat_file):
        print(f"Feature file not found: {feat_file}")
        return None
    if not os.path.exists(prop_file):
        print(f"Proposals file not found: {prop_file}")
        return None

    # Load features
    feat = np.load(feat_file)
    rgbsigma = feat['rgbsigma']  # H x W x D x 4
    rgb = rgbsigma[..., :3]
    density = rgbsigma[..., 3]

    # Filter occupied voxels
    occupied = density > density_threshold
    coords = np.argwhere(occupied)  # N x 3 (in voxel coordinates)
    colors = np.clip(rgb[occupied], 0, 1)  # N x 3
    densities = density[occupied]

    # Subsample for performance
    if subsample > 1 and len(coords) > 10000:
        idx = np.arange(0, len(coords), subsample)
        coords = coords[idx]
        colors = colors[idx]
        densities = densities[idx]

    # Load proposals
    prop = np.load(prop_file, allow_pickle=True)
    proposals = prop['proposals']  # N x 7
    scores = prop['scores']

    # Top-K by score
    top_idx = np.argsort(scores)[-top_k:]
    boxes = proposals[top_idx]
    box_scores = scores[top_idx]

    print(f"  Scene: {scene_name}")
    print(f"  Occupied voxels: {len(coords)} (after subsample)")
    print(f"  Top-{top_k} proposal scores: [{box_scores.min():.3f}, {box_scores.max():.3f}]")
    print(f"  Volume shape: {rgbsigma.shape[:3]}")

    return {
        'coords': coords,
        'colors': colors,
        'densities': densities,
        'boxes': boxes,
        'scores': box_scores,
        'shape': rgbsigma.shape[:3],
    }


def render_frame(scene_data, elev, azim, figsize=(12, 9), dpi=100, bg_color='white'):
    """Render a single frame of the 3D scene."""
    coords = scene_data['coords']
    colors = scene_data['colors']
    boxes = scene_data['boxes']
    scores = scene_data['scores']
    shape = scene_data['shape']

    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

    # Point size based on voxel count
    n_points = len(coords)
    point_size = max(0.8, min(6.0, 50000 / n_points))

    # Enhance colors: increase saturation and contrast
    display_colors = colors.copy()
    # Convert to HSV-like enhancement: boost saturation and brightness
    gray = display_colors.mean(axis=1, keepdims=True)
    display_colors = gray + (display_colors - gray) * 1.6  # boost saturation
    display_colors = display_colors * 1.3  # boost brightness
    display_colors = np.clip(display_colors, 0, 1)

    # Plot occupied voxels
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=display_colors, s=point_size, alpha=0.8, edgecolors='none',
        rasterized=True, depthshade=True
    )

    # Draw bounding boxes - use vivid colors
    box_colors = [
        '#E63946',  # vivid red
        '#E76F51',  # burnt orange
        '#2A9D8F',  # teal
        '#264653',  # dark blue
        '#E9C46A',  # gold
        '#F4A261',  # sandy brown
        '#6A4C93',  # purple
        '#1982C4',  # blue
        '#8AC926',  # green
        '#FF595E',  # coral
    ]
    # Sort boxes by score for consistent coloring
    sorted_idx = np.argsort(scores)
    for i, idx in enumerate(sorted_idx):
        box = boxes[idx]
        score = scores[idx]
        color = box_colors[i % len(box_colors)]
        lw = 1.5 + 1.0 * (score - scores.min()) / (scores.max() - scores.min() + 1e-8)
        draw_obb_wireframe(ax, box, color=color, linewidth=lw, alpha=0.95)
        draw_obb_faces(ax, box, color=color, alpha=0.08)

    # Set view
    ax.view_init(elev=elev, azim=azim)

    # Clean up axes
    ax.set_axis_off()

    # Equal aspect with tight bounds around scene content
    all_pts = coords.astype(float)
    for box in boxes:
        corners = get_obb_corners(box)
        all_pts = np.vstack([all_pts, corners])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.6
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
    plt.close(fig)
    return data


def create_visualization(scene_data, output_gif, output_mp4=None,
                         n_frames=90, elev=25, figsize=(12, 9), dpi=100, fps=20):
    """Create rotating visualization GIF and optional MP4."""
    frames = []
    angles = np.linspace(0, 360, n_frames, endpoint=False)

    for i, azim in enumerate(angles):
        if i % 10 == 0:
            print(f"  Rendering frame {i+1}/{n_frames}...")
        frame = render_frame(scene_data, elev=elev, azim=azim, figsize=figsize, dpi=dpi)
        frames.append(frame)

    # Save GIF
    print(f"  Saving GIF: {output_gif}")
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    imageio.mimsave(output_gif, frames, fps=fps, loop=0)
    gif_size = os.path.getsize(output_gif) / (1024 * 1024)
    print(f"  GIF size: {gif_size:.1f} MB")

    # If GIF too large, reduce quality
    if gif_size > 10:
        print("  GIF > 10MB, creating optimized version...")
        small_frames = [f[::2, ::2] for f in frames]  # downsample 2x
        imageio.mimsave(output_gif, small_frames, fps=fps, loop=0)
        gif_size = os.path.getsize(output_gif) / (1024 * 1024)
        print(f"  Optimized GIF size: {gif_size:.1f} MB")

    # Save MP4
    if output_mp4:
        print(f"  Saving MP4: {output_mp4}")
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        writer = imageio.get_writer(output_mp4, fps=fps, codec='libx264',
                                     pixelformat='yuv420p', quality=8)
        for frame in frames:
            # Ensure even dimensions for h264
            h, w = frame.shape[:2]
            h2, w2 = h - h % 2, w - w % 2
            writer.append_data(frame[:h2, :w2])
        writer.close()
        mp4_size = os.path.getsize(output_mp4) / (1024 * 1024)
        print(f"  MP4 size: {mp4_size:.1f} MB")

    return frames


def main():
    parser = argparse.ArgumentParser(description='Create NeRF-UAVeL 3D visualization GIFs/videos')
    parser.add_argument('--dataset', type=str, default='front3d', choices=['front3d', 'scannet', 'all'])
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='Specific scene names. If not given, uses defaults.')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K proposals to show')
    parser.add_argument('--density_threshold', type=float, default=1.0, help='Density threshold for voxels')
    parser.add_argument('--subsample', type=int, default=3, help='Subsample factor for voxels')
    parser.add_argument('--n_frames', type=int, default=90, help='Number of frames for rotation')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')
    parser.add_argument('--elev', type=float, default=25, help='Elevation angle')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for rendering')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Paths
    base_data = '/home/tadessewakira/Work/Research/Projects/Lubbu/Code/NeRF/data'
    base_results = '/home/tadessewakira/Work/Research/Projects/Lubbu/Code/NeRF/nerf_rpn/results'
    release_dir = '/home/tadessewakira/Work/Research/Projects/Lubbu/Code/NeRF/nerf_rpn/github_release'

    datasets_config = {
        'front3d': {
            'features': f'{base_data}/front3d_rpn_data/features',
            'proposals': f'{base_results}/selected_result/proposals',
            'default_scenes': ['3dfront_0037_00', '3dfront_0089_00', '3dfront_0091_00', '3dfront_0004_00'],
        },
        'scannet': {
            'features': f'{base_data}/scannet_rpn_data/features',
            'proposals': f'{base_results}/scannet/full_eval/proposals',
            'default_scenes': ['scene0050_00', 'scene0000_00', 'scene0040_00', 'scene0054_00'],
        },
    }

    datasets_to_run = ['front3d', 'scannet'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets_to_run:
        config = datasets_config[dataset]
        scenes = args.scenes or config['default_scenes']

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Find available scenes
        available = []
        for s in scenes:
            prop_file = os.path.join(config['proposals'], f'{s}.npz')
            feat_file = os.path.join(config['features'], f'{s}.npz')
            if os.path.exists(prop_file) and os.path.exists(feat_file):
                available.append(s)
            else:
                print(f"  Skipping {s} (missing data)")

        if not available:
            print(f"  No available scenes for {dataset}!")
            continue

        print(f"  Available scenes: {available}")

        # Render each scene
        for scene in available:
            print(f"\n--- Processing {scene} ---")
            scene_data = load_scene(
                config['features'], config['proposals'], scene,
                top_k=args.top_k,
                density_threshold=args.density_threshold,
                subsample=args.subsample
            )
            if scene_data is None:
                continue

            gif_path = os.path.join(release_dir, 'assets', f'demo_{scene}.gif')
            mp4_path = os.path.join(release_dir, 'docs', 'static', 'videos', f'demo_{scene}.mp4')

            create_visualization(
                scene_data,
                output_gif=gif_path,
                output_mp4=mp4_path,
                n_frames=args.n_frames,
                elev=args.elev,
                figsize=(12, 9),
                dpi=args.dpi,
                fps=args.fps,
            )

        # Create a "best" symlink or copy for the dataset-level demo
        if available:
            best_scene = available[0]
            import shutil
            # Dataset-level GIF
            src_gif = os.path.join(release_dir, 'assets', f'demo_{best_scene}.gif')
            dst_gif = os.path.join(release_dir, 'assets', f'demo_{dataset}.gif')
            if os.path.exists(src_gif) and src_gif != dst_gif:
                shutil.copy2(src_gif, dst_gif)
                print(f"\n  Dataset demo: {dst_gif}")

            src_mp4 = os.path.join(release_dir, 'docs', 'static', 'videos', f'demo_{best_scene}.mp4')
            dst_mp4 = os.path.join(release_dir, 'docs', 'static', 'videos', f'demo_{dataset}.mp4')
            if os.path.exists(src_mp4) and src_mp4 != dst_mp4:
                shutil.copy2(src_mp4, dst_mp4)

    print("\n" + "="*60)
    print("Done! Outputs saved to:")
    print(f"  GIFs: {release_dir}/assets/")
    print(f"  MP4s: {release_dir}/docs/static/videos/")
    print("="*60)


if __name__ == '__main__':
    main()
