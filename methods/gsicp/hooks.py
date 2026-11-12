"""Subclasses of GS-ICP SLAM's Tracker and Mapper that the adapter swaps in.

GS-ICP runs tracking and mapping as two processes started with the 'spawn'
method, so a patch applied in the parent process would not reach them: the
children unpickle these classes by reference and import this module (the
adapter puts its folder and the GS-ICP source on sys.path, which spawn hands
down). Nothing of the SLAM itself changes:

- ExportTracker: times the run (image preloading and the tracking loop
  separately) and skips GS-ICP's own ATE against the identity stand-in poses
  the adapter stages (traj.txt is never the reference trajectory);
- ExportMapper: replaces GS-ICP's final evaluation over every frame
  (calc_2d_metric, its PSNR/SSIM/LPIPS) with the export the protocol needs,
  done the same way calc_2d_metric renders: the final map at the final
  poses, for the frames of protocol.required_renders, plus the keyframe list.
"""
import json
import os
import time

import numpy as np
import torch

import protocol
from gaussian_renderer import render
from mp_Mapper import Mapper
from mp_Tracker import Tracker

HOOK_DIR = "hook"  # under GS-ICP's output_path


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


class ExportTracker(Tracker):
    def evaluate_ate(self, gt_traj, est_traj):
        # GS-ICP prints its own ATE at the end; the staged traj.txt holds
        # identity matrices, so there is nothing to compare with. Scoring is
        # score.py's.
        return float("nan")

    def run(self):
        t0 = time.time()
        super().run()  # get_images (preload of every frame), then the tracking loop
        t_end = time.time()
        out = os.path.join(self.output_path, HOOK_DIR)
        os.makedirs(out, exist_ok=True)
        _write_json(os.path.join(out, "tracker_timing.json"), {
            "frames": int(self.num_images),
            "preload_s": self.total_start_time - t0,
            "loop_s": t_end - self.total_start_time,
            "gpu": torch.cuda.get_device_name(0),
        })
        print(f"hook: tracking loop {t_end - self.total_start_time:.1f} s for {self.num_images} frames", flush=True)


class ExportMapper(Mapper):
    def calc_2d_metric(self):
        out = os.path.join(self.output_path, HOOK_DIR)
        renders_dir = os.path.join(out, "renders")
        os.makedirs(renders_dir, exist_ok=True)
        poses = self.final_pose.numpy().astype(np.float64)  # camera-to-world, set by the tracker
        np.save(os.path.join(out, "poses_c2w.npy"), poses)
        keyframes = sorted({int(i) for i in self.keyframe_idxs})
        frames = protocol.required_renders(len(poses), keyframes)
        print(f"hook: rendering {len(frames)} frames from the final map "
              f"({self.gaussians.get_xyz.shape[0]} Gaussians, {self.train_iter} mapping iterations)", flush=True)
        cam = self.mapping_cams[0]
        with torch.no_grad():
            for j, i in enumerate(frames):
                # as calc_2d_metric: camera from the final pose, full resolution
                w2c = np.linalg.inv(poses[i])
                cam.R = torch.tensor(w2c[:3, :3].transpose())
                cam.t = torch.tensor(w2c[:3, 3])
                cam.update_matrix()
                image = render(cam, self.gaussians, self.pipe, self.background)["render"]
                np.save(os.path.join(renders_dir, f"{i:06d}.npy"),
                        image.permute(1, 2, 0).cpu().numpy().astype(np.float16))
                if (j + 1) % 50 == 0 or j + 1 == len(frames):
                    print(f"hook: rendered {j + 1}/{len(frames)}", flush=True)
        _write_json(os.path.join(out, "mapper_state.json"), {
            "keyframes": keyframes,
            "gaussians": int(self.gaussians.get_xyz.shape[0]),
            "mapping_iterations": int(self.train_iter),
            "image_size": [int(cam.image_width), int(cam.image_height)],
        })
