"""COLMAP-format package for DarkGS from a MonoGS Replica-format dataset at the
reference poses (PROTOCOL.md, Nhóm C). Ported from the data repo's
scripts/darkgs/build_dataset.py (the packages the DarkGS fork's README
describes), with the camera and depth scale from protocol.calibration()
instead of the dataset's config, and the training frames given by the caller.

    images/frameNNNNNN.jpg      the training frames (links to results/)
    sparse/0/cameras.txt        one PINHOLE camera (COLMAP pixel centres: cx + 0.5)
    sparse/0/images.txt         world-to-camera (qvec, tvec) from traj.txt (camera-to-world, OpenCV axes)
    sparse/0/points3D.ply       initial point cloud: the training frames' sensor depth back-projected
                                at the reference poses, merged on a voxel grid, with colours and
                                NON-ZERO normals pointing AWAY from the camera (DarkGS computes
                                cos = relu(n . (point - light)): zero normals give zero colour and no gradient)

No COLMAP: poses are the reference poses (metric, so --scaling_factor 1.0),
points come from depth instead of SfM (the RGB-D setting of this project).
"""
import os

import numpy as np
from PIL import Image

# Point cloud of the package, as the data repo's build_dataset.py defaults
VOXEL_M = 0.04          # merge cell
PIXEL_STRIDE = 4        # back-project every 4th pixel in each direction (grid shifted per frame)
MAX_DEPTH_M = 8.0       # drop points farther than this
# Depth-continuity tests of the normal estimate (relative to the centre depth)
SMOOTH_FAR, SMOOTH_NEAR = 0.06, 0.03


def rotmat2qvec(R):
    """Rotation matrix -> quaternion (w, x, y, z), COLMAP's read_write_model.py algorithm, w >= 0."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                  [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, v = np.linalg.eigh(K)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def write_cameras(path, cal):
    # COLMAP puts pixel centres at +0.5: the OpenCV principal point (299.5, 169.5) is (300, 170) here.
    # DarkGS reads only fx, fy (field of view, principal point at the image centre).
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                "# Number of cameras: 1\n")
        f.write("1 PINHOLE %d %d %.10g %.10g %.10g %.10g\n" % (cal["width"], cal["height"], cal["fx"], cal["fy"],
                                                            cal["cx"] + 0.5, cal["cy"] + 0.5))


def write_images(path, idxs, c2w):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n# Number of images: %d\n" % len(idxs))
        for k, i in enumerate(idxs):
            R_w2c = c2w[i][:3, :3].T
            q, t = rotmat2qvec(R_w2c), -R_w2c @ c2w[i][:3, 3]
            f.write("%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g 1 frame%06d.jpg\n\n" % ((k + 1,) + tuple(q) + tuple(t) + (i,)))


def normals_from_depth(P):
    """Central-difference normals of a camera point map (H, W, 3), facing the camera; returns
    (n (H-2, W-2, 3), smooth (H-2, W-2)): smooth = depth continuous around the pixel."""
    dpu = P[1:-1, 2:] - P[1:-1, :-2]
    dpv = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dpu, dpv)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    n[(n * P[1:-1, 1:-1]).sum(-1) > 0] *= -1
    z = P[..., 2]
    zc = z[1:-1, 1:-1]
    smooth = ((np.abs(z[1:-1, 2:] - z[1:-1, :-2]) < SMOOTH_FAR * zc) & (np.abs(z[2:, 1:-1] - z[:-2, 1:-1]) < SMOOTH_FAR * zc)
              & (zc > 0))
    for nb in (z[1:-1, 2:], z[2:, 1:-1], z[1:-1, :-2], z[:-2, 1:-1]):
        smooth &= np.abs(nb - zc) < SMOOTH_NEAR * zc
    return n, smooth


def make_points(idxs, colors, depths, c2w, cal):
    """Back-project the frames' depth at the reference poses, with colours and normals (away from the
    camera); merge on a voxel grid (mean normal of the views; the viewing ray where none is valid)."""
    H, W = cal["height"], cal["width"]
    vv, uu = np.mgrid[0:H, 0:W]
    P, C, NG, NR = [], [], [], []
    for k, i in enumerate(idxs):
        d = np.array(Image.open(depths[i])).astype(np.float32) / cal["depth_scale"]
        rgb = np.array(Image.open(colors[i]).convert("RGB"))
        if d.shape != (H, W) or rgb.shape[:2] != (H, W):
            raise ValueError(f"frame {i}: depth {d.shape}, colour {rgb.shape}, camera {W}x{H}")
        pmap = np.stack([(uu - cal["cx"]) / cal["fx"] * d, (vv - cal["cy"]) / cal["fy"] * d, d], -1).astype(np.float64)
        n_cam, smooth = normals_from_depth(pmap)
        v, u = np.mgrid[k % PIXEL_STRIDE:H:PIXEL_STRIDE, (3 * k) % PIXEL_STRIDE:W:PIXEL_STRIDE]
        z = d[v, u]
        ok = (z > 0) & (z <= MAX_DEPTH_M)
        v, u = v[ok], u[ok]
        pc = pmap[v, u]
        R = c2w[i][:3, :3]
        inner = (v >= 1) & (v < H - 1) & (u >= 1) & (u < W - 1)
        n_c = np.zeros_like(pc)
        good = np.zeros(len(v), bool)
        n_c[inner] = n_cam[v[inner] - 1, u[inner] - 1]
        good[inner] = smooth[v[inner] - 1, u[inner] - 1]
        NG.append(-(n_c @ R.T) * good[:, None])
        NR.append((pc / np.linalg.norm(pc, axis=1, keepdims=True)) @ R.T)
        P.append((pc @ R.T + c2w[i][:3, 3]).astype(np.float32))
        C.append(rgb[v, u])
    P, C, NG, NR = np.concatenate(P), np.concatenate(C).astype(np.float64), np.concatenate(NG), np.concatenate(NR)
    n_raw = len(P)
    q = np.floor((P - P.min(0)) / VOXEL_M).astype(np.int64)
    m = q.max(0) + 1
    key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    xyz = np.stack([np.bincount(inv, P[:, a].astype(np.float64)) for a in range(3)], -1) / cnt[:, None]
    rgb = np.stack([np.bincount(inv, C[:, a]) for a in range(3)], -1) / cnt[:, None]
    ng = np.stack([np.bincount(inv, NG[:, a]) for a in range(3)], -1)
    nr = np.stack([np.bincount(inv, NR[:, a]) for a in range(3)], -1)
    has = np.linalg.norm(ng, axis=1) > 1e-6
    nrm = np.where(has[:, None], ng, nr)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    return (xyz.astype(np.float32), np.rint(rgb).clip(0, 255).astype(np.uint8), nrm.astype(np.float32),
            {"raw_points": n_raw, "points": int(len(xyz)), "voxel_m": VOXEL_M, "pixel_stride": PIXEL_STRIDE,
             "max_depth_m": MAX_DEPTH_M, "geometric_normals": float(has.mean())})


def write_ply(path, xyz, rgb, nrm):
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    el = np.zeros(len(xyz), dt)
    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["nx"], el["ny"], el["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    el["red"], el["green"], el["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\n"
           "property uchar blue\nend_header\n" % len(xyz))
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(el.tobytes())


def write_binary(sparse, idxs, c2w, cal, xyz, rgb):
    """COLMAP's binary model (cameras.bin, images.bin, points3D.bin; no 2D points, no tracks), the
    same content as the text files and the .ply, for readers that need it: gsplat's pycolmap fork
    reads .bin first and its text reader is broken under Python 3 (np.array(map(...)))."""
    import struct

    with open(os.path.join(sparse, "cameras.bin"), "wb") as f:  # model 1 = PINHOLE: fx, fy, cx, cy
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<IiQQ", 1, 1, cal["width"], cal["height"]))
        f.write(struct.pack("<4d", cal["fx"], cal["fy"], cal["cx"] + 0.5, cal["cy"] + 0.5))
    with open(os.path.join(sparse, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(idxs)))
        for k, i in enumerate(idxs):
            R_w2c = c2w[i][:3, :3].T
            q, t = rotmat2qvec(R_w2c), -R_w2c @ c2w[i][:3, 3]
            f.write(struct.pack("<I4d3dI", k + 1, *q, *t, 1))
            f.write(("frame%06d.jpg" % i).encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(os.path.join(sparse, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for j, (p, c) in enumerate(zip(xyz.astype(np.float64), rgb)):
            f.write(struct.pack("<Q3d3BdQ", j + 1, *p, *(int(x) for x in c), 0.0, 0))


def build(out, idxs, colors, depths, c2w, cal, binary=False):
    """Writes the package under out/; returns the point-cloud statistics.
    binary: also COLMAP's binary model (write_binary), same content."""
    images, sparse = os.path.join(out, "images"), os.path.join(out, "sparse", "0")
    os.makedirs(images)
    os.makedirs(sparse)
    for i in idxs:
        os.symlink(os.path.abspath(colors[i]), os.path.join(images, "frame%06d.jpg" % i))
    write_cameras(os.path.join(sparse, "cameras.txt"), cal)
    write_images(os.path.join(sparse, "images.txt"), idxs, c2w)
    xyz, rgb, nrm, stats = make_points(idxs, colors, depths, c2w, cal)
    write_ply(os.path.join(sparse, "points3D.ply"), xyz, rgb, nrm)
    if binary:
        write_binary(sparse, idxs, c2w, cal, xyz, rgb)
    return stats
