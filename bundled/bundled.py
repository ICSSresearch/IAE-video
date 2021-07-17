from tqdm import tqdm
import cv2
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import copy
import math
import sharedmem
import joblib
from numba import njit
import numba as nb
import logging
from threadhandler import ThreadHandler, ProcessHandler
from time import *
import sys

try:
    from meshhomo import MeshHomo
    from meshwarp import MeshWarp
    from asap import Asap
except:
    from .meshhomo import MeshHomo
    from .meshwarp import MeshWarp
    from .asap import Asap


######################
# Config
######################
MAX_PROCESS = 1     # Rendering is using processes
MAX_THREAD = 12     # Track features is using thread

nb.config.NUMBA_THREADING_LAYER = "safe"
nb.config.NUMBA_NUM_THREADS = MAX_THREAD

mhomo = MeshHomo()
mwarp = MeshWarp()

# opencv
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

# constants
EYE_MATRIX = np.eye(3)
GT_SIGMA = 10   # omega computation: first term sigma
GM_SIGMA = 800  # omega computation: second term sigma

BUNDLED_MESH_SIZE = 8
BUNDLED_BATCH_SIZE = 15
BUNDLED_NUM_NEIGH_FRAMES = 70
BUNDLED_MAX_ITE = 10
BUNDLED_RIGIDITY = 2
BUNDLED_FPS = 30
BUNDLED_PADDING = 200
BUNDLED_NUM_FEATURES = 8000
BUNDLED_DISABLE_PROGRESS_BAR = False

BUNDLED_ADAPTIVE_LABMDA = False                 # no effect
BUNDLED_CROPPING_RATIO_THRESHOLD = 0.3
BUNDLED_DISTORTION_RATIO_THRESHOLD = 0.6        # no effect
BUNDLED_ADAPTIVE_MAXIMUM_ITERATION = 15         # no effect

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# TODO: Exception use bundled_logger: user decides whether to catch those
# TODO: Other levels should use logging root, and set level to error for now, user can decide to turn it on if they want
bundled_logger = logging.Logger(name="bundled", level=logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
bundled_logger.addHandler(handler)


def set_bundled_logger_filename(filename):
    logger_filehandler = logging.FileHandler(filename=filename, mode="a")
    logger_filehandler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    bundled_logger.addHandler(logger_filehandler)


def set_asap(height, width, quad_height, quad_width, asap_alpha, padding):
    asap = Asap(height, width, quad_height, quad_width, asap_alpha, padding)
    mhomo.set(asap)
    mwarp.set(asap)


def bar(name=None, iterator=None, unit="it", position=None, total=None, initial=0, disable=BUNDLED_DISABLE_PROGRESS_BAR):
    if total is None and iterator is None:
        raise TypeError("Iterator and Total are both None.")
    if total:
        return tqdm(total=total, unit=unit, ncols=100, desc=name, mininterval=0, position=position, initial=initial,
                    leave=True, disable=disable)
    else:
        return tqdm(iterator=iterator, ncols=100, desc=name, unit=unit, mininterval=0, position=position,
                    initial=initial, leave=True, disable=disable)


#########################
# Batch Helper
#########################


@njit(fastmath=True)
def compute_num_batch(num_iterations, batch_size):
    # get last batch size and number of batches required
    if batch_size > num_iterations: batch_size = num_iterations
    if batch_size == 0: batch_size = 1
    num_batch = int(math.ceil(num_iterations / batch_size))
    last_batch_size = num_iterations - batch_size * math.floor(num_iterations / batch_size)
    if last_batch_size == 0:
        last_batch_size = batch_size
    return num_batch, batch_size, last_batch_size


@njit(fastmath=True)
def get_batch_sizes(num_iterations, batch_size):
    # return a list of batch sizes in order
    if batch_size == 0: batch_size = 1
    (num_batch, batch_size, last_batch_size) = compute_num_batch(num_iterations, batch_size)
    batch_sizes = [batch_size for _ in range(num_batch)]
    if len(batch_sizes) > 0:
        batch_sizes[-1] = last_batch_size
    return batch_sizes


@njit(fastmath=True)
def get_frame_ids_from_batch_id(batch_id, num_batch, batch_size, last_batch_size):
    curr_bsize = batch_size if batch_id < num_batch - 1 else last_batch_size
    start = batch_id * batch_size
    frame_ids = list(range(start, start + curr_bsize))
    return frame_ids


###########################
# IO
############################


def load_video(path=None, vidcap=None, num_frames=None, target_height=None, target_width=None, start_frame=0, skip=0):
    if num_frames is None: num_frames = 2e16

    if path is not None:
        vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    vidnumframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, vidnumframes)
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if target_height is None else target_height
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) if target_width is None else target_width

    images = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    loadbar = bar(name="load_vid", total=num_frames, unit="fr")
    idx = 0
    for i in range(num_frames * (1 + skip)):
        ret, f = vidcap.read()
        if ret and i % (skip + 1) == 0:
            if target_height is not None and target_width is not None:
                f = cv2.resize(f, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            images[idx] = f
            idx += 1
        elif not ret:
            break
        loadbar.update()
    images = images[0:idx, ...]
    num_frames = idx
    return images, vidcap.get(cv2.CAP_PROP_FPS), num_frames


def display_video(frames):
    for idx, frame in enumerate(frames):
        cv2.imshow("frame", frame)
        cv2.waitKey()


def write(frames, vidwriter, comparison_frames=None):
    for frame_id in range(frames.shape[0]):
        warp_frame = np.uint8(frames[frame_id, ...])
        if comparison_frames is not None:
            warp_frame = np.concatenate([warp_frame, comparison_frames[frame_id, ...]], axis=0)
        vidwriter.write(warp_frame)


def load_features(frame_id, filepath=None, f=None):
    if filepath is not None:
        f = open(filepath, "rb")
    data = None
    for _ in range(frame_id+1):
        data = joblib.load(f)
    return data


def load_all_features(filepath):
    f = open(filepath, "rb")
    features = []
    while True:
        try:
            features.extend(joblib.load(f))
        except:
            break
    return features



######################################
# Features Detection
######################################


@njit(fastmath=True)
def norm_matrix(matrix, replace_matrix):
    if matrix[2, 2] == 0:
        return replace_matrix
    else:
        return matrix / matrix[2, 2]


def track_frame(args):
    (curridx, frames, detector, matcher, features, num_features) = args
    previdx = curridx - 1 if curridx != 0 else 0
    old_gray = cv2.cvtColor(frames[previdx, ...], cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frames[curridx, ...], cv2.COLOR_BGR2GRAY)
    kp_dst, des_dst = detector.detectAndCompute(frame_gray, None)
    kp_src, des_src = detector.detectAndCompute(old_gray, None)
    try:
        if des_dst is not None and des_src is not None and len(des_dst) >= 2 and len(des_src) >= 2:
            matches = matcher.knnMatch(des_src, des_dst, k=2)
        else:
            matches = []
    except Exception as e:
        bundled_logger.exception("track_frame: Unexpected matcher fail.")
        matches = []

    # Need to draw only good matches, so create a mask
    # matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good = [[] for _ in range(len(matches))]
    idx = 0
    for (i, kk) in enumerate(matches):
        if len(kk) == 2:
            (m, n) = kk
            if m.distance < 0.7 * n.distance:
                # matchesMask[i] = [1, 0]
                good[idx] = [m.distance, m]
                idx += 1
    good = good[0:idx]
    if num_features is not None:
        good.sort(key=lambda x: x[0])
        good = good[0:num_features]

    if len(good) >= 4:
        pts_src = np.array([kp_src[m[1].queryIdx].pt for m in good], dtype=np.float64)
        pts_dst = np.array([kp_dst[m[1].trainIdx].pt for m in good], dtype=np.float64)
        good_new = pts_dst
        good_old = pts_src
    else:
        good_new = np.array([])
        good_old = np.array([])

    features[curridx] = (good_old, good_new)


def track_batch(args):
    # normal batch size to compute the correct starting point for last batch
    (frame_ids, frames, detector, matcher, features, num_features) = args
    for frame_id in frame_ids:
        track_frame((frame_id, frames, detector, matcher, features, num_features))


#########################
# Estimator
#########################
@njit(fastmath=True)
def warp_to_find_inliers(pre_homo, currf, prvsf, dist_threshold, height, width):
    # stack currf [N x 2] with ones to make homogeneous coordinate ==> [3 x N]
    inv_pre_homo = inv(pre_homo)
    if inv_pre_homo is None: inv_pre_homo = EYE_MATRIX
    curr_warp = np.ascontiguousarray(inv_pre_homo) @ np.concatenate((np.transpose(currf), np.ones((1, currf.shape[0]))),axis=0)
    # normalize by the constant (third row)
    curr_warp = curr_warp/curr_warp[2, :]
    # grab the coordinate ==> [N x 2]
    curr_warp = np.transpose(curr_warp[0:2, :])
    # find inliers based on distance threshold
    dist_within_threshold = np.sum((prvsf-curr_warp) ** 2, axis=1) < dist_threshold
    warp_within_frame = (curr_warp[:, 0] > 0) & (curr_warp[:, 0] < height) & \
                        (curr_warp[:, 1] > 0) & (curr_warp[:, 1] < width)
    prvs_inliers = prvsf[dist_within_threshold & warp_within_frame, :]
    curr_inliers = curr_warp[dist_within_threshold & warp_within_frame, :]

    return prvs_inliers, curr_inliers


@njit(fastmath=True, nogil=True)
def update_curr_path_from_asap(orig_path, pre_homo, asap_homo, frame_idx, mesh_size, use_eye_matrix):
    # compute camera orig_path by warping it from previous camera position
    for meshrow in range(mesh_size):
        for meshcol in range(mesh_size):
            if use_eye_matrix:
                homo_at_mesh = EYE_MATRIX
            else:
                homo_at_mesh = pre_homo @ asap_homo[meshrow, meshcol]  # globalH * localH
                homo_at_mesh = norm_matrix(homo_at_mesh, replace_matrix=EYE_MATRIX)
            # get prev cam
            prev_cam_path = orig_path[frame_idx - 1, meshrow, meshcol, ...]
            # warp with previous path
            current_cam_path = homo_at_mesh @ prev_cam_path
            # normalize the matrix by the third row
            current_cam_path = norm_matrix(current_cam_path, prev_cam_path)
            # if singular, use prev_cam_path
            orig_path[frame_idx, meshrow, meshcol, ...] = current_cam_path if is_not_singular(current_cam_path) else prev_cam_path


#################################
# optimizer
#################################


@njit(fastmath=True)
def gaussian(x, sigma, mu=0):
    return np.exp(-((x-mu)**2) / (2.0 * (sigma**2)))


@njit(fastmath=True)
def update_weights(weights, start, end, curr_frame_idx, mrow, mcol, orig_path):
    # update weights from path[start] to path[end]
    for local_idx, neigh_frame_idx in enumerate(range(start, end)):
        # weights = Gaussian(abs(frame_idx-neigh_frame_idx))-Gaussian(norm2(C[neigh_frame_idx]-C[frame_idx]))
        # Note in original paper cam_pose_changes = |translation difference|
        cam_pose_changes = norm(orig_path[neigh_frame_idx, mrow, mcol, ...] -
                                orig_path[curr_frame_idx, mrow, mcol, ...])
        idx_diff = abs(neigh_frame_idx - curr_frame_idx)
        weights[local_idx] = gaussian(idx_diff, GT_SIGMA) * \
                             gaussian(cam_pose_changes, GM_SIGMA) if curr_frame_idx != neigh_frame_idx else 0


@njit(fastmath=True)
def is_not_singular(matrix):
    # assume matrix is square i.e. won't check rank
    cond_threshold = 1 / np.finfo(np.float64).eps
    r = np.linalg.cond(matrix, p=None)
    good_condition = r < cond_threshold
    return good_condition


@njit
def singular_error():
    raise ValueError("Singular Matrix? Shouldn't be. Check.")


@njit(fastmath=True)
def mesh_optimizer(fid, mrow, mcol, orig_path, kernel_size, mesh_size, num_frames, lambda_t, stable_w,
                   prev_opt_path, opt_path):
    # This algorithm is based on bundle path robust fitting algorithm
    # The objective function is derived based on Jacobi-based iteration
    # update omega which is associated with every neighboring frame (fid = frame idx)

    # get current mesh's original camera path: Ci(t)
    curr_mesh_orig_cam = orig_path[fid, mrow, mcol, ...]

    # determine who are current frame's neighbouring frames
    neighfr_hb = fid + kernel_size
    neighfr_lb = fid - kernel_size
    # move the boundary according if the other boundary needs more frames
    neighfr_hb += max(-neighfr_lb, 0)
    neighfr_lb -= max(neighfr_hb - num_frames, 0)
    # prevent the boundary from going beyond the scope
    neighfr_hb = min(neighfr_hb, num_frames)
    neighfr_lb = max(neighfr_lb, 0)
    num_neigh_frame = neighfr_hb - neighfr_lb

    # weights for neighbouring frames: used to optimize_path path
    weights = np.zeros(num_neigh_frame, dtype=np.float64)
    update_weights(weights, neighfr_lb, neighfr_hb, fid, mrow, mcol, orig_path)

    # get optimized path of neighbouring frames and reshape neighbour paths to [num_neighbours x 9] for computation
    neigh_frame_paths = np.ascontiguousarray(prev_opt_path[neighfr_lb:neighfr_hb, mrow, mcol, ...])
    neigh_frame_paths = neigh_frame_paths.reshape((num_neigh_frame, 9))

    # compute sum of 2*lambda_t*omega(r, t)*P_i(r) with r in neighbour frames (r != t)
    # r = neigh_frame_idx, t = frame_idx, lambda_t is a smoothness parameter
    # due to the reshape operation, the result is a 1 x 9 matrix. Reshape to [3 x 3]
    wsum_neigh_frame_paths = 2.0 * lambda_t * (weights @ neigh_frame_paths)
    wsum_neigh_frame_paths = wsum_neigh_frame_paths.reshape((3, 3))

    # determine neighouring cells around current mesh i (usually 8 meshes) p.s. + 2 due to exclusiveness of indexing
    nmrow_lb, nmrow_hb = max(0, mrow - 1), min(mrow + 2, mesh_size)
    nmcol_lb, nmcol_hb = max(0, mcol - 1), min(mcol + 2, mesh_size)

    # sum of 2*P_j*C_j^-1*Ci Note: Cj^-1*Ci isn't in the original paper, and is to replace the adaptive lambda
    wsum_neigh_mesh_paths = np.zeros((3, 3), dtype=np.float64)
    for nmrow in range(nmrow_lb, nmrow_hb):
        for nmcol in range(nmcol_lb, nmcol_hb):
            if nmrow != mrow and nmcol != mcol:
                curr_opt_neigh_cam = prev_opt_path[fid, nmrow, nmcol, ...]
                inv_curr_orig_neigh_cam = np.ascontiguousarray(inv(orig_path[fid, nmrow, nmcol, ...]))
                if inv_curr_orig_neigh_cam is None: singular_error()
                wsum_neigh_mesh_paths += curr_opt_neigh_cam @ inv_curr_orig_neigh_cam @ curr_mesh_orig_cam
    wsum_neigh_mesh_paths *= 2.0 * stable_w

    # compute gamma: 2 * lambda * sum(weights) + 2 * number of neighbouring meshes - 1
    num_neigh_mesh = (nmrow_hb - nmrow_lb) * (nmcol_hb - nmcol_lb) - 1.0
    gamma = 2.0 * lambda_t * np.sum(weights) + 2.0 * stable_w * num_neigh_mesh - 1.0

    # update opt path = 1/gamma * (ci + sum neighbouring frames + sum neighouring mesh)
    opt_path[fid, mrow, mcol, ...] = (curr_mesh_orig_cam + wsum_neigh_frame_paths + wsum_neigh_mesh_paths) / gamma


@njit
def optimize_frame(mesh_size, frame_id, orig_path, kernel_size, num_frames,
                   lambda_t, stable_w, last_iter_path, opt_path):
    for mrow in range(mesh_size):
        for mcol in range(mesh_size):
            mesh_optimizer(frame_id, mrow, mcol, orig_path, kernel_size, mesh_size, num_frames, lambda_t, stable_w,
                           last_iter_path, opt_path)


@njit
def optimize_batch(argv):
    (batch_id, last_iter_path, opt_path, orig_path, kernel_size, mesh_size,
     num_frames, lambda_t, stable_w, num_batch, batch_size, last_batch_size) = argv
    frame_ids = get_frame_ids_from_batch_id(batch_id, num_batch, batch_size, last_batch_size)
    for frame_id in frame_ids:
        optimize_frame(mesh_size, frame_id, orig_path, kernel_size,
                       num_frames, lambda_t, stable_w, last_iter_path, opt_path)


def optimize_batches(args):
    for i in range(len(args)):
        optimize_batch(args[i])


###################################
# warp
###################################


@njit(fastmath=True)
def is_inbound(mrow, mcol, mesh_size):
    within_bound = False
    if 0 <= mrow < mesh_size and 0 <= mcol < mesh_size:
        within_bound = True
    return within_bound


@njit(fastmath=True)
def get_neighbour_mesh(x, y, mesh_size):
    meshes = np.empty((9, 2), dtype=np.int32) # maximum to have 9 meshes
    idx = 0
    for i in range(x - 1, x + 1):
        for j in range(y - 1, y + 1):
            if is_inbound(i, j, mesh_size):
                meshes[idx, 0] = i
                meshes[idx, 1] = j
                idx += 1
    meshes = meshes[0:idx]
    return meshes


@njit(fastmath=True)
def warp_vertex(meshes, curr_frame_id, curr_vrow, curr_vcol, orig_path, opt_path, quad_width,
                quad_height, width):
    # return warped vertex based on its neighbouring vertices and itself's warp
    # warp vertex with path at each neighbouring mesh
    # then return the middle points of the nearest two points
    vertex_coord = np.array([curr_vcol * quad_width, curr_vrow * quad_height, 1])

    # warp current vertex (curr_vrow, curr_vcol) with each neighbouring mesh camera path (warp matrix)
    warped_points = np.empty((meshes.shape[0], 2), dtype=np.int32)
    for mid in range(meshes.shape[0]):
        x, y = meshes[mid, 0], meshes[mid, 1]
        inv_orig_path = np.ascontiguousarray(inv(orig_path[curr_frame_id, x, y, ...]))
        if inv_orig_path is None: singular_error()
        # warp vertex
        warp_m = opt_path[curr_frame_id, x, y, ...] @ inv_orig_path
        warped_vertex = warp_m @ vertex_coord
        warped_vertex[0:2] /= warped_vertex[2]      # normalize
        warped_points[mid][0] = warped_vertex[0]    # x
        warped_points[mid][1] = warped_vertex[1]    # y

    # get mean points of the warping restuls
    # find the closest two point
    # for one point, it will not go through the for loop: therefore [0,0]
    # for two points, this will return their indices
    # for four points, this will return the closest two points (the two has minimum distance)
    min_pt1_idx, min_pt2_idx = 0, 0
    min_d = width
    for idx_pt1 in range(warped_points.shape[0]):
        for idx_pt2 in range(warped_points.shape[0]):
            d = (warped_points[idx_pt1][0] - warped_points[idx_pt2][0]) ** 2 - \
                (warped_points[idx_pt1][1] - warped_points[idx_pt2][1]) ** 2
            if d < min_d:
                min_d = d
                min_pt1_idx, min_pt2_idx = idx_pt1, idx_pt2
    # compute the mean between the two points (for the same point, it will return the same point)
    vertex_x = 0.5 * (warped_points[min_pt1_idx][0] + warped_points[min_pt2_idx][0])
    vertex_y = 0.5 * (warped_points[min_pt1_idx][1] + warped_points[min_pt2_idx][1])
    return vertex_x, vertex_y


##################################################
# Stabilization Quality: Ratios larger the better
##################################################


@njit(fastmath=True)
def compute_cropping_ratio(orig_img, stab_img):
    # Note naive approach
    orig_img = np.sum(orig_img, axis=0)     # assume a tranposed image
    stab_img = np.sum(stab_img, axis=2)

    # ratio of their non-black pixels
    cratio = 1
    num_non_black_in_orig = np.sum(orig_img != 0)
    if num_non_black_in_orig != 0:
        cratio = np.sum(stab_img != 0) / num_non_black_in_orig

    return cratio


@njit(fastmath=True)
def compute_distortion_ratio(inv_orig_path_of_a_frame, opt_path_of_a_frame):
    # My guess is this is accounted by singular matrix check - so it is not very neccessary
    # TODO
    if inv_orig_path_of_a_frame is None:
        dratio = 0
    else:
        warp_transform = inv_orig_path_of_a_frame * opt_path_of_a_frame

        # get affine matrix
        affine_components = warp_transform[0:2, :]  # 2 x 3
        warp_transform = np.concatenate((affine_components, np.array([[0, 0, 1]])), axis=0)
        eigenvalues = np.linalg.eigvals(warp_transform)
        if eigenvalues is None:
            dratio = 0
            # bundled_logger.warning("eigenvalues are complex. Automatically set distortion ratio to 0")
        else:
            # find two largest
            eigenvalues = np.abs(eigenvalues)
            ids = eigenvalues.argsort()
            print(eigenvalues, ids)
            maximum, sec_maximum = eigenvalues[ids[-1]], eigenvalues[ids[-2]]
            if maximum != 0:
                # compute distortion ratio
                dratio = sec_maximum / maximum
            else:
                dratio = 0
            # if dratio > 1: dratio = 0
    return dratio


@njit(fastmath=True)
def generate_image_coord(height, width):
    # create coordinate grid
    x = np.zeros((width))
    for i in range(1, height):
        x = np.concatenate((x, np.ones((width)) * i))
    y = np.linspace(0, width-1, width)
    for i in range(height-1):
        y = np.concatenate((y, np.linspace(0, width-1, width)))
    ones = np.ones((height * width))
    npts = np.stack((x, y, ones), axis=0)
    return npts


####################
# PreProc
####################
def preproc(frames):
    return frames


####################
# Bundled Stablizer TODO: Adjustable parameters lambda_t, stable_w etc.
####################

class Bundled:
    def __init__(self, frames, mesh_size=BUNDLED_MESH_SIZE, padding=BUNDLED_PADDING, maxIte=BUNDLED_MAX_ITE,
                 num_neigh_frames=BUNDLED_NUM_NEIGH_FRAMES, rigidity=BUNDLED_RIGIDITY, output_path=None,
                 frame_rate=BUNDLED_FPS, batch_size=BUNDLED_BATCH_SIZE, num_features=BUNDLED_NUM_FEATURES):
        self.frames = preproc(frames)
        self.width = self.frames.shape[2]
        self.height = self.frames.shape[1]
        self.num_frames = self.frames.shape[0]
        self.outframes = sharedmem.empty((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        self.mesh_size = mesh_size
        self.vertex_size = mesh_size + 1                  # vertex has one more row and column than mesh
        self.quad_height = self.height/self.mesh_size     # height of each mesh
        self.quad_width = self.width/self.mesh_size       # width of each mesh
        self.default_batch_size = batch_size

        # check if quadrant height/width are integer
        assert self.quad_height.is_integer()
        assert self.quad_width.is_integer()

        self.padding = padding                            # padding for warping images
        self.maxIte = maxIte
        self.kernel_size = num_neigh_frames
        self.features = [None for _ in range(0, self.num_frames)]
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.num_features = num_features

        # initialize the first position with firstcam
        self.orig_path = sharedmem.empty((self.num_frames, self.mesh_size, self.mesh_size, 3, 3), dtype=np.float64)
        self.init_first_cam()
        self.opt_path = sharedmem.empty((self.num_frames, self.mesh_size, self.mesh_size, 3, 3), dtype=np.float64)

        # PARAMETERS
        self.num_batch, self.batch_size, self.last_batch_size = compute_num_batch(self.num_frames, batch_size)

        # as similar as possible homography estimation alpha
        self.asap_alpha = 3.0

        # initial homography distance threshold
        self.dist_threshold = 1000.0

        # parameter for optimize_path path
        self.lambda_t = 3.0  # smoothness
        self.stable_w = 20.0*rigidity   # Note: this is added to eliminate distortion

        set_asap(self.height, self.width, self.quad_height, self.quad_width, self.asap_alpha, self.padding)

        self.num_clips = 0

    def init_first_cam(self):
        for row in range(self.mesh_size):
            for col in range(self.mesh_size):
                self.orig_path[0, row, col, ...] = EYE_MATRIX

    def set_first_cam(self, firstcam):
        # initialize the first position with eye matrix
        assert self.orig_path[0, ...].shape == firstcam.shape
        self.orig_path[0, ...] = copy.deepcopy(firstcam)

    def update_frames(self, frames):
        self.frames = preproc(frames)

        if self.frames.shape[0] > self.num_frames:
            self.num_frames = self.frames.shape[0]
            self.features = [None for _ in range(0, self.num_frames)]
            self.outframes = sharedmem.empty((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
            self.orig_path = sharedmem.empty((self.num_frames, self.mesh_size, self.mesh_size, 3, 3), dtype=np.float64)
            self.init_first_cam()
            self.opt_path = sharedmem.empty((self.num_frames, self.mesh_size, self.mesh_size, 3, 3), dtype=np.float64)
            self.num_batch, self.batch_size, self.last_batch_size = compute_num_batch(self.num_frames, self.default_batch_size)
        elif self.frames.shape[0] < self.num_frames:
            # not need to reallocate
            self.num_frames = self.frames.shape[0]
            self.features = [None for _ in range(0, self.num_frames)]
            self.outframes = self.outframes[:self.num_frames, ...]
            self.orig_path = self.orig_path[:self.num_frames, ...]
            self.init_first_cam()
            self.opt_path = self.opt_path[:self.num_frames, ...]
            self.num_batch, self.batch_size, self.last_batch_size = compute_num_batch(self.num_frames, self.default_batch_size)

    def update_path(self, path):
        assert path.shape[0] == self.frames.shape[0]
        assert path.dtype == np.float64
        self.orig_path[:] = path
        self.opt_path[:] = self.orig_path

    def update_features(self, features):
        assert len(features) == self.frames.shape[0]
        self.features = features

    def track_features(self):
        trackbar = bar(total=self.num_batch, name="tracker_%02dthread" % MAX_THREAD, unit="bch")

        # create detector and matcher
        # detector = cv2.xfeatures2d_SURF.create()  # non-free xfeatures2d.
        # detector.setHessianThreshold(0)
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)  # SIFT/SURF
        detector = cv2.ORB_create(nfeatures=self.num_features)
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1) # ORB
        search_params = dict(checks=10)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # prepare arguments
        args = [None for _ in range(self.num_batch)]
        for batch_id in range(self.num_batch):
            frame_ids = get_frame_ids_from_batch_id(batch_id, self.num_batch, self.batch_size, self.last_batch_size)
            args[batch_id] = (frame_ids, self.frames, detector, flann, self.features, self.num_features)

        # call batch tracker
        if MAX_THREAD == 1:
            for arg in args:
                track_batch(arg)
                trackbar.update()
        else:
            threads = ThreadHandler(track_batch, args, MAX_THREAD, trackbar)
            threads.start()
        trackbar.close()
        return self.features

    def initialize_path(self):
        estbar = bar(total=self.num_frames - 1, name="estimate%02dthread" % MAX_PROCESS, unit=" fr")

        for frame_id in range(1, self.num_frames):
            self.initialize_frame(frame_id)
            estbar.update()

        # get rid of nan value in camera orig_path
        self.orig_path[np.isnan(self.orig_path)] = 0
        self.opt_path[:] = self.orig_path
        estbar.close()
        return self.orig_path

    def initialize_frame(self, frame_idx):
        homo = np.zeros([self.mesh_size, self.mesh_size, 3, 3], dtype=np.float64)
        pre_homo = EYE_MATRIX
        use_eye_matrix = True
        prvsf = self.features[frame_idx][0]
        currf = self.features[frame_idx][1]

        if len(prvsf) != 0 and len(currf) != 0:
            # pre warp (global homo) to find Inliers
            pre_homo, _ = cv2.findHomography(prvsf, currf, cv2.RANSAC, ransacReprojThreshold=1)
            if pre_homo is None: pre_homo = EYE_MATRIX
            prvs_inliers, curr_inliers = warp_to_find_inliers(pre_homo, currf, prvsf, self.dist_threshold,
                                                              self.height, self.width)
            # Homography for each mesh: AsSimilarAsPossible
            mhomo.computeHomos(np.ascontiguousarray(prvs_inliers[:, 0]), np.ascontiguousarray(prvs_inliers[:, 1]),
                               np.ascontiguousarray(curr_inliers[:,0]), np.ascontiguousarray(curr_inliers[:,1]), homo)
            use_eye_matrix = False
        update_curr_path_from_asap(self.orig_path, pre_homo, homo, frame_idx, self.mesh_size, use_eye_matrix)

    def optimize_path(self):
        optbar = bar(total=self.maxIte, name="optimize%02dthread" % MAX_PROCESS, unit="bch")
        for _ in range(self.maxIte):
            last_iter_path = copy.deepcopy(self.opt_path)
            args = [(batch_id, last_iter_path, self.opt_path, self.orig_path, self.kernel_size, self.mesh_size,
                     self.num_frames, self.lambda_t, self.stable_w, self.num_batch, self.batch_size,
                     self.last_batch_size) for batch_id in range(self.num_batch)]
            optimize_batches(args)
            optbar.update()

        optbar.close()
        return self.opt_path

    def render_frame(self, args):
        (frame_id, out_frames) = args
        out_frame_placeholder = np.ascontiguousarray(self.outframes[frame_id, ...], dtype=np.float32)
        for vrow in range(self.vertex_size):
            for vcol in range(self.vertex_size):
                # compute the 4 transforms at each mesh: warp the neighbour mesh with the mesh vertex
                neighbour_meshes = get_neighbour_mesh(vrow, vcol, self.mesh_size)
                projected_vertex = warp_vertex(neighbour_meshes, frame_id, vrow, vcol, self.orig_path, self.opt_path,
                                               self.quad_width, self.quad_height, self.width)
                mwarp.setVertex(vrow, vcol, projected_vertex[0], projected_vertex[1])

        # warp frame (meshFuncs.warp warp by each mesh)
        success = mwarp.warp(self.frames[frame_id, ...].astype(np.float32, copy=False), out_frame_placeholder)
        if not success:
            out_frame_placeholder = np.transpose(self.frames[frame_id, ...],(1,2,0))
        cratio = compute_cropping_ratio(self.frames[frame_id, ...], out_frame_placeholder)
        if cratio < BUNDLED_CROPPING_RATIO_THRESHOLD:
            out_frame_placeholder = np.transpose(self.frames[frame_id, ...], (1, 2, 0))
        out_frames[frame_id, ...] = out_frame_placeholder

    def render_batch(self, args):
        (batch_id, outframes) = args
        frame_ids = get_frame_ids_from_batch_id(batch_id, self.num_batch, self.batch_size, self.last_batch_size)
        for frame_id in frame_ids:
            self.render_frame((frame_id, outframes))

    def render(self):
        renderbar = bar(total=self.num_batch, name="renderer%02dthread" % MAX_PROCESS, unit="bch")

        # warp vertex and save them
        self.frames = np.transpose(self.frames, (0, 3, 1, 2))

        # prepare arguments
        args = [(batch_id, self.outframes) for batch_id in range(self.num_batch)]

        # render call on threads
        if MAX_PROCESS == 1:
            for arg in args:
                self.render_batch(arg)
                renderbar.update()
        else:
            threads = ProcessHandler(self.render_batch, args, MAX_PROCESS, renderbar)
            threads.start()
        renderbar.close()
        return self.outframes

    def write(self):
        """Write only stabilized videos"""
        if self.output_path is not None:
            vidwriter = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), self.frame_rate,
                                        (self.width, self.height), True)
            write(self.outframes, vidwriter)
        else:
            raise FileNotFoundError("Bundled: No path is provided. Can't write.")

    def write_both(self):
        """Original and stabilized side by side"""
        if self.output_path is not None:
            vidwriter = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), self.frame_rate,
                                        (self.width*2, self.height), True)
            frames = np.concatenate((np.transpose(self.frames, (0, 2, 3, 1)), self.outframes), axis=2)
            write(frames, vidwriter)
        else:
            raise FileNotFoundError("Bundled: No path is provided. Can't write.")

    def stablize(self):
        self.track_features()
        self.initialize_path()
        self.optimize_path()
        self.render()
        return self.outframes

    def save(self, pickle_filename):
        pass
