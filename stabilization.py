import logging
import sys
import cv2
import numpy as np
from time import perf_counter, time
import joblib

from bundled import Bundled, get_batch_sizes, load_video, display_video, BUNDLED_MESH_SIZE

mylogger = logging.Logger(name="mylogger", level=logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
mylogger.addHandler(handler)


def estimate_motions(vidpath, batch_size, mesh_size, target_height=None, target_width=None):
    s = time()
    # create vidreader
    vidcap = cv2.VideoCapture(vidpath)
    # number of frames
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # create json to store features and paths
    feat_file = open('.'.join(vidpath.split('.')[0:-1]) + ".features", "wb")

    # create paths to store all paths
    paths = np.zeros([num_frames, mesh_size, mesh_size, 3, 3], dtype=np.float64)

    # create frames to store
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if target_height is None else target_height
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) if target_width is None else target_width
    frames = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

    # compute number of batches: new window will take one more frame from previous batch as the first frame
    if batch_size > num_frames: batch_size = num_frames
    batches = [batch_size]
    batches.extend(get_batch_sizes(num_frames - batch_size, batch_size - 1))

    # create bundled for this movie and initialize first cam
    bundled = Bundled(frames)
    paths[0, ...] = bundled.orig_path[0, ...]
    break_at_the_end = False
    total_num_frame_loaded = 0
    curr_bid = 0
    try:
        for idx, bsize in enumerate(batches):
            curr_bid = idx
            # load bsize # frames
            loaded_frames, _, num_loaded = load_video(vidcap=vidcap, num_frames=bsize, target_width=target_width,
                                                      target_height=target_height)
            if num_loaded == 0: break
            #  break this loop after this iteration if video ends before frame count reaches (wrong frame count)
            if bsize != num_loaded: break_at_the_end = True
            bsize = num_loaded

            # get local and global indices of curr window lb = lower bound of new frame, ub = upper bound of new frame
            num_retained = int(bool(idx))
            local_ub = num_retained + bsize
            local_ub = num_retained + bsize
            global_lb = idx * (batch_size - 1) + num_retained
            global_ub = global_lb + bsize
            frames[num_retained:local_ub] = loaded_frames

            # assign last camera position from last batch to first camera position
            bundled.update_frames(frames[:local_ub, ...])
            bundled.set_first_cam(paths[global_lb - num_retained, ...])

            # track and compute path
            features = bundled.track_features()
            paths[global_lb-num_retained:global_ub, ...] = bundled.initialize_path()
            # save new features - to load this, read line by line and do json.load
            joblib.dump(features[num_retained:local_ub], feat_file)

            # assign last read frame to first frame
            frames[0, ...] = frames[local_ub-1, ...]
            total_num_frame_loaded += bsize
            if break_at_the_end:
                mylogger.warning("EOF before total frame count reached at batch %05d / %05d completed"
                                       "%07d / %07d" % (idx+1, len(batches), total_num_frame_loaded, num_frames))
                break
    except:
        mylogger.exception("%s at batch %05d / %05d with bsize = %05d" % (vidpath, curr_bid+1, len(batches), batch_size-1))

    # save path
    joblib.dump(paths[:total_num_frame_loaded], '.'.join(vidpath.split('.')[0:-1]) + ".paths", compress=0)
    # close features file
    feat_file.close()
    mylogger.info("%s - %.4f FPS" % (vidpath, (total_num_frame_loaded / (time() - s))))
    return paths, vidcap


def stablize_video(vid_path, target_height, target_width, output_dir):
    mylogger.info("Stablizing %s" % vid_path)
    start_frame = 0
    start_time = perf_counter()
    loaded_frames, fps, num_loaded = load_video(path=vid_path, target_height=target_height, target_width=target_width,
                                                start_frame=start_frame, skip=0)
    bundled = Bundled(loaded_frames, frame_rate=fps,
                      output_path='%s/stab-%s' % (output_dir, vid_path.split('/')[-1]))
    bundled.stablize()
    fps = num_loaded / (perf_counter() - start_time)
    bundled.write_both()
    # uncomment to display frame by frame, navigate by pressing any key
    # display_video(bundled.outframes)
    mylogger.info("Done: FPS %02.2f" % fps)


if __name__ == "__main__":
    import os

    PROCESS_BATCH_SIZE = 33216

    command = sys.argv[1]   # "stablize" or "extract"
    file_or_dir = sys.argv[2]

    if command == 'stablize' and os.path.isfile(file_or_dir):
        outpath = sys.argv[3]
        twidth, theight = int(sys.argv[5]), int(sys.argv[4])
        stablize_video(file_or_dir, theight, twidth, outpath)

    elif command == 'stablize' and os.path.isdir(file_or_dir):
        outpath = sys.argv[3]
        twidth, theight = int(sys.argv[5]), int(sys.argv[4])
        files = [os.path.join(root, file) for root, _, files in os.walk(file_or_dir) for file in files if
                 file.endswith('.mp4')]
        for vpath in files:
            stablize_video(vpath, theight, twidth, outpath)

    elif command == 'extract' and os.path.isfile(file_or_dir):
        twidth, theight = int(sys.argv[4]), int(sys.argv[3])
        estimate_motions(file_or_dir, batch_size=PROCESS_BATCH_SIZE, mesh_size=BUNDLED_MESH_SIZE,
                         target_height=theight, target_width=twidth)

    elif command == 'extract' and os.path.isdir(file_or_dir):
        twidth, theight = int(sys.argv[4]), int(sys.argv[3])
        files = [os.path.join(root, file) for root, _, files in os.walk(file_or_dir) for file in files if
                 file.endswith('.mp4')]
        failed_cases = []
        mylogger.info('%s cases, %s' % (len(files), str(files)))
        for fileid, vpath in enumerate(files):
            print("Extracting", vpath)
            mylogger.info('%05d/%05d Extracting %s' % (fileid+1, len(files), vpath))
            try:
                estimate_motions(vpath, batch_size=PROCESS_BATCH_SIZE, mesh_size=BUNDLED_MESH_SIZE,
                                 target_height=theight, target_width=twidth)
            except:
                mylogger.exception("In %s" % vpath)
                failed_cases.append(vpath)

        if failed_cases:
            mylogger.info("These cases failed: %s" % str(failed_cases))

    else:
        raise ValueError("Does not recognize command %s or file/dir %s is invalid." % (command, file_or_dir))
