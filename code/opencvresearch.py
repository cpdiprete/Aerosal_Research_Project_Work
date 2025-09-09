import cv2
import numpy as np
import matplotlib.pyplot as plt
# from glob import glob
import IPython.display as ipd
import subprocess
from pynput import keyboard
import time
import ffmpeg

import plotting

#idea to find the bounding box:
# K-means clustering based on the frequencies of change (2 clusters) 
# 
##https://en.wikipedia.org/wiki/Periodogram
##https://en.wikipedia.org/wiki/Fourier_transform
##https://en.wikipedia.org/wiki/Spectral_density_estimation

step1_color = (0, 0, 255) # Red
step2_color = (255, 0, 255) # Pinkish
step3_color = (0, 255, 255) # yellowish
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 3.0
thickness = 2
lineType = cv2.LINE_AA
start_time_secs = 30
minute = 60
start_time_milliscnds = start_time_secs * 1000


def get_num_segments_fps(file):
    start_time = time.time()
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise IOError("Cannot open vid")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliscnds)
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # want to count the number of 60 second segments
    count = 0
    incrementing = 0
    while (incrementing < (frame_count - (minute * frames_per_second))):
        count += 1
        incrementing += (frames_per_second * 60)
    print(f"NUMBER OF SEGMENTS = {count}")
    endtime = time.time()
    print(f"------- num_segments took: {round(endtime - start_time, 2)} to run")
    return (count, frames_per_second)

def makeBox(file, grayscaled = False): 
    """The first stage of the video process. Takes in the users keyboard inputs to move/reshape a bounding box around the droplet formation.

    Args:
        file (mp4): The video that we want to count droplets for

    Raises:
        IOError: _description_

    Returns:
        (int, int, int): 
    """
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise IOError("Cannot open vid")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliscnds)
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total # of frames: {frame_count}")
    # print(f"Frames per second: {frames_per_second}")
    # print(f"Length of video: {frame_count / frames_per_second}")
    # cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliscnds)
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    height, width = frame.shape[:2]
    box_size = 50
    x, y = 0, 0
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print("---------------------")
    # print(img_height)
    # print(img_width)
    # print("---------------------")
    details = None
    
    while True:
        # make a copy so I can draw on the original without 
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = step1_color
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        vstart = img_height // 30
        hstart = img_width // 10
        vstep = img_height // 30
        
        cv2.putText(img, "D => move right | A => Move left | W => move up | S => Move down  ", (hstart, vstart), font, fontScale, step1_color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "V => shrink box, B => expand box", (hstart, (vstart + vstep)), font, fontScale, step1_color, thickness, lineType)
        cv2.putText(img, "Y => Done/Submit", (hstart, vstart + (2 * vstep)), font, fontScale, step1_color, thickness, lineType)
        if grayscaled:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.namedWindow('Gray Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gray Frame', 1600, 1000)
            cv2.moveWindow('Gray Frame', 0, 0)  # Move top-left corner to (x=100, y=50)
            cv2.imshow('Gray Frame', gray_image)
        else:
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Frame', 1600, 1000)
            cv2.moveWindow('Frame', 0, 0)  # Move top-left corner to (x=100, y=50)
            cv2.imshow('Frame', img)
            # cv2.imshow('Frame', img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            details = (x, y, box_size)
            break
        elif key == ord('d'):  # move right
            x = min(x + 50, width - 50)
        elif key == ord('a'):  # move left
            x = max(x - 50, 0)
        elif key == ord('s'):  # move down
            y = min(y + 50, height - 50)
        elif key == ord('w'):  # move up
            y = max(y - 50, 0)
        elif key == ord('b'):  # increase box size
            box_size += 50
        elif key == ord('v'):  # shrink box size
            box_size -= 50
        # else loop again

    cap.release()
    cv2.destroyAllWindows()  
    if details is None:
        print("-----------returned was never established-----------")
    return details

def userSelectsFrames(x, y, box_size, vid_file, greyscaled = False):
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliscnds)
    i = 0
    beginning_found = 0
    
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vstart = img_height // 25
    hstart = img_width // 3
    vstep = img_height // 35
    
    while not beginning_found:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = step2_color
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        
        # cv2.putText(img, "FIND THE MOMENT AFTER A DROPLET DISAPPEARS", (100, 15), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "Step 2: Finding when a droplet disappears", (hstart, vstart), font, fontScale, step2_color, thickness, lineType)
        cv2.putText(img, "R => Step forward in video (hold down to move faster)", (hstart, vstart + (vstep)), font, fontScale, step2_color, thickness, lineType)
        cv2.putText(img, "Y => Done/Submit", (hstart, vstart + (vstep * 2)), font, fontScale, step2_color, thickness, lineType)
        if greyscaled:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.namedWindow('grey', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('grey', 1600, 1000)
            cv2.moveWindow('grey', 0, 0)  # Move top-left corner to (x=100, y=50)
            cv2.imshow('grey', gray_image)
        else:
            cv2.namedWindow('Stage 2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stage 2', 1600, 1000)
            cv2.moveWindow('Stage 2', 0, 0)  # Move top-left corner to (x=100, y=50)
            cv2.imshow('Stage 2', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
                i += 1
        elif key == ord('y'):
                # print("ready to look for the end")
                beginning_found = i
                cv2.destroyAllWindows()
        elif key == ord("q"):
                return
            
    end_found = 0
    while not end_found:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = step3_color
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        cv2.putText(img, "Step 3: Find the NEXT moment a droplet disappears (1 full droplet formation cycle)", (hstart, vstart), font, fontScale, step3_color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "R => Step forward in video (hold down to move faster)", (hstart, vstart + vstep), font, fontScale, step3_color, thickness, lineType)
        cv2.putText(img, "Y => DONE/Submit", (hstart, vstart + (2 * vstep)), font, fontScale, step3_color, thickness, lineType)
        if greyscaled:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.namedWindow('Stage 3', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stage 3', 1600, 1000)
            cv2.moveWindow('Stage 3', 0, 0)  # Move top-left corner to (x=100, y=50)
            cv2.imshow('Stage 3', gray_image)
        else:
            cv2.namedWindow('Stage 3', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stage 3', 1600, 1000)
            cv2.moveWindow('Stage 3', 0, 0)  # Move top-left corner to (x=100, y=50)
            cv2.imshow('Stage 3', img)
            
        key = cv2.waitKey(0) & 0xFF
        # while (key != ord('r') or key != ord('y')):
        if key == ord('r'):
            i += 1
        elif key == ord('y'):
            # print("ready to look for the end")
            end_found = i
            # start_stop[1] = i
        elif key == ord("q"):
                return
    # print(f"START: {beginning_found} | STOP: {end_found}")
    formation_frame_time = end_found - beginning_found
    return formation_frame_time

def fast_rgbcount_ffmpeg(x, y, box_size, droplet_formation_time, vid_file, num_segments, fps, grayscaled):
    skip_secs = 30
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open vid")
    # cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliscnds)
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_frame_count -= (skip_secs * frames_per_second)
    cap.release()
    
    
# rgbcount_no_watch(x, y, box_size, droplet_formation_time, vid_file, num_segments, fps, grayscaled = False):
    start_time = time.time()
    # Get video metadata (dimensions, frame count, etc.)
    probe = ffmpeg.probe(vid_file)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = eval(video_stream['r_frame_rate'])  # e.g. "30/1" → 30.0
    num_frames = int(video_stream['nb_frames'])

    crop_expr = f"crop={box_size}:{box_size}:{x}:{y}"

    process = (
        ffmpeg
        .input(vid_file, ss=skip_secs)
        .filter('crop', box_size, box_size, x, y)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    b_vals, g_vals, r_vals = [], [], []

    num_pixels = box_size * box_size
    bytes_per_frame = num_pixels * 3  # RGB24 = 3 bytes per pixel

    while True:
        in_bytes = process.stdout.read(bytes_per_frame)
        if len(in_bytes) < bytes_per_frame:
            break  # End of stream
        frame = np.frombuffer(in_bytes, np.uint8).reshape((box_size, box_size, 3))
        r_vals.append(frame[:, :, 0].mean())
        g_vals.append(frame[:, :, 1].mean())
        b_vals.append(frame[:, :, 2].mean())
    process.wait()

    b_vals = np.array(b_vals)
    g_vals = np.array(g_vals)
    r_vals = np.array(r_vals)
    end_time = time.time()
    print(f"------- ffmpeg took: {round(end_time - start_time, 2)} to run")
    print(f"Extracted {len(r_vals)} frames at ROI [{x}, {y}] with box size {box_size}")
    # if grayscaled:
    #     plotting.makegrayPlot(gray_box_vals, droplet_formation_time)
    droplet_vals = b_vals + g_vals + r_vals
    # plotting.makePlot(droplet_vals, droplet_formation_time, num_segments, fps)
    # droplet_formation_time *= (newvidlen/normalvidlen)
    droplet_formation_time *= (len(droplet_vals)/cap_frame_count)
    droplet_formation_time = int(droplet_formation_time)
    print(f"Normalized DROPLET FOMRATION TIME: {droplet_formation_time}")
    plotting.makeInteractivePlot(droplet_vals, droplet_formation_time, num_segments, fps)
    
    # def makeInteractivePlot(droplet_vals, droplet_formation_time, num_segments, fps):


def rgbcount_no_watch(x, y, box_size, droplet_formation_time, vid_file, num_segments, fps, grayscaled = False):
    start_time = time.time()
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliscnds)
    b_vals, g_vals, r_vals = [], [], []
    gray_box_vals = []
    start_reading_frames_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscaled:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_box_mean = frame[y:y+box_size, x:x+box_size].mean()
            gray_box_vals.append(gray_box_mean)
        else:
            b_mean = frame[y:y+box_size, x:x+box_size, 0].mean()
            g_mean = frame[y:y+box_size, x:x+box_size, 1].mean()
            r_mean = frame[y:y+box_size, x:x+box_size, 2].mean()
            
            b_vals.append(b_mean)
            g_vals.append(g_mean)
            r_vals.append(r_mean)

    cap.release()
    cv2.destroyAllWindows()
    time_bf_numpy_conversion = time.time()
    print(f"------- Reading the actual rgb vals took: {round(time_bf_numpy_conversion - start_reading_frames_time, 2)} to run")
    b_vals = np.array(b_vals)
    g_vals = np.array(g_vals)
    r_vals = np.array(r_vals)
    gray_box_vals = np.array(gray_box_vals)
    print(f"------- Dumbass numpy conversion took: {round(time.time() - time_bf_numpy_conversion, 2)} to run")
    print("Finished reading video — plotting now.")
    endtime = time.time()
    print(f"------- rgb_count_no_watch took: {round(endtime - start_time, 2)} to run")
    
    if grayscaled:
        plotting.makegrayPlot(gray_box_vals, droplet_formation_time)
    else:
        droplet_vals = b_vals + g_vals + r_vals
        # plotting.makePlot(droplet_vals, droplet_formation_time, num_segments, fps)
        plotting.makeInteractivePlot(droplet_vals, droplet_formation_time, num_segments, fps)

def countJumps(total, median, droplet_formation_time):
    min_val = np.min(total)
    medianMinusQ1 = median - np.quantile(total, .25)
    medianMinusmin = median - min_val
    # print(f"Min is {min:.2f}, change check is {medianMinusmin:.2f}")
    count = 0
    droplet_disappear_x_coords = []
    droplet_disappear_y_coords = []
    droplet_fully_formed_y_coords = []
    droplet_fully_formed_x_coords = []
    i = 0
    while i < len(total):
        cur_max = (0, 0)
        for j in range(i, len(total)):
            if ((j - i) > (20)): ## cuts it of when its over 20 frames (from testing this seemed to be the appropriate gap), this would be filled in with the inputs if they frame pick a full droplet cycle
                break
            if total[j] > cur_max[1]:
                cur_max = (j, total[j])
            if (total[j] > (total[i] + (medianMinusQ1 * (2/3)))): #finds the first candidate for a max point, need to check the ones directly following it
            # if (total[j] > (total[i] + (medianMinusQ1))): #finds the first candidate for a max point, need to check the ones directly following itQ
                start = j
                # check some spots after to ensure I have the actual maximum
                while (start < (j + (droplet_formation_time* 0.7))) and (start < len(total)):
                    if total[start] > cur_max[1]:
                        cur_max = (start, total[start])
                    start += 1
                count += 1
                droplet_disappear_y_coords.append(cur_max[1])
                droplet_disappear_x_coords.append(cur_max[0])
                # local_max_y_coords.append(total[i][1])
                # local_max_x_coords.append(total[i][0])
                i = cur_max[0]
                break
        i += 1  
    ## want to find the mins preceding all of the x_cord droplet dispersions
    for x in range(len(droplet_disappear_x_coords)):
        peak = droplet_disappear_y_coords[x]
        peak_coord = droplet_disappear_x_coords[x]
        c_min = (peak_coord, peak)
        t = droplet_disappear_x_coords[x]
        # if (droplet_disappear_x_coords[x] == debugging_last_coord[0]):
        #     print("want to check behavior after this")
        while ((t > 0) and (t > (peak_coord - (droplet_formation_time * 1.01)))):
            if (total[t] <= c_min[1]):
                c_min = (t, total[t])
            # if (droplet_disappear_x_coords[x] == debugging_last_coord[0]):
                # print(f"Stopping conditions (t, peak_coord, droplet_formation_time) = ({t}, {peak_coord}, {droplet_formation_time})")
            t -= 1
        if (t > 0): ## testing this because when a segment was cut in half it would plot twice, I think the play is to check if the dropley fully fomerf
            droplet_fully_formed_y_coords.append(c_min[1]) 
            droplet_fully_formed_x_coords.append(c_min[0])
    # print(f"Number of droplets formed and then dissolved in this video: {count}")
    # print(len(total))
    # return local_max_x_coords
    # print(f"Plotted x_coords = : {dude_x_coords}")
    return (droplet_fully_formed_x_coords, droplet_fully_formed_y_coords, droplet_disappear_x_coords, droplet_disappear_y_coords, len(droplet_fully_formed_y_coords))
    # return (local_max_x_coords, local_max_y_coords, count)

#Feedback from meeting (general flow of how it should work):

#Ideally he wants to be prompted to upload an *individual* video instead of having to add file location manually in code
# code will identify length of video, and start counting 30 seconds after start of video
# record droplets per minute after this 30 seconds discard
# return a list of these droplets per minute

#at the end of the video (changes flow rate). So each individual video will be a single flow rate.

#data hes looking ot collect: droplets per minute for like 3 minutes, and then average the figures/numbers for that particular flow rate

#Normalize individual peaks( )

def run_video_file(fileName):
    num_segments, fps = get_num_segments_fps(fileName)
    box_details = makeBox(fileName, False) ## ry to make the diag boz
    droplet_formation_time = userSelectsFrames(box_details[0], box_details[1], box_details[2], fileName, False)
    # rgbcount_no_watch(box_details[0], box_details[1], box_details[2], droplet_formation_time, vid_file, num_segments, fps, False)
    fast_rgbcount_ffmpeg(box_details[0], box_details[1], box_details[2], droplet_formation_time, fileName, num_segments, fps, False)

# if __name__ == "__main__":
    # # vid_file = "./labvid.mov"
    # # vid_file = "../videos/Shortvid.mov"
    # # vid_file = "../videos/without_start_blur.mp4"  #it says theres <153> drops in this
    # # vid_file = "../videos/30UL.mov"  #it says theres <153> drops in this
    # # vid_file = "../videos/40UL.mov"  #it says theres <153> drops in this
    # # vid_file = "../videos/50UL.mov"  #it says theres <153> drops in this
    
    # vid_file = "../videos/100UL.MOV"  # THIS ONE HAS A CRAZY RGB DIP AT THE END, ID ASSUME THE LIGHTS DIM OR SOMETHING
    # # vid_file = "./first_2_mins_76_drops.mp4" # code says there's <76> drops in this
    # # vid_file = "./last_2_mins.mp4" # code says theres 77 drops in this
    # # details = makeBox(vid_file) ## ry to make the diag boz
    # # window_test()
    # num_segments, fps = get_num_segments_fps(vid_file)
    # box_details = makeBox(vid_file, False) ## ry to make the diag boz
    # droplet_formation_time = userSelectsFrames(box_details[0], box_details[1], box_details[2], vid_file, False)
    
    # # fast_rgbcount_ffmpeg(box_details[0], box_details[1], box_details[2], vid_file)
    # # (x, y, box_size, vid_file)
    # # rgbcount_no_watch(box_details[0], box_details[1], box_details[2], droplet_formation_time, vid_file, num_segments, fps, False)
    # fast_rgbcount_ffmpeg(box_details[0], box_details[1], box_details[2], droplet_formation_time, vid_file, num_segments, fps, False)
    
    
    
