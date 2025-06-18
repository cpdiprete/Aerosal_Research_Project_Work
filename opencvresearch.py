import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd
import subprocess
from pynput import keyboard


def makeBox(file):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise IOError("Cannot open vid")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    height, width = frame.shape[:2]
    box_size = 50
    x, y = 0, 0
    details = None
    
    while True:
        # make a copy so we can draw on the original
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = (0, 0, 200)
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # fontScale
        fontScale = 0.5
        
        # Black color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 2
        lineType = cv2.LINE_AA
        
        cv2.putText(img, "R => move right | L => Move left | U => move up | D=> Move down  ", (100, 15), font, fontScale, color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "S => shrink box, W => expand box", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "Y => Run", (100, 45), font, fontScale, color, thickness, lineType)
        cv2.imshow('Frame', img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            details = (x, y, box_size)
            break
        elif key == ord('r'):  # move right
            x = min(x + 50, width - 50)
        elif key == ord('l'):  # move left
            x = max(x - 50, 0)
        elif key == ord('d'):  # move down
            y = min(y + 50, height - 50)
        elif key == ord('u'):  # move up
            y = max(y - 50, 0)
        elif key == ord('w'):  # increase box size
            box_size += 50
        elif key == ord('s'):  # shrink box size
            box_size -= 50
        # else loop again

    cap.release()
    cv2.destroyAllWindows()  
    if details is None:
        print("-----------returned was never established-----------")
    return details


def userSelectsFrames(x, y, box_size, vid_file):
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    i = 0
    beginning_found = 0
    
    while not beginning_found:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = (0, 0, 200)
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1.0
        # Black color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 3
        lineType = cv2.LINE_AA
        
        cv2.putText(img, "FIND THE MOMENT AFTER A DROPLET DISAPPEARS", (100, 15), font, fontScale, color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "R => ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "Y => SUBMIT YAYY", (100, 45), font, fontScale, color, thickness, lineType)
        cv2.imshow('Select the statr of image', img)
        
        key = cv2.waitKey(0) & 0xFF
        while key != ord('r') or key != ord('y'):
            if key == ord('r'):
                i += 1
                break
            elif key == ord('y'):
                print("ready to look for the end")
                beginning_found = i
                # start_stop[0] = i
                break
            elif key == ord("q"):
                return
            
    end_found = 0
    while not end_found:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = (0, 0, 200)
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1.0
        # Black color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 3
        lineType = cv2.LINE_AA
        
        cv2.putText(img, "FIND it again mf", (100, 15), font, fontScale, color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "R => ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "Y => SUBMIT YAYY", (100, 45), font, fontScale, color, thickness, lineType)
        cv2.imshow('2nd stage', img)
        key = cv2.waitKey(0) & 0xFF
        while (key != ord('r') or key != ord('y')):
            if key == ord('r'):
                i += 1
                break
            elif key == ord('y'):
                print("ready to look for the end")
                end_found = i
                # start_stop[1] = i
                break
            elif key == ord("q"):
                return
    print("WE READY BITH-----------------------------------")
    print(f"START: {beginning_found} | STOP: {end_found}")
    return (beginning_found, end_found)
        
    
def rgbCount_watch_vid(x, y, box_size, vid_file):

    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    
    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"height: {height}, width: {width}")

    b_vals, g_vals, r_vals = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # img[y:y+box_size, x:x+box_size] = (0, 0, 200)
        # compute per-channel mean
        # frame[y+10:y+box_size - 10, x+10:x+box_size - 10] = (200, 0, 0)
        
        b_mean = frame[y:y+box_size, x:x+box_size, 0].mean()
        g_mean = frame[y:y+box_size, x:x+box_size, 1].mean()
        r_mean = frame[y:y+box_size, x:x+box_size, 2].mean()

        b_vals.append(b_mean)
        g_vals.append(g_mean)
        r_vals.append(r_mean)

        coppp = frame.copy()
        keep = frame[y+10:y+box_size - 10, x+10:x+box_size - 10]
        coppp[y:y+box_size, x:x+box_size] = (200, 0, 0)
        coppp[y+10:y+box_size - 10, x+10:x+box_size - 10] = keep
        
        
        cv2.imshow('Frame', coppp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    b_vals = np.array(b_vals)
    g_vals = np.array(g_vals)
    r_vals = np.array(r_vals)
    print("Finished reading video — plotting now.")
    makePlot(b_vals, g_vals, r_vals)

def rgbcount_no_watch(x, y, box_size, vid_file):
    # cap = cv2.VideoCapture("./Shortvid.mov")
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    
    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"height: {height}, width: {width}")

    b_vals, g_vals, r_vals = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        b_mean = frame[y:y+box_size, x:x+box_size, 0].mean()
        g_mean = frame[y:y+box_size, x:x+box_size, 1].mean()
        r_mean = frame[y:y+box_size, x:x+box_size, 2].mean()

        b_vals.append(b_mean)
        g_vals.append(g_mean)
        r_vals.append(r_mean)

        coppp = frame.copy()
        keep = frame[y+10:y+box_size - 10, x+10:x+box_size - 10]
        coppp[y:y+box_size, x:x+box_size] = (200, 0, 0)
        coppp[y+10:y+box_size - 10, x+10:x+box_size - 10] = keep
        

    cap.release()
    cv2.destroyAllWindows()
    b_vals = np.array(b_vals)
    g_vals = np.array(g_vals)
    r_vals = np.array(r_vals)
    print("Finished reading video — plotting now.")
    makePlot(b_vals, g_vals, r_vals)
    
    
def makePlot(b_vals, g_vals, r_vals):
    total = b_vals + g_vals + r_vals
    mean = np.mean(total)
    median = np.median(total)
    print("mean: ", mean)
    print("median", median)
    x_axis = countJumps(total, median)
    # mean = np.ones(np.shape(total)) * mean
    median = np.ones(np.shape(total)) * median
    frames = np.arange(len(total))  # x-axis: frame index/time
    # frames = np.arange(len(b_vals))  # x-axis: frame index/time
    # plt.figure(figsize=(10, 4))
    plt.figure(figsize=(80, 8))
    
    #uncomment these lines to see individual rgb values
    # ------------------------------------------------------------------
    # plt.plot(frames, r_vals, label='Red', color = 'r')
    # plt.plot(frames, g_vals, label='Green', color = 'g')
    # plt.plot(frames, b_vals, label='Blue', color = 'b')
    #-------------------------------------------------------------------
    plt.plot(frames, total, label='RGB', color = 'm')
    # plt.plot(frames, mean, label="Midpoint", color='y')
    plt.plot(frames, median, label="Midpoint", color='y')
    plt.xlabel('Frame index')
    plt.ylabel('Mean channel value')
    plt.title('RGB Mean over Time')
    for x_spot in x_axis:
        plt.axvline(x=x_spot, color='r', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def countJumps(total, mean):
    min = np.min(total)
    mean -= min
    print(f"Min is {min}, change check is {mean*(2/3)}")
    count = 0
    local_max_x_coords = []
    i = 0
    while i < len(total):
        # print(f"i: {i}")
        cur_max = (0, 0)
        for j in range(i, len(total)):
            if ((j - i) > 35): ## cuts it of when its over 35 frames (from testing this seemed to be the appropriate gap), this would be filled in with the inputs if they frame pick a full droplet cycle
                break
            if total[j] > cur_max[1]:
                cur_max = (j, total[j])
            if (total[j] > (total[i] + mean*(2/3))): #finds the first candidate for a max point, need to check the ones directly following it
                start = j
                while start < (j + 20) and start < len(total):
                    if total[start] > cur_max[1]:
                        cur_max = (start, total[start])
                    start += 1
                
                count += 1
                # local_max_x_coords.append(i)
                local_max_x_coords.append(cur_max[0])
                # print(f"i: {i} || j: {j}")
                # i = j ----------------------------
                i = cur_max[0]
                # print(f"new i: {i}")
                break
        i+= 1
            
    print(count)
    print(len(total))
    return local_max_x_coords
    
            
    

if __name__ == "__main__":
    # rgbCount()
    startx = 0
    starty = 0
    # vid_file = "./labvid.mov"
    # vid_file = "./Shortvid.mov"
    vid_file = "./without_start_blur.mp4"  #it says theres <154> drops in this
    # vid_file = "./first_2_mins_76_drops.mp4" # code says there's <77> drops in this
    # vid_file = "./last_2_mins.mp4" # code says theres 77 drops in this
    # details = makeBox("./Shortvid.mov") ## ry to make the diag boz
    details = makeBox(vid_file) ## ry to make the diag boz
    start, finish = userSelectsFrames(details[0], details[1], details[2], vid_file)
    rgbcount_no_watch(details[0], details[1], details[2], (start, finish) vid_file)
    
    
    
