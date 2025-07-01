import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd
import subprocess
from pynput import keyboard

step1_color = (0, 0, 255) # Red
step2_color = (255, 0, 255) # Pinkish
step3_color = (0, 255, 255) # yellowish
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
thickness = 2
lineType = cv2.LINE_AA

def makeBox(file): 
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
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    height, width = frame.shape[:2]
    box_size = 50
    x, y = 0, 0
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    details = None
    
    while True:
        # make a copy so I can draw on the original without 
        img = frame.copy()
        img[y:y+box_size, x:x+box_size] = step1_color
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        vstart = img_height // 30
        hstart = img_width // 3
        vstep = img_height // 30
        
        cv2.putText(img, "R => move right | L => Move left | U => move up | D=> Move down  ", (hstart, vstart), font, fontScale, step1_color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "S => shrink box, W => expand box", (hstart, (vstart + vstep)), font, fontScale, step1_color, thickness, lineType)
        cv2.putText(img, "Y => Done/Submit", (hstart, vstart + (2 * vstep)), font, fontScale, step1_color, thickness, lineType)
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
        cv2.imshow('Stage 2', img)
        
        key = cv2.waitKey(0) & 0xFF
        while key != ord('r') or key != ord('y'):
            if key == ord('r'):
                i += 1
                break
            elif key == ord('y'):
                print("ready to look for the end")
                beginning_found = i
                # start_stop[0] = i
                cv2.destroyAllWindows()
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
        img[y:y+box_size, x:x+box_size] = step3_color
        copyright = frame.copy()[y+10:y+box_size - 10, x+10:x+box_size - 10]
        img[y+10:y+box_size - 10, x+10:x+box_size - 10] = copyright
        
        cv2.putText(img, "Step 3: Find the NEXT moment a droplet disappears (1 full droplet formation cycle)", (hstart, vstart), font, fontScale, step3_color, thickness, lineType)
        # cv2.putText(img, "U => move up, D=> Move down ", (100, 30), font, fontScale, color, thickness, lineType)
        cv2.putText(img, "R => Step forward in video (hold down to move faster)", (hstart, vstart + vstep), font, fontScale, step3_color, thickness, lineType)
        cv2.putText(img, "Y => DONE/Submit", (hstart, vstart + (2 * vstep)), font, fontScale, step3_color, thickness, lineType)
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

def rgbcount_no_watch(x, y, box_size, start_finish, vid_file):
    # cap = cv2.VideoCapture("./Shortvid.mov")
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise IOError("Cannot open video")

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

    cap.release()
    cv2.destroyAllWindows()
    b_vals = np.array(b_vals)
    g_vals = np.array(g_vals)
    r_vals = np.array(r_vals)
    print("Finished reading video — plotting now.")
    makePlot(b_vals, g_vals, r_vals, start_finish)
    # blackScreenWithCount(b_vals, g_vals, r_vals, start_finish) # uncomment this to just see the black screen
    
    
def makePlot(b_vals, g_vals, r_vals, start_finish):
    total = b_vals + g_vals + r_vals
    mean = np.mean(total)
    median = np.median(total)
    lower_Q = np.quantile(total, .25)
    
    window_size = (start_finish[1] - start_finish[0]) // 3
    total = np.convolve(total, np.ones(window_size)/window_size, mode='valid')
    print("mean: ", mean)
    print("median", median)
    x_axis = countJumps(total, median, start_finish)[0]
    # mean = np.ones(np.shape(total)) * mean
    median = np.ones(np.shape(total)) * median
    lower_Q = np.ones(np.shape(total)) * lower_Q
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
    plt.plot(frames, lower_Q, label="Q1", color='b')
    plt.xlabel('Frame index')
    plt.ylabel('Mean channel value')
    plt.title('RGB Mean over Time')
    for x_spot in x_axis:
        plt.axvline(x=x_spot, color='r', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def countJumps(total, mean, start_finish):
    min = np.min(total)
    len_drop_formation = start_finish[1] - start_finish[0]
    print(f"Length of droplet formation: {len_drop_formation}")
    meanminusQ1 = mean - np.quantile(total, .25)
    meanminusmin = mean - min
    print(f"Min is {min}, change check is {meanminusmin}")
    count = 0
    local_max_x_coords = []
    i = 0
    while i < len(total):
        cur_max = (0, 0)
        for j in range(i, len(total)):
            if ((j - i) > (20)): ## cuts it of when its over 20 frames (from testing this seemed to be the appropriate gap), this would be filled in with the inputs if they frame pick a full droplet cycle
                break
            if total[j] > cur_max[1]:
                cur_max = (j, total[j])
                
            # if (total[j] > (total[i] + meanminusmin*(1/3))): #finds the first candidate for a max point, need to check the ones directly following it
            if (total[j] > (total[i] + (meanminusQ1 * (2/3)))): #finds the first candidate for a max point, need to check the ones directly following it
                start = j
                # check some spots after to ensure I have the actual maximum
                while (start < (j + (len_drop_formation * 0.7))) and (start < len(total)):
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
            
    print(f"Number of droplets formed and then dissolved in this video: {count}")
    # print(len(total))
    # return local_max_x_coords
    return (local_max_x_coords, count)
    
def blackScreenWithCount(b_vals, g_vals, r_vals, start_finish):
    total = b_vals + g_vals + r_vals
    mean = np.mean(total)
    median = np.median(total)
    
    window_size = (start_finish[1] - start_finish[0]) // 3
    total = np.convolve(total, np.ones(window_size)/window_size, mode='valid')
    print("mean: ", mean)
    print("median", median)
    number = str(countJumps(total, median, start_finish)[1])
    
    blackscreen = np.zeros((1000, 2000, 3))
    displayString = "Number of droplets formed in this video: " + number
    # cv2.putText(blackscreen, "Hiiiiiiiii", 500, 2)
    # cv2.putText(blackscreen, "Y => Done/Submit", (300, 400), font, fontScale, (255, 255, 255), thickness, lineType)
    cv2.putText(blackscreen, displayString, (100, 200), font, 2.0, (255, 255, 255), thickness, lineType)
    cv2.putText(blackscreen, "Press Y or Q to close this window", (100, 400), font, 2.0, (255, 255, 255), thickness, lineType)
    
    cv2.imshow('Final count', blackscreen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    

if __name__ == "__main__":
    # rgbCount()
    startx = 0
    starty = 0
    # vid_file = "./labvid.mov"
    # vid_file = "./Shortvid.mov"
    vid_file = "./without_start_blur.mp4"  #it says theres <153> drops in this
    # vid_file = "./first_2_mins_76_drops.mp4" # code says there's <76> drops in this
    # vid_file = "./last_2_mins.mp4" # code says theres 77 drops in this
    details = makeBox(vid_file) ## ry to make the diag boz
    start, finish = userSelectsFrames(details[0], details[1], details[2], vid_file)
    rgbcount_no_watch(details[0], details[1], details[2], (start, finish), vid_file)
    
    
    
