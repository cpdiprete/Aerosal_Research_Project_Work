import numpy as np
import matplotlib.pyplot as plt
import opencvresearch
import plotly.graph_objs as go
import plotly.io as pio
import time

minute = 60

def makePlot(droplet_vals, droplet_formation_time, num_segments, fps):
    plt.figure(figsize=(80, 8))
    normalizing_window_size = droplet_formation_time // 3
    droplet_vals = np.convolve(droplet_vals, np.ones(normalizing_window_size)/normalizing_window_size, mode='valid')
    total_len = len(droplet_vals)
    frames = np.arange(total_len)
    aggregate_plotted_points = []
    aggregate_droplet_counts = []
    
    last_frames_start = 0
    colors = ['m', 'r', 'y', 'b', 'p']
    for i in range(1, num_segments + 1):
        current_color = colors[i % (len(colors) - 1)]
        slice_start = int((i - 1) * (minute*fps))
        slice_end = int((i * (minute * fps)))
        droplet_slice = droplet_vals[slice_start:slice_end]
        slice_median = np.median(droplet_slice)
        slice_lower_Q = np.quantile(droplet_slice, .25)
        
        formed_drop_x_points, formed_drop_y_points, disappear_x_points, disappear_y_points, _ = opencvresearch.countJumps(droplet_slice, slice_median, droplet_formation_time)
        formed_drop_x_points = np.array(formed_drop_x_points)
        formed_drop_x_points += slice_start
        formed_drop_x_points = list(formed_drop_x_points)
        plt.plot(formed_drop_x_points, formed_drop_y_points, 'o', color='r')
        plt.plot(disappear_x_points, disappear_y_points, 'o', color='black')
        num_droplets_formed_this_slice = len(formed_drop_x_points)
        
        
        # Duration of the slice in seconds
        slice_duration_seconds = (len(droplet_slice)) / fps

        # Droplets per minute = (droplets / seconds) * 60
        droplet_per_minute = round((num_droplets_formed_this_slice / slice_duration_seconds) * 60, 2)
        
        x_pos = slice_start + (slice_end - slice_start) // 2
        # Set y position to a little above the current slice max
        y_pos = np.min(droplet_slice) - .25  # tweak this value if needed
        plt.text(x_pos, y_pos, f"{droplet_per_minute} drops/min", fontsize=10, color=current_color, ha='center', va='bottom')
        
    # """
    #     print(f"Shape of frames: {np.shape(frames)}")
    #     print(f"Shape of total array of droplets: {total_len}")
    #     print(f"Shape of plotted subsection: {np.shape(droplet_slice)}")
    #     print(f"Shape of frames_axis: {np.shape(frames[slice_start:slice_end])}")
    #     np.ones(np.shape(np.median(droplet_slice))) * np.median(droplet_slice)
    #     slice_median = np.ones(np.shape(droplet_slice)) * np.median(droplet_slice)
    # """
        # for x_spot in slice_plotted_points:
        #     plt.axvline(x=x_spot, color=current_color, linestyle='--')

        slice_median = np.ones(np.shape(droplet_slice)) * np.median(droplet_slice)
        plt.plot(frames[slice_start:slice_end], droplet_slice, label='RGB', color = current_color)
        plt.plot(frames[slice_start:slice_end], slice_median, label="Midpoint", color= current_color)
        
    plt.ylim(top=np.max(droplet_vals) + 1)
    plt.ylim(bottom=np.min(droplet_vals) - 1)
    plt.ylabel('RGB val of bounding box')
    plt.xlabel('Video frame')
    plt.gca().invert_yaxis()
    
    plt.show()

def makeInteractivePlot(droplet_vals, droplet_formation_time, num_segments, fps):
    start_time = time.time()
    pio.renderers.default = "browser"
    
    normalizing_window_size = droplet_formation_time // 3
    # droplet_vals = np.convolve(droplet_vals, np.ones(normalizing_window_size)/normalizing_window_size, mode='valid') # 
    total_len = len(droplet_vals)
    frames = np.arange(total_len)
    
    fig = go.Figure()
    colors = ['magenta', 'red', 'indigo', 'forestgreen', 'blue', 'purple']  # HTML color names

    ## Calvin Idea, I can collect all of the 'count jumps' mins, and then loop over them and remove any that appear within less of (0.9 * droplet_formation_time)
    ## this would handle the issues with double plotting a droplet because of the index split when splitting into segments
    
    for i in range(1, num_segments + 1):
        current_color = colors[i % (len(colors) - 1)]
        slice_start = int((i - 1) * (minute * fps))
        slice_end = int(i * (minute * fps))
        droplet_slice = droplet_vals[slice_start:slice_end]
        slice_median = np.median(droplet_slice)
        slice_lower_Q = np.quantile(droplet_slice, .25)

        # x_points, y_points, _ = opencvresearch.countJumps(droplet_slice, slice_median, droplet_formation_time)
        formed_drop_x_points, formed_drop_y_points, disappear_x_points, disappear_y_points, _ = opencvresearch.countJumps(droplet_slice, slice_median, droplet_formation_time)
        formed_drop_x_points = np.array(formed_drop_x_points) + slice_start  # shift to global frame index
        disappear_x_points = np.array(disappear_x_points) + slice_start

        # Plot RGB line
        fig.add_trace(go.Scatter(
            x=frames[slice_start:slice_end],
            y=droplet_slice,
            mode='lines',
            name=f'Segment {i}',
            line=dict(color=current_color)
        ))

        # Plot midpoint
        midpoint_line = np.ones_like(droplet_slice) * slice_median
        fig.add_trace(go.Scatter(
            x=frames[slice_start:slice_end],
            y=midpoint_line,
            mode='lines',
            name=f'Midpoint {i}',
            line=dict(color=current_color, dash='dot'),
            showlegend=False
        ))
        # # Plot Q1
        # Q1_line = np.ones_like(droplet_slice) * slice_lower_Q
        # fig.add_trace(go.Scatter(
        #     x=frames[slice_start:slice_end],
        #     y=Q1_line,
        #     mode='lines',
        #     name=f'Q1 {i}',
        #     line=dict(color='black', dash='dot'),
        #     showlegend=False
        # ))

        # Plot detected droplet points
        fig.add_trace(go.Scatter(
            x=formed_drop_x_points,
            y=formed_drop_y_points,
            mode='markers',
            marker=dict(color='black', size=6),
            name=f'Droplets {i}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=disappear_x_points,
            y=disappear_y_points,
            mode='markers',
            marker=dict(color='red', size=6),
            name=f'Droplet {i}',
            showlegend=False
        ))
        # Add drops/min annotation
        num_droplets_formed = len(formed_drop_x_points)
        slice_duration_seconds = len(droplet_slice) / fps
        droplets_per_min = round((num_droplets_formed / slice_duration_seconds) * 60, 2)

        x_pos = slice_start + (slice_end - slice_start) // 2
        y_pos = np.min(droplet_slice) - 0.25
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=f"{droplets_per_min} drops/min",
            showarrow=False,
            font=dict(size=14, color=current_color),
            yanchor='bottom'
        )

    fig.update_layout(
        title='Droplet RGB Values Over Time',
        xaxis_title='Frame',
        yaxis_title='RGB',
        yaxis=dict(autorange='reversed'),
        height=800,
        width=2500,
        hovermode='x unified'
    )

    pio.show(fig)
    end_time = time.time()
    


def makegrayPlot(vals, start_finish):
    # pixels = np.column_stack([b_vals, g_vals, r_vals]) # (n, 3) matrix
    # vector_periodogram(pixels) 
    # total = b_vals + g_vals + r_vals
    total = vals
    mean = np.mean(total)
    median = np.median(total)
    lower_Q = np.quantile(total, .25)
    
    window_size = (start_finish[1] - start_finish[0]) // 3
    total = np.convolve(total, np.ones(window_size)/window_size, mode='valid')
    print("mean: ", mean)
    print("median", median)
    x_axis = opencvresearch.countJumps(total, median, start_finish)[0]
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