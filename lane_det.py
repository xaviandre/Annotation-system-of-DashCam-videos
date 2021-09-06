import cv2 as cv
import numpy as np

width = 768
height = 432
horizon = 241


def do_canny(original_frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally
    # expensive
    gray = cv.cvtColor(original_frame, cv.COLOR_RGB2GRAY)

    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # Applies Canny edge detector with minVal of 31 and maxVal of 40
    return cv.Canny(blur, 31, 40)


def do_segment(canny_frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image,
    # we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels]
    # of the dimensions of the frame

    # Creates a rectangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([[(201, height), (296, horizon), (472, horizon), (567, height)]])

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(canny_frame)

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)

    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    return cv.bitwise_and(canny_frame, mask)


def calculate_lines(lines, previous_left_parameters, previous_right_parameters):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []

    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)

        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe
        # the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)

        m = parameters[0]
        b = parameters[1]

        if m < -.3 or m > .3:
            # Calculate the x coordinate when the line intersects the 3/4 of the height
            x_top = int((((3 / 4) * height) - b) / m)
            x_height = int((height - b) / m)

            if int((1 / 5) * width) < x_height < int((4 / 5) * width) and int((1 / 3) * width) < x_top < int((2 / 3) * width):
                # Identify lines side by splitting the image in half
                if x_top < width // 2 and x_height < width // 2:
                    left.append(parameters)
                elif x_top >= width // 2 and x_height >= width // 2:
                    right.append(parameters)

    # Selects the 3 lines with the center closer to x=width/2
    left_lines = sorted(left, key=lambda y: int((((horizon + height) / 2) - y[1]) / y[0]), reverse=True)[:3]
    right_lines = sorted(right, key=lambda y: int((((horizon + height) / 2) - y[1]) / y[0]), reverse=False)[:3]

    # If it isn't the first frame we select the lines with the least slope variation
    if previous_left_parameters is not None:
        left_lines = sorted(left_lines, key=lambda y: abs(y[0] - previous_left_parameters[0]), reverse=False)
        right_lines = sorted(right_lines, key=lambda y: abs(y[0] - previous_right_parameters[0]), reverse=False)

    influence = .01

    # Calculates the x1, y1, x2, y2 coordinates for the left line
    if len(left_lines) != 0:
        if previous_left_parameters is not None:
            influenced_left_line = np.array(
                [influence * left_lines[0][0] + (1. - influence) * previous_left_parameters[0],
                 influence * left_lines[0][1] + (1. - influence) * previous_left_parameters[1]])
        else:
            influenced_left_line = left_lines[0]
        left_line = calculate_coordinates(influenced_left_line)
        left_parameters = influenced_left_line
    else:
        if previous_right_parameters is None:
            return [], [], []
        left_line = calculate_coordinates(previous_left_parameters)
        left_parameters = previous_left_parameters

    # Calculates the x1, y1, x2, y2 coordinates for the right line
    if len(right_lines) != 0:
        if previous_right_parameters is not None:
            influenced_right_line = np.array(
                [influence * right_lines[0][0] + (1 - influence) * previous_right_parameters[0],
                 influence * right_lines[0][1] + (1 - influence) * previous_right_parameters[1]])
        else:
            influenced_right_line = right_lines[0]
        right_line = calculate_coordinates(influenced_right_line)
        right_parameters = influenced_right_line
    else:
        if previous_right_parameters is None:
            return [], [], []
        right_line = calculate_coordinates(previous_right_parameters)
        right_parameters = previous_right_parameters

    # Save the lane obtained by the actual frame
    lane = np.array([left_line, right_line])

    return lane, left_parameters, right_parameters


def calculate_coordinates(parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = height
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def visualize_lines(lines_frame, lines):
    if len(lines) > 1:
        # Creates an image filled with zero intensities with the same dimensions as the frame
        lines_visual = np.zeros_like(lines_frame)
        overlay = np.zeros_like(lines_frame)

        contours = np.array([[lines[0][0], lines[0][1]], [lines[0][2], lines[0][3]], [lines[1][2], lines[1][3]], [lines[1][0], lines[1][1]]])
        cv.fillPoly(overlay, pts=[contours], color=(17, 226, 49))
        cv.addWeighted(overlay, 0.49, lines_visual, 0.51, 0, lines_visual)

        # Checks if any lines are detected
        if lines is not None:
            right = False
            for x1, y1, x2, y2 in lines:
                # Draws lines between two coordinates with green color and 4 thickness
                if right is True:
                    cv.line(lines_visual, (x1, y1), (x2, y2), (0, 0, 102), 4)
                else:
                    cv.line(lines_visual, (x1, y1), (x2, y2), (0, 102, 0), 4)
                    right = True
        return lines_visual
    else:
        return lines_frame


def process_frame(frame, left_parameters, right_parameters, video_lanes, draw_lines=True):  # bb_vertices
    lanes = video_lanes.copy()
    canny = do_canny(frame)
    # cv.imshow("Canny", canny)

    # Applies area of interest
    segment = do_segment(canny)
    # cv.imshow("Segment", segment)

    # Applies hough
    hough = cv.HoughLinesP(segment, 1, np.pi / 180, 25, np.array([]), minLineLength=10, maxLineGap=50)
    # Selects multiple detected lines from hough into one line for left border of lane and one line for right border of lane
    lines_calculated, left_params, right_params = calculate_lines(hough, left_parameters, right_parameters)

    # Visualizes the lines
    if draw_lines is True:
        if len(lines_calculated) == 0 and len(left_params) == 0 and len(right_params) == 0:
            return frame, left_parameters, right_parameters, video_lanes
        lanes.append(lines_calculated)
        lines_visualize = visualize_lines(frame, lines_calculated)

        # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
        result = cv.addWeighted(frame, .9, lines_visualize, 1, 1)
        cv.imshow("result", result)

        # Opens a new window and displays the output frame
        return result, left_params, right_params, lanes

    if len(lines_calculated) == 0 and len(left_params) == 0 and len(right_params) == 0:
        return left_parameters, right_parameters, video_lanes

    lanes.append(lines_calculated)

    return left_params, right_params, lanes


cap = cv.VideoCapture("data/videos/test2.mp4")
left_param = None
right_param = None
video_lane = list()

while cap.isOpened():
    ret, f = cap.read()

    if ret is True:
        output, left_param, right_param, video_lane = process_frame(f, left_param, right_param, video_lane)
        if len(video_lane) > 0:
            visualize_lines(output, video_lane[-1])
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
