import cv2 as cv
import numpy as np
from scipy.signal import find_peaks, peak_widths, argrelextrema
import numpy.linalg as LA
import statistics
from tasks.segmentation.peak_removal_adaptive_thresholding import init_filter_out_highest_peak, \
                                                                filter_out_highest_peak_multidim
from tasks.segmentation.aggregateRescaling import init_aggregate_rescaling 


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

    """ Finds and returns the x-range of the highest peak in the
        given channel of frame """

    f = frame[:, :, channel].flatten()

    # Some semi-hardcoded values :)
    num_bins = max(int((np.amax(f)-np.amin(f)) / 4), 10)
    # num_bins = 30

    hist, bins = np.histogram(f, bins=num_bins)

    hist[0] = 0 # get rid of stuff that was thresholded to 0
    hist = np.hstack([hist, [0]]) # make stuff at 255 into a peak
    bins = np.hstack([bins, [bins[bins.shape[0]-1] + 1]])

    peaks, properties = find_peaks(hist, height=0.1)
    if len(peaks) > 0:
        i = np.argmax(properties['peak_heights'])
        widths = peak_widths(hist, peaks, rel_height=peak_width_height)[0]
        # i = np.argmax(widths)
        largest_peak = (int((bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i]//2), 
                                int((bins[peaks[i]]+bins[peaks[i]+1])//2+widths[i]//2)) # beginning and end of the peak
    else:
        argest_peak = (0, 0)

    return largest_peak

    only_once = False
    weights = []
    max_min = {'max': 90, 'min': -20}

    def aggregate_rescaling(frame):  # you only pca once
        nonlocal only_once
        nonlocal weights
        nonlocal max_min

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        r, c, d = frame.shape
        A = np.reshape(frame, (r * c, d))

        if not only_once:

            A_dot = A - A.mean(axis=0)[np.newaxis, :]

            _, eigv = LA.eigh(A_dot.T @ A_dot)
            weights = eigv[:, 0]

            red = np.reshape(A_dot @ weights, (r, c))
            only_once = True
        else:
            red = np.reshape(A @ weights, (r, c))

        if np.min(red) < max_min['min']:
            max_min['min'] = np.min(red)
        if np.max(red) > max_min['max']:
            max_min['max'] = np.max(red)

        red -= max_min['min']
        red *= 255.0 / (max_min['max'] - max_min['min'])
        """
		if False:#not paused:
			print(np.min(red), np.max(red), max_min['min'], max_min['max'])
		"""
        red = red.astype(np.uint8)
        red = np.expand_dims(red, axis=2)
        red = np.concatenate((red, red, red), axis=2)

        if show_frame:
            # cv.imshow('frame', frame_gray)
            cv.imshow('One Time PCA plus all time aggregate rescaling', red)
        return red

    return aggregate_rescaling

    """ Takes in an hsv image! Returns an hsv image"""
    # low pass filter
    # vk* = vk*lambda + v*(1-lambda)
    # lambda = 0.9-0.4

    prev_hsv_threshes = [[] for i in range(len(filters))]
    hsv_labels = (('H','S','V'), ("red","purple","gray"))
    bgr_labels = (('B','G','R'), ("blue","green","red"))

    # Figure out how the procedure to convert among hsv and bgr.
    # Format of stuff in fitler_fns:
    # [<'c' convert or 'f' filter>, <target colorspace>]
    filter_fns = []
    curr_color = input_colorspace
    for f in filters:
        if f != curr_color:
            filter_fns.append(['c',f])
            curr_color = f
        filter_fns.append(['f',f])
    if return_colorspace != "any" and return_colorspace != curr_color:
        filter_fns.append(['c', return_colorspace])

    def filter_out_highest_peak(frame, cache, display_plots=False, title=None, labels=None, colors=None):

        background_thresh = find_peak_ranges(frame, display_plots, title, labels, colors)
        raw_thresh = background_thresh
        # multiply everything in cache by (1-lpf_lambda)
        if len(cache) > 0:
            # cache = np.array(cache) * (1-lpf_lambda)
            # calculate average
            for i in range(2):
                for j in range(3):
                    background_thresh[i][j] = (background_thresh[i][j] + sum([c[i][j] for c in cache])) // (len(cache) + 1)

        background_mask = cv2.bitwise_not(cv2.bitwise_or(
                            cv2.inRange(frame[:, :, 0], background_thresh[0][0], background_thresh[1][0]),
                            cv2.inRange(frame[:, :, 1], background_thresh[0][1], background_thresh[1][1]),
                            cv2.inRange(frame[:, :, 2], background_thresh[0][2], background_thresh[1][2])
                        ))
        no_background = cv2.bitwise_and(frame,frame, mask=background_mask)

        return background_thresh, raw_thresh, no_background

    def combine_threshes(th1, th2):
        return ([min(th1[0][0], th2[0][0]), min(th1[0][1], th2[0][1]), min(th1[0][2], th2[0][2])], 
                    [max(th1[1][0], th2[1][0]), max(th1[1][1], th2[1][1]), max(th1[1][2], th2[1][2])])

    def bgr_thresh2hsv_thresh(th):
        th = cv2.cvtColor(np.array([[th[0]], [th[1]]], np.uint8), cv2.COLOR_BGR2HSV).tolist()
        return ([min(th[0][0][0], th[1][0][0]), min(th[0][0][1], th[1][0][1]), min(th[0][0][2], th[1][0][2])],
                [max(th[0][0][0], th[1][0][0]), max(th[0][0][1], th[1][0][1]), max(th[0][0][2], th[1][0][2])])


    def do_filter(frame, display_plots=False):
        nonlocal prev_hsv_threshes
        if len(prev_hsv_threshes[0]) == lpf_cache_size:
            for x in prev_hsv_threshes:
                x.pop(0)

        filter_index = 0
        for f in filter_fns:
            if f[0] == 'c':
                if f[1] == "hsv":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                else:
                    # f[1] == "bgr"
                    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            else:
                if f[1] == "hsv":
                    thresh, raw_thresh, frame = filter_out_highest_peak(frame, prev_hsv_threshes[filter_index])
                    prev_hsv_threshes[filter_index].append(raw_thresh)
                else:
                    # f[1] == "bgr"
                    thresh, raw_thresh, frame = filter_out_highest_peak(frame, prev_hsv_threshes[filter_index])
                    thresh = bgr_thresh2hsv_thresh(thresh)
                    prev_hsv_threshes[filter_index].append(raw_thresh)
                filter_index += 1

        # Doesn't do anything :c
        # frame = cv2.fastNlMeansDenoising(frame)

        # # # Post processing
        # # Performs badly if there is a lot of noise or if there is no noise at all around targets
        # frame = remove_blotchy_chunks(frame, iterations=1, display_imgs=True)
        # cv2.imshow('after antiblotchy', frame)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('after open', frame)

        return frame

    return do_filter
    
    """ Estimates the "peak-ness" of each pixel in frame across color channels
        and thresholds out pixels that were "peak-like" in many colorspaces.
        frame is a stack of color channels (np.dstack()) and this will consider all channels
        in the final calculation
        
        @param res Resolution. Higher number is lower resolution
        @param percentile Threshold for pixels to keep in the overall_votes array
        List of colorspaces that can be converted to from BGR:
        https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a2a80354788f54d0bed59fa44ea2fb98e
        - HSV, GRAY, Lab, XYZ, YCrCb, Luv, HLS, YUV
        "Theoretically", the more colorspaces you consider, the better? But noise is added """

    from math import log
    def get_peak_votes(frame):
        """ Takes in a single-channel frame and returns an array that contains
            the number of other pixels with the same value at every pixel """
        dist = np.bincount(frame.flatten())
        recommended_weight = abs((np.max(np.nonzero(dist)) - np.min(np.nonzero(dist))))

        if res == 1:
            vote_arr = dist[frame]
        else:
            dist = np.array([np.mean(dist[i*res:i*res+res]) for i in range(len(dist) // res + 1)])
            vote_arr = dist[frame // res]

        return recommended_weight, vote_arr

    overall_votes = np.zeros(frame.shape[:2], np.uint8)
    overall_mask = np.zeros(frame.shape[:2], np.uint8)

    if print_weights:
        print('------------------------', custom_weights)
    for ch in range(frame.shape[2]):
        weight, vote_arr = get_peak_votes(frame[:,:,ch])
        if custom_weights is not None:
            weight = custom_weights[ch]
        if print_weights:
            print(weight)
        overall_votes = overall_votes + vote_arr * weight

    # Sometimes returns no pixels
    thresh = np.mean(overall_votes) - 2 * np.std(overall_votes)

    # thresh = np.percentile(overall_votes, percentile)

    # Only keep pixels that were very un-peak-like in every colorspace
    overall_mask[overall_votes <= thresh] = 255

    return overall_votes, cv.bitwise_and(frame, frame, mask=overall_mask)

def analyse(frame, base=25, tol=10, sort_by_area=True):
    max_brightness = max([b for b in frame[:, :, 0][0]])
    lowerbound = max(0.84*max_brightness, 120)
    upperbound = 255
    _,thresh = cv.threshold(frame,lowerbound, upperbound, cv.THRESH_BINARY)
    gray_filter = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    cnt = cv.findContours(gray_filter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]

    hull_list = []
    for i in range(len(cnt)):
        hull = cv.convexHull(cnt[i])
        hull_list.append(hull)
    area_diff = []
    area_rects = []
    area_cnts = []

    # remove all contours with zero area
    cnt = [cnt[i] for i in range(len(cnt)) if cv.contourArea(cnt[i]) > 0]

    for i in range(len(cnt)):
        area_cnt = cv.contourArea(cnt[i])
        area_cnts.append(area_cnt)
        area_rect = cv.boundingRect(cnt[i])[-2] * cv.boundingRect(cnt[i])[-1]
        area_rects.append(area_rect)
        area_hull = cv.contourArea(hull_list[i])
        # area_diff.append(((area_rect - area_cnt)/area_cnt) * 10 - area_cnt * 0.1 + (area_hull - area_cnt))
        area_diff.append(abs((area_rect - area_cnt)/area_cnt))

    if len(area_diff) >= 2:
        if sort_by_area:
            largest_area_idx = [area_cnts.index(sorted(area_cnts, reverse=True)[i]) for i in range(min(3, len(cnt)))]
            area_diff_copy = sorted([area_diff[i] for i in largest_area_idx])
            min_i1, min_i2 = area_diff.index(area_diff_copy[0]), area_diff.index(area_diff_copy[1])
        else:
            min_i1, min_i2 = area_diff.index(sorted(area_diff)[0]), area_diff.index(sorted(area_diff)[1])
        (x1, y1, w1, h1) = cv.boundingRect(cnt[min_i1])
        (x2, y2, w2, h2) = cv.boundingRect(cnt[min_i2])
        cv.rectangle(thresh, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)
        cv.rectangle(thresh, (x2, y2), (x2+w2, y2+h2), (0,255,0), 2)

        # begin drawing center dot
        center_x, center_y = (x1+x2)//2, ((y1+h1//2)+(y2+h2//2))//2

        # get starting center location, averaging over the first tol frames
        if len(center_x_locs) == 0:
            center_x_locs.append(center_x)
            center_y_locs.append(center_y)

        elif len(center_x_locs) < base:
            center_x_locs.append(center_x)
            center_y_locs.append(center_y)
            center_x = int(statistics.mean(center_x_locs))
            center_y = int(statistics.mean(center_y_locs))
        
        # use new center location only when it is close to the previous valid location
        else:
            if abs(center_x - center_x_locs[-1]) > tol or \
                abs(center_y - center_y_locs[-1]) > tol:
                center_x, center_y = center_x_locs[-1], center_y_locs[-1]
            else:
                center_x_locs.append(center_x)
                center_y_locs.append(center_y)

        cv.circle(thresh, (center_x, center_y), 5, (0,0,255), -1)

    return thresh

cap = cv.VideoCapture('../GOPR1142.MP4')

filter_peaks = init_filter_out_highest_peak(['hsv,', 'bgr', 'hsv'], 'hsv')
agg_res = init_aggregate_rescaling()

paused = False
speed = 1

center_x_locs, center_y_locs = [], []

while(cap.isOpened()):
    if not paused:
        for _ in range(speed):
            ret, frame = cap.read()
    if ret:
        frame = rescale_frame(frame, 20)

        red = agg_res(frame)
        votes, multi_filter2 = filter_out_highest_peak_multidim(np.dstack([red, 
        cv.cvtColor(red, cv.COLOR_BGR2HSV)]))
        multi_filter2 = multi_filter2[:, :, :3]

        masked_filter = analyse(multi_filter2, sort_by_area=True)

        cv.imshow('frame', frame)
        cv.imshow('multi_filter2', multi_filter2)
        cv.imshow('masked_filter', masked_filter)
        
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
    if key == ord('p'):
        paused = not paused
    if key == ord('i') and speed > 1:
        speed -= 1
    if key == ord('o'):
        speed += 1

cap.release()
cv.destroyAllWindows()