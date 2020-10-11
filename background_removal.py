#%matplotlib inline
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
#import imageio



import numpy as np
import cv2

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def find_peak_ranges(frame, display_plots=False, title=None, labels=None, colors=None):
    """ Finds a returns the widest peak's x-range in all three channels of frame
        Result is formatted to fit cv2.inRange() -> ((low1, low2, low3), (hi1, hi2, hi3))
        Shape of frame must have 3 dimensions (pass in np.expand_dims(frame, 2) if erroring) """

    # TODO: Maybe use a different combination of peak characteristics to more accurately
    #       select the entire peak (only the tip is selected right now)
    peak_width_height = 0.95 # How far down the peak that the algorithm draws
                            # the horizontal width line

    def find_highest_peak(channel, display_plots=False):
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

            if display_plots:
                ax = plt.gca()
                print(max(f))
                ax.set_xlim([0, max(255, max(f))])
                #Plot values in this channel
                plt.plot(bins[1:],hist, label=labels[channel], color=colors[channel])
                # Plot peaks
                plt.plot(bins[peaks+1], hist[peaks], "x")
                # Plot peak widths
                plt.hlines(hist[peaks]*0.9, bins[peaks+1]-widths//2, bins[peaks+1]+widths//2)
        else:
            largest_peak = (0, 0)

        return largest_peak

    if display_plots:
        fig = plt.figure(hash(title))
        plt.clf()

    background = (np.empty(frame.shape[2]),np.empty(frame.shape[2]))
    for channel in range(frame.shape[2]):
        low, high = find_highest_peak(channel, display_plots)
        background[0][channel] = low
        background[1][channel] = high

    if display_plots:
        plt.title(title)
        plt.legend()
        plt.draw()

    return background

def filter_out_highest_peak_multidim(frame, res=69, percentile=10, custom_weights=None, print_weights=False):
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

        # Set the recommended weight to (the spread of the pixel values from 0-255)
        # recommended_weight = abs(np.subtract(*np.percentile(np.nonzero(dist), [75, 25])) - (255//2))
        recommended_weight = abs((np.max(np.nonzero(dist)) - np.min(np.nonzero(dist))))
        # Stretch out extremes
        # recommended_weight = pow(2, recommended_weight//8)

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

    return overall_votes, cv2.bitwise_and(frame, frame, mask=overall_mask)


def init_aggregate_rescaling(show_frame=True):
    only_once = False
    weights = []
    max_min = {'max': 90, 'min': -20}

    def aggregate_rescaling(frame):  # you only pca once
        nonlocal only_once
        nonlocal weights
        nonlocal max_min

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            red = rescale_frame(red, percent=40)
            cv2.imshow('One Time PCA plus all time aggregate rescaling', red)
        return red

    return aggregate_rescaling


cap = cv2.VideoCapture('GOPR1142.MP4')
agg_res = init_aggregate_rescaling()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
    
        pca = agg_res(frame)
        votes, multi_filter2 = filter_out_highest_peak_multidim(np.dstack([pca, cv2.cvtColor(pca, cv2.COLOR_BGR2HSV)]))
        multi_filter2 = multi_filter2[:, :, :3]

            # kmeans_groups = k_means_segmentation(votes, frame.shape)
        frame = rescale_frame(frame, percent=40)
        cv2.imshow('original', frame)



        # gray = cv2.cvtColor(multi_filter1, cv2.COLOR_BGR2GRAY)
        # gray = rescale_frame(gray, percent=40)

        # dst = cv2.equalizeHist(gray)
      
        cv2.imshow('bgr', multi_filter2)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
