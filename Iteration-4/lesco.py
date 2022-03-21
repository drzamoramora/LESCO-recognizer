##############################################################
#
#  LESCO LIBRARY
#  created by Dr. Juan Zamora-Mora
#  licensed: Attribution 4.0 International (CC BY 4.0)
#  https://creativecommons.org/licenses/by/4.0/
#
##############################################################

import numpy as np
import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
mp_hands = mp.solutions.hands

# flat a list of lists into a single list [[1,2],[3,4]] => [1,2,3,4]
def flatten(t):
    return [item for sublist in t for item in sublist]


def has_hands(image):
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        return results.multi_hand_landmarks != None
        
    

# returns a 2x21 = 42 lenght array [lefthand-righthand] array.
# xy positions are combined into a single aray with PCA.
# then left hand array is appended to the right hand array for a 
# 42 item array of a sign.
def get_hands(image, dim_reduction_fn):
    final_hand = []
    right = [0] * 21
    left = [0] * 21
    labels = [0,1]
    dra = dim_reduction_fn #dimensional reduction algorithm
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand in results.multi_handedness:
                handType=hand.classification # right or left          
                hand_label = handType[0].label
                hand_index = handType[0].index
                labels[hand_index] = hand_label
                
            for index, hands in enumerate(results.multi_hand_landmarks):
                current_class = labels[index]
                xy = []
                
                for lmk in hands.landmark:
                    xy.append([lmk.x, lmk.y])
                    
                #dra.fit(xy)
                x = dra.fit_transform(xy)
                x_flat = flatten(x)
                
                if current_class == 'Right':
                    right = x_flat
                else:
                    left = x_flat
        else:
            # this means there are no hands recognized in the frame...
            return None

        return np.array(left + right)

# this method transform a list of list of all frames hands into a
# 1-dimensional array of size by using PCA
def get_sign(sign, dim_reduction_fn):
    dra = dim_reduction_fn
    #dra.fit(sign)
    sign = dra.fit_transform(sign)
    return flatten(sign)

# rotates an image 5 degrees (example, -5 or 5)
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

# reads a sign-language video and returns its time-series representation
def process_video(file_url, dra, flip = False, rotate = False, rotate_degree = 5, show_video = False):
        
        sign = []
        final_sign = None
        
        try:
            i = 0
            cap = cv.VideoCapture(file_url)
            
            while cap.isOpened():
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    break
                    
                frame = cv.resize(frame, (480,300), interpolation = cv.INTER_AREA)            

                if flip:
                    frame = cv.flip(frame, 1)

                if rotate:
                    frame = rotate_image(frame, rotate_degree)

                if (i < 50):
                    hand_array = get_hands(frame, dra)
                    if not hand_array is None:
                        sign.append(hand_array)
                    if (show_video):
                        cv.imshow('frame', frame)
                i = i + 1

                if cv.waitKey(1) == ord('q'):
                    break

            cap.release()
            cv.destroyAllWindows()
            print("frames processed", i)

            final_sign = get_sign(sign, dra)
        except Exception as e:
            print("error processing video", file_url, str(e))
        
        
        return final_sign
    
# similarity between images using Mean Squared Error
def mse_images(a,b):
    s = mean_squared_error(a,b)
    return s

# separates a video into n-chunks. Each chunk should be a lesco sign.
def video_segmentation(video_url, window_size, seg_threshold):
    
    # Load video and frames with hands.
    video = []
    count = 0
    cap = cv.VideoCapture(video_url)
    while cap.isOpened():

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        
        # reduce image size for speed
        frame = cv.resize(frame, (480,300), interpolation = cv.INTER_AREA)      

        # if there are hands in the frame, then add this to the tensor.
        if (has_hands(frame)):
            video.append(frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
    # create video cuts based on the mse
    cuts = []
    buffer_size = 4
    for i in range(0, len(video[:-buffer_size])):
        diff_1 = mse_images(video[i].flatten(), video[i+1].flatten())
        diff_2 = mse_images(video[i+1].flatten(), video[i+2].flatten())
        diff_3 = mse_images(video[i+2].flatten(), video[i+3].flatten())
        variation = abs(diff_1 - diff_2 - diff_3)
        if variation  <= seg_threshold:
            print("diff frames", i, i+1, i+2, diff_1, diff_2, variation)
            cuts.append(i)
    
    # create a tensor just of the important moments.
    video_tensor = []
    for cut in cuts:
        video_tensor.append(video[cut:cut+window_size])
    
    return video_tensor

def process_image(frame, flip = False, rotate = False, rotate_degree = 5):
    frame = cv.resize(frame, (480,300), interpolation = cv.INTER_AREA)  
    if flip:
        frame = cv.flip(frame, 1)
    if rotate:
        frame = rotate_image(frame, rotate_degree)
    svd = TruncatedSVD(n_components=1) 
    hands_array = get_hands(frame, svd)
    return hands_array