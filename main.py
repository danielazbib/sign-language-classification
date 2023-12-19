"""
This project was built using OpenCV to take in landmarks of hand images. 
These hand images were then used to train various classifer models. 
The accuracy score of determined which model was to be pickled and deployed to the main webpage.
The main webpage aims to aid the process of learning sign language. 

"""
import pickle
import cv2 as cv
import mediapipe as mp
import numpy as np
# from arduinoController import controller as cnt

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Constructing a dictionary with keys as numbers (0-9) and values as digits.

labels_dict = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'a',
    11: 'b',
    12: 'c',
    13: 'd',
    14: 'e',
    15: 'f',
    16: 'g',
    17: 'h',
    18: 'i',
    19: 'j',
    20: 'k',
    21: 'l',
    22: 'm',
    23: 'n',
    24: 'o',
    25: 'p',
    26: 'q',
    27: 'r',
    28: 's',
    29: 't',
    30: 'u',
    31: 'v',
    32: 'w',
    33: 'x',
    34: 'y',
    35: 'z'
}
#takes in the data from the active computer vision

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
            break

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        max_len = max(len(data_aux), 84)
        # print(data_aux)
        data_aux_padded = data_aux + [0] * (max_len - len(data_aux))
        # print(data_aux_padded)

        prediction = model.predict([np.asarray(data_aux_padded)])[0]
        # print(prediction)
        # predicted_character = labels_dict[prediction]

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0, 0), 4)
        cv.putText(frame, prediction, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, 'Press q to exit', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv.LINE_AA)
        cv.imshow('frame', frame)


        #ends loop
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

cap.release()
cv.destroyAllWindows()