import pickle
import cv2 as cv
import mediapipe as mp
import numpy as np

class SignLanguageInterpreter:
    def __init__(self, model_path='model.p'):
        """
        Initializes the SignLanguageInterpreter class.

        Params:
            model_path (str): path from pickled pre trained model
        """
        #load pre trained model
        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']

        #initialize video capture
        self.cap = cv.VideoCapture(0)

        #initalize hand landmark readings 
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        self.labels_dict = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
            20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
            30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'
        }

    def process_frame(self, frame):
        """
        Processes a video frame for hand landmarks and makes predictions.

        Parameters:
            frame (numpy array): input from cv video capture

        Returns:
            tuple: A tuple containing the processed frame with overlays and the predicted sign character.
        """
        #intialize empty lists
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        #process hand landmarks using mediapipe
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            #draw handmarks and connections on frame
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            #extract x, and y coordinates of handlandmarks
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
                break
            
            #calculate border box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            #pad data to fixed length for accurate model input
            max_len = max(len(data_aux), 84)
            data_aux_padded = data_aux + [0] * (max_len - len(data_aux))

            #make predictions using trained model
            prediction = self.model.predict([np.asarray(data_aux_padded)])[0]

            #draws bordering boix and predicted character on video frame
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0, 0), 4)
            cv.putText(frame, prediction, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0, 0), 3, cv.LINE_AA)
            cv.putText(frame, 'Press q to exit', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv.LINE_AA)

            return frame, prediction

        #returns a default value if no hand landmarks are found
        return frame, None

    def run(self):
        """
        Runs the main loop for capturing video frames and processing them.

        Para: None
        Returns: None
        """
        while True:
            ret, frame = self.cap.read()
            #process frame
            processed_frame, prediction = self.process_frame(frame)
            #display frame
            cv.imshow('frame', processed_frame)
            #check for exit key
            if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        #close out
        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    interpreter.run()
