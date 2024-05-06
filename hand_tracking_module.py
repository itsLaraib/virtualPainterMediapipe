import cv2 as cv
import mediapipe as mp
import time


class handDetector:
    def __init__(
        self,
        mode=False,
        maxHands=2,
        complexity=1,
        detectionConfidence=0.5,
        trackingConfidence=0.5,
    ):
        self.static_image_mode = mode
        self.max_num_hands = maxHands
        self.model_complexity = complexity
        self.min_detection_confidence = detectionConfidence
        self.min_tracking_confidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.model_complexity,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def drawHands(self, img, color1=(0, 0, 255), color2=(0, 255, 0), draw=True):
        self.points_color = self.mpDraw.DrawingSpec(color1)
        self.connections_color = self.mpDraw.DrawingSpec(color2)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                    img,
                    handLmks,
                    self.mpHands.HAND_CONNECTIONS,
                    self.points_color,
                    self.connections_color,
                )

        return img

    def findPoints(self, img, draw=True):
        list = []
        if self.results.multi_hand_landmarks:
            for handLmks in self.results.multi_hand_landmarks:
                for lm in handLmks.landmark:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    list.append((x, y))
                    if draw == True:
                        cv.circle(img, (x, y), 4, (255, 0, 255), 2)

        return list


def main():
    cTime = 0
    pTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        detector.drawHands(img)
        detector.findPoints(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(
            img, str(int(fps)), (10, 100), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5
        )
        cv.imshow("Capture", img)
        if cv.waitKey(20) & 0xFF == ord("d"):
            break


cv.destroyAllWindows()

if __name__ == "__main__":
    main()
