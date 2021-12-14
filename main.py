import cv2
import numpy as np
import time


class MotionDetection():
    def __init__(self,):
        self.BaseLine_Frame = None
    
    def proccess_image(self , frame  ):
        self.frame = frame
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        
        if self.BaseLine_Frame is None:
            self.BaseLine_Frame = gray
            return False
        
        frameDelta = cv2.absdiff(self.BaseLine_Frame, gray)
        self.BaseLine_Frame = gray
        thresh = cv2.threshold(frameDelta, 5, 20, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_,cnts,_) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        if cnts == [] :
            return False
        

        cnts = max(cnts, key = cv2.contourArea)
            
        if cv2.contourArea(cnts) < 1000:
            return False

        (x, y, w, h) = cv2.boundingRect(cnts)
        area = str(100*cv2.contourArea(cnts)//(640*480)) + "%"
        if cv2.contourArea(cnts) < 0.1 * 640*480 :
            self.frame = cv2.putText(self.frame , area, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
            cv2.rectangle(self.frame , (x, y), (x + w, y + h), (0, 255, 0), 2)
        else : 
            area = "ALARM!!!!!!!"
            self.frame = cv2.putText(self.frame , area, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,)
            cv2.rectangle(self.frame , (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        
        
        

        

def main():

    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(4)
    image_left = MotionDetection()
    image_right = MotionDetection()
   
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if ret_left and ret_right  == True:

            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            # scale_percent = 60 # percent of original size
            # width = int(frame.shape[1] * scale_percent / 100)
            # height = int(frame.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # # resize image
            # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            image_left.proccess_image(frame_left )
            image_right.proccess_image(frame_right )
    
            
            #cv2.imshow("Thresh", thresh)
            #cv2.imshow("Frame Delta", frameDelta)
            vis = np.concatenate((image_left.frame, image_right.frame), axis=1)
            cv2.imshow("vis",vis )
            #cv2.imshow("frameleft",image_left.frame )
            #cv2.imshow("frameright",image_right.frame )
            #cv2.imshow("BaseLine_Frame",BaseLine_Frame)

            k = cv2.waitKey(25)    
            
            if k == ord('q'):
                break
       
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

main()