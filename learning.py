import torch
import cv2 
#import beepy as beep
import numpy as np 

def detect(image , model):
    results = model(image)
    points = results.xyxy[0]

    if points.size()[0] == 0 :
        return None
    
    data = results.pandas().xyxy[0].copy()
    #if you need to change the confidence, change 0.7 (70%) below:
    data = data[(data.confidence > 0.7 )]
    area = [ (row['xmax']-row['xmin'])*row['ymax']-row['ymin'] for index, row in data.iterrows() ]
    data["Area"] = area
    data.sort_values(by=['Area'], inplace=True, ascending=False)

    
    print(data)
    if len(data) == 0 :
        return None
    
    x1 = int(data.iloc[0]["xmin"])
    y1 = int(data.iloc[0]["ymin"])
    x2 = int(data.iloc[0]["xmax"])
    y2 = int(data.iloc[0]["ymax"])
    
    area = int(data.iloc[0]["Area"])
    name = data.iloc[0]["name"]
    if name not in ["car","person"]:
        name = "others"
    return (x1,y1,x2,y2,area,name)


def draw(image , obj_data):
    x1,y1,x2,y2,obj_area, obj_name = obj_data
    image_area = image.shape[0]*image.shape[1]
    
        
    if obj_area < 0.4*image_area:
        color = (0,255,0)
        text = f"{obj_name} is Far Away"
    else:
        #beep.beep(1)
        color = (0,0,255)
        text = f"Alarm!! {obj_name} is Close!!!!!!!"
    
    image = cv2.putText(image, text, (x1,int(0.8*y1)), cv2.FONT_HERSHEY_SIMPLEX,1,color, 2, cv2.LINE_AA)
    image = cv2.rectangle(image, (x1,y1), (x2,y2) , color, 5 )
    return image   

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5m, yolov5l, yolov5x, custom


def single_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image = frame.copy()
        obj_data = detect(image , model)
        if obj_data != None :  
            image = draw(image,obj_data)
        cv2.imshow("frame",image )
        k = cv2.waitKey(25)          
        if k == ord('q'):
            break
    cap.release()

def multi_camera():

    cap_left = cv2.VideoCapture(4)
    cap_right = cv2.VideoCapture(2)

    while True:
    
        ret , frame_left = cap_left.read()
        ret , frame_right = cap_right.read()
    
        image_left = frame_left.copy()
        image_right = frame_right.copy()
    
        points_left = detect(image_left , model)
        if points_left != None :  
            image_left = draw(image_left,points_left)
    
        points_right = detect(image_right , model)
        if points_right != None :  
            image_right = draw(image_right,points_right)
        vis = np.concatenate((image_left, image_right), axis=1)
        cv2.imshow("vis",vis )
    
 
        k = cv2.waitKey(25)          
        if k == ord('q'):
            break

    cap_left.release()
    cap_right.release()

single_camera()
cv2.destroyAllWindows()