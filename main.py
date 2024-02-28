import argparse
import tensorflow as tf
from flask import Flask, render_template,request
import time
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size,check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh,set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



import cv2
import time
import numpy as np
import mediapipe as mp  
from tensorflow.keras.models import load_model

import re



from collections import Counter

from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play


from functions import mediapipe_detection,draw_styled_landmarks,extract_keypoints




app=Flask(__name__ ,template_folder='templates')
app.app_context().push()





mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities









actions = np.array(['','hello','nice to','meet you','i love you','see you later','thank you'])




mediapipe_model = load_model('/home/sneha/DL/web_app/action.h5')




def generate_audio(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        audio = AudioSegment.from_file(audio_stream)
        play(audio)
    except:
        pass



# words=[]

def common_label(lst):
    # start_time = time.time()

    counts = Counter(lst)
    label = counts.most_common(1)[0][0]
    generate_audio(label)
   



def detect_mediapipe():

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    global face_encodings
    global face_names

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,enable_segmentation=True,smooth_segmentation=True) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            # frame=cv2.flip(frame,1)
            # Make detections

            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                
                res = mediapipe_model.predict(np.expand_dims(sequence, axis=0))[-1]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                

                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                generate_audio(actions[np.argmax(res)])
                                
                        else:
                            sentence.append(actions[np.argmax(res)])
                            generate_audio(actions[np.argmax(res)])
                            



                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV_Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()





def detect_yolo(save_img=False):
    label_list=[]
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    webcam = source.isnumeric()

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)



    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        modified_label = re.sub(r'[\d_]+', '', label)
                        label_list.append(modified_label)
                        
                        if len(label_list)>2:
                            common_label(label_list)
                            label_list.clear()
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()  
                    break

        if key == ord('q'):
            cv2.destroyAllWindows()  
            break



    print(f'Done. ({time.time() - t0:.3f}s)')






@app.route("/")
def hello_world():


    return render_template('home.html')




@app.route("/yolo_detect")
def yolo_detect():

    detect_yolo()  

    return render_template('home.html')







@app.route("/mediapipe_detect")
def mediapipe_detect():
    detect_mediapipe()

    return render_template('home.html')




@app.route("/asl")
def asl_conversion():
   
    return render_template('asl.html')





@app.route("/avatar",methods=['GET','POST'])
def avatar_play():
    signs=['hello','welcome','world','sign','language','nice to meet you','sign language','see you later','thank you']
    if request.method=='POST':

        sign=request.form.get('sign')

        if sign.lower() not in signs:
            return render_template('avatar.html',sign='not found')

   
    return render_template('avatar.html',sign=sign)




 


if __name__ == '__main__':

     

    parser = argparse.ArgumentParser()
     
    parser.add_argument('--weights', nargs='+', type=str, default='/home/sneha/DL/web_app/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)
    app.run(debug=True)


    
