import cv2
import pyautogui
import numpy as np
import time 
import pyautogui

import mediapipe as mp 

screenWidth, screenHeight = pyautogui.size()
currentMouseX, currentMouseY = pyautogui.position()
start_width,end_width = 70,130
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

#distance calculation
fl = 1500
object_width = 5.5
mp_face_mesh = mp.solutions.face_mesh
seen = 0
i = 0 
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:


    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if ret: 
            if seen%5!=0:
                seen += 1

                continue
            seen += 1
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = frame.copy()
            img_h, img_w = frame.shape[:2]
            #processed = face_mesh.process(frame)
            results = face_mesh.process(frame)
            
            #print(results)
            if results.multi_face_landmarks:
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                # for face_landmarks in results.multi_face_landmarks:
                #     mp_drawing.draw_landmarks(
                #     image=annotated_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                #     )
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                leyepp = mesh_points[476]
                reyepp = mesh_points[469]
                l2 = mesh_points[474]
                l3 = mesh_points[475]
                l4 = mesh_points[477]
                #looking at top left corner
                   



                left_eye_sp = mesh_points[362]
                rt_eye_sp =  mesh_points[133]
                #print(left_eye_sp,rt_eye_sp)
                sidewise_left = np.sqrt(((leyepp[0]-left_eye_sp[0])**2)+((leyepp[1]-left_eye_sp[1])**2))
                sidewise_right = np.sqrt(((reyepp[0]-rt_eye_sp[0])**2)+((reyepp[1]-rt_eye_sp[1])**2))
                #pupil_dist = np.sqrt(((center_left[0]-center_right[0])**2)+((center_left[1]-center_right[1])**2))
               # distfromscreen = (object_width*fl)//pupil_dist
                #print(distfromscreen)
                sidewise_left = sidewise_left*10
                if sidewise_left>50:
                    
                    movex = screenWidth//(sidewise_left-50)
                    chd = sidewise_left-50
                    movex = int(chd*movex)
                    movey = 100
                    
                
                    pyautogui.moveTo(movex,movey,duration=0.2, tween=pyautogui.easeInOutQuad)
                  

                #print(pupil_dist)
                print(sidewise_left)
                cv2.putText(annotated_image,str(sidewise_left), (50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),1,cv2.LINE_AA)
                
                if sidewise_left<8:
                    i += 1

                    print(f'looking at left corner {i}')
                if sidewise_left>15:
                    print(f'looking at right corner {i}')
                    i += 1
                #print(ldist)
                cv2.line(annotated_image,leyepp,left_eye_sp,(0,255,0),2)
                #cv2.line(annotated_image,reyepp,rt_eye_sp,(0,255,0),2)
                #cv2.line(annotated_image,center_left,center_right,(0,255,0),2)
                cv2.circle(annotated_image, left_eye_sp, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.circle(annotated_image, rt_eye_sp, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.circle(annotated_image, leyepp, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.circle(annotated_image, reyepp, 1, (0,255,0), 2, cv2.LINE_AA)
                #print(results.multi_face_landmarks)
            if cv2.waitKey(1)==ord('q'):
                break
            cv2.imshow('frmae',annotated_image)
            cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

