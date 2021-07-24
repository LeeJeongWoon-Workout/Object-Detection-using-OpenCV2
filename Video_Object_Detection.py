!wget -O ./data/Jonh_Wick_small.mp4 https://github.com/chulminkw/DLCV/blob/master/data/video/John_Wick_small.mp4?raw=true
  
video_input_path = '/content/data/Jonh_Wick_small.mp4'

cap = cv2.VideoCapture(video_input_path)
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt)

#VideoCapture를 이용하여 Video를 frame별로 capture 할 수 있도록 설정
#VideoCapture의 속성을 이용하여 Video Frame의 크기 및 FPS 설정.
#VideoWriter를 위한 인코딩 코덱 설정 및 영상 write를 위한 설정


video_input_path = '/content/data/Jonh_Wick_small.mp4'
video_output_path = './data/John_Wick_small_cv01.mp4' #Obejct_Detection 이 끝난 영상을 저장할 위치

cap = cv2.VideoCapture(video_input_path) #cv2 내장함수 VideoCapture가 프레임 별로 사진을 캡쳐한다.

codec = cv2.VideoWriter_fourcc(*'XVID') #XVID 형식으로 작성

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) #크기 설정 크기는 원본 영상 파일 유지 
vid_fps = cap.get(cv2.CAP_PROP_FPS ) #프레임 설정 원본 영상 프레임 유지
    
vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size) #저장위치,작성할형식,프레임,크기 저장 Object_Detection이 끝난 영상을 어떻게 배포할 것인지 결정한다.

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt)

# bounding box의 테두리와 caption 글자색 지정
green_color=(0, 255, 0)
red_color=(0, 0, 255)

while True:

    hasFrame, img_frame = cap.read() #hasFrame은 다음 프레임 사진이 있는지 참,거짓으로 반환하는 함수로 만약 없다면 이 과정이 끝나고 영상을 배포한다.
    if not hasFrame:
        print('더 이상 처리할 frame이 없습니다.')
        break

    #아래 과정은 단일 사진의 Object_Detection과 똑같다.   
    rows = img_frame.shape[0]
    cols = img_frame.shape[1]
    # 원본 이미지 배열 BGR을 RGB로 변환하여 배열 입력
    cv_net.setInput(cv2.dnn.blobFromImage(img_frame,  swapRB=True, crop=False))
    
    start= time.time()
    # Object Detection 수행하여 결과를 cv_out으로 반환 
    cv_out = cv_net.forward()
    frame_index = 0
    # detected 된 object들을 iteration 하면서 정보 추출
    for detection in cv_out[0,0,:,:]:
        score = float(detection[2])
        class_id = int(detection[1])
        # detected된 object들의 score가 0.5 이상만 추출
        if score > 0.5:
            # detected된 object들은 scale된 기준으로 예측되었으므로 다시 원본 이미지 비율로 계산
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            # labels_to_names_0딕셔너리로 class_id값을 클래스명으로 변경.
            caption = "{}: {:.4f}".format(labels_to_names_0[class_id], score)
            #print(class_id, caption)
            #cv2.rectangle()은 인자로 들어온 draw_img에 사각형을 그림. 위치 인자는 반드시 정수형.
            cv2.rectangle(img_frame, (int(left), int(top)), (int(right), int(bottom)), color=green_color, thickness=2)
            cv2.putText(img_frame, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)
    print('Detection 수행 시간:', round(time.time()-start, 2),'초')
    vid_writer.write(img_frame) #Object-Detection이 끝난 영상을 vid_writer.write를 통해 VideoWriter에 설정된 정보대로 수정한다.
# end of while loop

vid_writer.release() # 이런 형식으로
cap.release()    #Object_Detection이 완료된 영상을 배출한다.
