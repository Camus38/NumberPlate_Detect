import socket 
import cv2
from os.path import exists
import sys
import time
import numpy as np
import threading
import pytesseract
from _thread import *
import threading


def changeImage(image):
    height,width,channel = image.shape
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    imgTopHat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,structuringElement)
    imgBlackHat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray,imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat,imgBlackHat)

    img_blurred = cv2.GaussianBlur(gray,ksize=(5,5),sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
            img_blurred,maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=0)

    contours, _=cv2.findContours(img_thresh,mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((height,width,channel),dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1,
            color=(255,255,255))
    contours_dict = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result,pt1=(x,y),pt2=(x+w,y+h),
                color=(255,255,255),thickness=2)

        contours_dict.append({
            'contour':contour,
            'x':x,
            'y':y,
            'w':w,
            'h':h,
            'cx':x +(w/2),
            'cy':y +(h/2)
            })

    MIN_AREA = 150 
    MIN_WIDTH, MIN_HEIGHT = 17,45 
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
                    d['idx'] = cnt
                    cnt += 1
                    possible_contours.append(d)
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result,pt1=(d['x'],d['y']),pt2=(d['x']+d['w'],d['y']+d['h']),color=(255,255,255),
                thickness=2)

    def find_chars(contour_list):
        MAX_DIAG_MULTIPLYER = 5
        MAX_ANGLE_DIFF = 8.0
        MAX_AREA_DIFF = 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 3
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue
                dx = abs(d1['cx'] - d2['cx']) #다음배열과의 차이 절대값
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                distance = np.linalg.norm(np.array([d1['cx'],d1['cy']]) - np.array([d2['cx'],d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy/dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] *d2['h'])/(d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w'])/d1['w']
                height_diff = abs(d1['h'] - d2['h'])/d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER\
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF\
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                            matched_contours_idx.append(d2['idx'])

            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []

            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            recursive_contour_list = find_chars(unmatched_contour) #재귀
        
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel),dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result,pt1=(d['x'],d['y']),pt2=(d['x']+d['w'],d['y']+d['h']),color=(255,255,255),thickness=2)

    PLATE_WIDTH_PADDING = 1.0
    PLATE_HEIGHT_PADDING = 1.1
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx'])/2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy'])/2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x'])*PLATE_WIDTH_PADDING +1
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)+1

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'],sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'],sorted_chars[-1]['cy']])
                )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(image, M=rotation_matrix, dsize=(width,height))

        img_cropped = cv2.getRectSubPix(
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)+10),
                center=(int(plate_cx), int(plate_cy))
                )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1]/img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width /2),
            'y': int(plate_cy - plate_height /2),
            'w': int(plate_width),
            'h': int(plate_height)
            })


    longest_idx, longest_text = -1,0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0,0),fx=1.1, fy=1.1)
        plate_img = cv2.cvtColor(plate_img,cv2.COLOR_BGR2GRAY)
        plate_img = cv2.GaussianBlur(plate_img,ksize=(5,5),sigmaX=0)
        _,plate_img = cv2.threshold(plate_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #    contorus, _= cv2.findContours(plate_img,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)
        #plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        #plate_max_x, plate_max_y = 0, 0

        #for contour in contours:
        #    x,y,w,h = cv2.boundingRect(contour)

         #   area = w * h
         #   ratio = w / h

        #    if area > MIN_AREA \
        #    and w > MIN_WIDTH and h > MIN_HEIGHT \
        #    and MIN_RATIO < ratio < MAX_RATIO:
        #                if x < plate_min_x:
        #                    plate_min_x = x
        #                if y < plate_min_y:
        #                    plate_min_y = y
        #                if x + w > plate_max_x:
        #                    plate_max_x = x + w
        #                if y + h > plate_max_y:
        #                    plate_max_y = y + h

        img_out = plate_img
        #img_out = cv2.getRectSubPix(img_out,patchSize=(int(plate_width),int(plate_height)),
        #       center=(int(plate_cx),int(plate_cy)))
        #img_out = cv2.resize(img_out,dsize=(0,0),fx=1.5,fy=1.5)
        #img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        #img_out = cv2.GaussianBlur(img_out,ksize=(3,3),sigmaX=0)
        #_,img_out = cv2.threshold(img_out,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #img_out = cv2.adaptiveThreshold(img_out,255.0,adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #        thresholdType=cv2.THRESH_BINARY_INV,blockSize=19,C=9)
        #img_out = cv2.copyMakeBorder(img_out,top=5,bottom=5,left=5,right=5,borderType=cv2.BORDER_CONSTANT,
        #        value=(0,0,0))
        chars = pytesseract.image_to_string(img_out, lang='kor',config='--psm 7 --oem 1')
        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

    info = plate_infos[longest_idx]
    chars = plate_chars[longest_idx]

    return img_out, chars



def recvall(sock):
    size = sock.recv(4)
    length = int.from_bytes(size,"little")
    data =bytearray()
    while True:
        part = sock.recv(4096)
        data.extend(part)
        if len(data)==length:
            break
    return data

def recv(sock):
    try:
        while True:
            print("a")
            parts = sock.recv(10).decode('utf-8')
            print(parts)
            if len(parts)>3:
                break
    except Exception as e:
        print(e)
    return parts

def getnum(num):
    return num

def endecode(data):
    encoded = np.frombuffer(data,dtype = np.uint8)
    image = cv2.imdecode(encoded,flags=1)
    cimage, chars = changeImage(image)
    encoded_img = cv2.imencode(".jpg",cimage)[1].tobytes()
    return encoded, chars

def threaded(client_socket,addr):
    print('Connected by : ',addr[0],':',addr[1])
    #size = client_socket.recv(4)
    #data = recvall(client_socket)
    while True:
        try:
            data=recvall(client_socket)
            if not data:
                print('Disconnected')
                break
            print('Received from '+addr[0],':',addr[1])
            result,chars = endecode(data)
            print(chars)
            client_socket.send(result)
            #client_socket.send(chars.encode('utf-8'))
            #client_socket.close()
            break
        except Exception as e:
            print(e)
    client_socket.close()


ip = ''
port =30000

server_socket = socket.socket(socket.AF_INET)
server_socket.bind((ip,port))
server_socket.listen()
print('클라이언트 기다리는중...')

try:
    while True:
        print('클라이언트 대기 중 ...')
        client_socket, addr = server_socket.accept()
        #start_new_thread(threaded(client_socket,addr))
        th = threading.Thread(threaded(client_socket,addr))
        th.start()
except Exception as  e:
    print(e)
finally:
    server_socket.close()
server_socket.close()
