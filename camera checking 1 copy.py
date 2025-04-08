import PySimpleGUI as sg
import io
import base64
from PIL import Image
from datetime import datetime
import time
from time import process_time 
import cv2
import numpy as np
import socket
import threading
import json
import os
import pyfeats
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

BORDER_COLOR = '#194369'
DARK_HEADER_COLOR = '#112C45'
# img_HEIGHT = 600
# img_WIDTH = 800
# img_Setting = 400
# image_SIZE = (img_WIDTH, img_HEIGHT)
sizeDisplay = (1920, 1080)
headerHeigh = 50
sizeSetting = 450
padding = 10
sizeImageGUI = (sizeDisplay[0] - sizeSetting - 2 * padding, sizeDisplay[1] - headerHeigh - 2 * padding)
sizeSetting = (sizeSetting - padding, sizeDisplay[1] - headerHeigh - 2* padding)

with open(r"logo MCC.png", "rb") as img_file:
    iconb64 = base64.b64encode(img_file.read())
icon = iconb64

def get_image64(filename):
    with open(filename, "rb") as img_file:
        image_data = base64.b64encode(img_file.read())
    buffer = io.BytesIO()
    imgdata = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(imgdata))
    new_img = img.resize(sizeImageGUI)  # x, y
    new_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue())
    return img_b64

img_b64 = get_image64(r"test2.jpg")

top_banner = [sg.Column([[
    sg.Text('Vision System Camera Checking' + ' ' * 90,
            font=('Any 30'), background_color=DARK_HEADER_COLOR),
    sg.Text(' ', font='Any 30', key='timetext',
            background_color=DARK_HEADER_COLOR,)
]], size=(sizeDisplay[0], headerHeigh), pad=((0, 0), (0, 0)), background_color=DARK_HEADER_COLOR
)]

set_file_saving = [sg.Column(
    [
        [sg.T('Image Configuration', font=('Helvetica', 30, "bold"),
              background_color=BORDER_COLOR,)],
        [sg.T('Image Saving',
              font=('Helvetica', 18, "bold"),
              background_color=BORDER_COLOR)],
        [sg.CB('Realtime Video',
               font=('Helvetica', 12),
               enable_events=True,
               k='-isRealtime-',
               background_color=BORDER_COLOR,
               default=sg.user_settings_get_entry('-isRealtime-', ''))],
        [sg.CB('enable saving',
               font=('Helvetica', 12),
               enable_events=True,
               k='-isSaveImage-',
               background_color=BORDER_COLOR,
               default=sg.user_settings_get_entry('-isSaveImage-', ''))],
        [sg.T('File Location',
              font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-locImage-', ''),
                  key='-locImage-',
                  enable_events=True,
                  size=(50,1),
                  disabled=True,
                  use_readonly_for_disable=False,), sg.FolderBrowse()]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

set_image_load = [sg.Column(
    [
       [sg.T('Load Image',
              font=('Helvetica', 18, "bold"),
              background_color=BORDER_COLOR)],
       [sg.T('File Location',
              font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-fileLoad-', ''),
                  size=(30, 1), key='-fileLoad-',
                  enable_events=True, visible=False),
         sg.FileBrowse()] 
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

set_tcp_ip = [sg.Column(
    [
        [sg.T('TCP/IP configuration', font=('Helvetica',
              18, "bold"), background_color=BORDER_COLOR)],
        [sg.CB('enable TCP', font=('Helvetica', 12), enable_events=True, k='-isTCPActive-',
               background_color=BORDER_COLOR, default=sg.user_settings_get_entry('-isTCPActive-', ''))],
        [sg.T('TCP Server IP : Port', font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-IPSetting-', ''), key='-IPSetting-', size=(15, 1)),
         sg.T(':', font=('Helvetica', 12), background_color=BORDER_COLOR),
         sg.Input(sg.user_settings_get_entry('-PortSetting-', ''),
                  key='-PortSetting-', size=(10, 1)),
         sg.B('update', key='updateIpTcpServer')
         ]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

set_device_id = [sg.Column(
    [
        [sg.T('Device Name', font=('Helvetica', 16), background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-deviceName-', ''), key='-deviceName-', size=(30, 1)),
         sg.B('update', key='updateDevice')]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

Set_Checking_enable = [sg.Column(
    [
        [sg.T('Checking Enable',
              font=('Helvetica', 16),
              background_color=BORDER_COLOR)],
        [sg.CB('enable Checking',
               font=('Helvetica', 14),
               enable_events=True, k='-isCheckEnable-',
               background_color=BORDER_COLOR,
               default=sg.user_settings_get_entry('-isCheckEnable-', ''))]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

set_column = [sg.Column(
    [
        [sg.T('Decision',
              size=(13, 1),
              font=('Helvetica', 35, "bold"),
              background_color=BORDER_COLOR,
              key='-decisionlabel-',
              justification='center')],
        [sg.T('',
              size=(9, 1),
              font=('Helvetica', 50, "bold"),
              background_color=BORDER_COLOR,
              text_color='#FF0000',
              justification='center')]
    ], background_color=BORDER_COLOR
)]

cnt_image = [[sg.Image(data=img_b64, pad=(0, 0), key='image')]]
cnt_setting = [set_file_saving,
               set_image_load,
               set_tcp_ip,
               set_device_id,
               Set_Checking_enable,
               set_column
               ]

content_layout = [
    sg.Column(cnt_image,
              size=sizeImageGUI,
              pad=((10, 10), (10, 10))
              ),
    sg.Column(cnt_setting,
              size=sizeSetting,
              pad=((0, 10), (0, 0)),
              background_color=BORDER_COLOR)]

layout = [top_banner, content_layout,
          [sg.Button('EXIT', button_color=('white', 'firebrick3')),
           sg.Button('capture', button_color=('white', 'firebrick3'))]]
window = sg.Window('Vision System Camera Checking',
                   layout, finalize=True,
                   resizable=True,
                   no_titlebar=True,
                   margins=(0, 0),
                   grab_anywhere=True,
                   icon=icon, location=(0, 0))

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)
cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

TCPEnable = sg.user_settings_get_entry('-isTCPActive-', '')
host = sg.user_settings_get_entry('-IPSetting-', '')
if sg.user_settings_get_entry('-PortSetting-', '') == "":
    port = 5000
else:
    port = int(sg.user_settings_get_entry('-PortSetting-', ''))
    
client_socket = socket.socket()  # instantiate

if TCPEnable:
    try:
        client_socket.connect((host, port))  # connect to the server
    except:
        print("can not connect to server")

# need receive {"checking_request" : true}
def receive_response(client_socket, directory, imageSaving):
    while True:
        try:
            # Menerima respons dari server
            response = client_socket.recv(1024)
            if response:
                print('Menerima respons: {}'.format(response.decode()))
                dataJson = json.loads(response.decode())

                if "request" in dataJson:
                    checking_request = dataJson["request"]
                    if checking_request == "checking":
                        start_time = process_time()
                        capture_image_and_save(
                            client_socket, directory, imageSaving)
                        end_time = process_time()
                        process = end_time - start_time
                        print(f"Sending time: {process:.4f} seconds")
        except:
            break


def capture_image_and_save(client_socket, directory, imageSaving):
    start_time = process_time()
    ret, frame = cap.read()
    frameShow = cv2.resize(frame, sizeImageGUI)
    imgbytes = cv2.imencode('.png', frameShow)[1].tobytes()  # ditto
    window['image'].update(data=imgbytes)
    
    end_time = process_time()
    process = end_time - start_time

    if imageSaving:
        now = datetime.now()
        filename = now.strftime("%Y%m%d%H%M%S%f") + ".png"
        new_file_name = os.path.join(directory, filename)
        cv2.imwrite(new_file_name, frameShow)

    imgbytesSend = cv2.imencode('.png', cv2.resize(frameShow, (650,400)))[1].tobytes()  # ditto
    dataImage = base64.b64encode(imgbytesSend).decode('ascii')
    
    dataResponse = {
        "response": "complete",
        "data": {
            "deviceID": id,
            "deviceName": deviceName,
            "result": 0,
            "resultDescription": "good",
            "imageRaw": dataImage
        }
    }
    
    TCPdataResponse = json.dumps(dataResponse)
    client_socket.sendall(TCPdataResponse.encode())

    print(f"Capturing time: {process:.4f} seconds")
    
def capture_image():
    start_time = process_time()
    ret, frame = cap.read()
    frame = cv2.resize(frame, sizeImageGUI)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
    dataImage = base64.b64encode(imgbytes).decode('ascii')    
    content = base64.b64decode(dataImage)
    window['image'].update(data=content)
    
    end_time = process_time()
    proses = end_time - start_time
    
    print(f"Capturing time: {proses:.4f} seconds")

def load_image_from_folder(folder_path):
    # Get a list of all JPG files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

    if not jpg_files:
        print("No JPG files found in the folder.")
        return None

    # Choose the first JPG file (you can modify this logic based on your requirements)
    selected_file = os.path.join(folder_path, jpg_files[0])

    # Load the selected JPG file
    img_b64 = get_image64(selected_file)

    return img_b64
    
def find_roi(image):
    # point1 = (0, 0)
    # point2 = (750, 550)
    point1 = (1250,1200)
    point2 = (2000,1750)
    # point1 = (1250,1100)
    # point2 = (2000,1850)
    imageROI = image[point1[1]:point2[1], point1[0]:point2[0], :]
    imageGray = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
    return imageGray

def detectContour(img):    
    ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 10
    detected_contours = 0

    sum_contour_pixels = np.zeros_like(img, dtype=np.uint8)
    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area >= min_contour_area:
            detected_contours += 1
            
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            roi = cv2.bitwise_and(img, mask) 
            sum_contour_pixels += roi

    image_kontur = img_as_ubyte(sum_contour_pixels)
    return image_kontur


realtime = sg.user_settings_get_entry('-isRealtime-', '')
isSaving = sg.user_settings_get_entry('-isSaveImage-', '')
directory = sg.user_settings_get_entry('-locImage-', '')
id = 2
deviceName = sg.user_settings_get_entry('-deviceName-', '')

while True:
    window['timetext'].update(time.strftime('%H:%M:%S'))
    event, values = window.read(timeout=20)
    if event == 'EXIT' or event == sg.WIN_CLOSED:
        break           # exit button clicked

    if realtime:
        capture_image()

    if TCPEnable:
        response_thread = threading.Thread(
            target=receive_response, args=(client_socket, directory, isSaving))
        response_thread.daemon = True
        response_thread.start()

    if event == '-locImage-':
        sg.user_settings_set_entry('-locImage-', values['-locImage-'])
        directory = sg.user_settings_get_entry('-locImage-', '')
        window['-isTCPActive-'].update(False)
        TCPEnable = False

    elif event == 'updateIpTcpServer':
        sg.user_settings_set_entry('-IPSetting-', values['-IPSetting-'])
        sg.user_settings_set_entry('-PortSetting-', values['-PortSetting-'])
        host = sg.user_settings_get_entry('-IPSetting-', '')
        port = int(sg.user_settings_get_entry('-PortSetting-', ''))
        
    elif event == 'updateDevice':
        sg.user_settings_set_entry('-deviceName-', values['-deviceName-'])
        deviceName = values['-deviceName-']
        
    elif event == '-isTCPActive-':
        sg.user_settings_set_entry('-isTCPActive-', values['-isTCPActive-'])
        TCPEnable = sg.user_settings_get_entry('-isTCPActive-', '')
        if TCPEnable:
            client_socket = socket.socket()  # instantiate
            try:
                client_socket.connect((host, port))  # connect to the server
            except:
                # Terjadi kesalahan, keluar dari loop
                print("can not connect to server")
        else:
            client_socket.close()

    elif event == '-isSaveImage-':
        sg.user_settings_set_entry('-isSaveImage-', values['-isSaveImage-'])
        isSaving = values['-isSaveImage-']
        print(isSaving)
        
    elif event == '-isCheckEnable-':
        sg.user_settings_set_entry(
            '-isCheckEnable-', values['-isCheckEnable-'])
        if values['-isCheckEnable-']:
            window['-decisionlabel-'].update('Decision')
        else:
            window['-decisionlabel-'].update('')

    elif event == '-isRealtime-':
        sg.user_settings_set_entry('-isRealtime-', values['-isRealtime-'])
        realtime = values['-isRealtime-']
        
    elif event == 'capture':
        capture_image()
    
    elif event == '-loadImage-':
        file_path = values['-fileLoad-']
        if file_path:
            img_b64 = load_image_from_folder(file_path)
            window['image'].update(data=img_b64)

window.close()
