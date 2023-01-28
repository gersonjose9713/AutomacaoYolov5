import torch
import cv2
import mss
import numpy as np
import keyboard
from keys import MoveMouse,LeftClick,LeftRelease
import time

modelo = torch.hub.load('ultralytics/yolov5','custom', path='best.pt')

with mss.mss() as sct:
  monitor = {'top': 200, 'left':350, 'width':450, 'height':370}

def clickModelo(modelo_lista):
  if len(modelo_lista) > 0:
    #certeza do modelo [0][4]
    if modelo_lista[0][4] > .80:
      #classe do modelo [0][5] ex: topeira 0 , start 1, restart 2
      if modelo_lista[0][5] == 1 or modelo_lista[0][5] == 2 or modelo_lista[0][5] == 0:
        #cordenadas x y ,minimo modelo [0][0], modelo [0][1]
        #cordenadas x y ,maximos modelo [0][2], modelo [0][3]
        #centro xmin + xmax /2 , ymin + ymax /2
        # x + monitor 'left' , y + monitor 'top'
        x = int(((modelo_lista[0][0] + modelo_lista[0][2])/2)+350)
        y = int(((modelo_lista[0][1] + modelo_lista[0][3])/2)+200)
        MoveMouse(x,y)
        LeftClick()
        LeftRelease()
  
def tela():
  tempo_inicial = time.time()  
  img = np.array(sct.grab(monitor))
  modelo_img = modelo(img)
  modelo_lista = modelo_img.xyxy[0].tolist()
  clickModelo(modelo_lista)
  telinha = np.squeeze(modelo_img.render())
  cv2.imshow(window_name, telinha)
  cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
  print("FPS: ", round(1.0 / (time.time() - tempo_inicial)))
  
print('Starting')
window_name = "Vision"

while True:
  tela()
  if cv2.waitKey(1) & keyboard.is_pressed('q'):
    cv2.destroyAllWindows()
    break
