import cv2 

def main():
  cap = cv2.VideoCapture(2) 
  while True: 
    ret, frame = cap.read() 
    cv2.imshow("webcam", frame)

    if (cv2.waitKey(38) == 27): 
      break 

if __name__ == "__main__":
  main() 