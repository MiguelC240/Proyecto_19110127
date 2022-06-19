import cv2
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
AlexaClassif = cv2.CascadeClassifier('cascade.xml')
while True:
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    toy = AlexaClassif.detectMultiScale(gray,
    scaleFactor = 5,
    minNeighbors = 47,
    minSize=(80,120))#,
    #maxSize=(150,150))
    #Gray es donde va a detectar los objeto
    #scaleFactor: Especifica que tanto se reducira la imagen
    #Specifica cuantos vecinos debe de tener cada rectangulo candidato
    #Tamaño minimo del objeto a detectar
    #Tamaño mazimo del objeto a detectar
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Alexa',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
