# Images generation
gray = cv2.imread('./markdown_res/Amelia_Vega_0001.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('./markdown_res/Amelia_Vega_0001_gray.bmp', img)
face_cascade = cv2.CascadeClassifier('./faces_data/face.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
face = sorted(faces, key=lambda x: (x[2] * x[3]), reverse=True)[0]
x, y, width, height = face
gray = cv2.rectangle(gray, (x, y), (x + width, y + height), (255, 0, 0), 5)
cv2.imwrite('./markdown_res/Amelia_Vega_0001_face.bmp', gray)
face_gray = gray[y: y + height, x: x + width]
cv2.imwrite('./markdown_res/Amelia_Vega_0001_cropped.bmp', face_gray)

kernel = np.ones((3,3),np.float32)/9
face_gray = cv2.filter2D(face_gray,-1,kernel)
cv2.imwrite('./markdown_res/Amelia_Vega_0001_kernel.bmp', face_gray)

resized = cv2.resize(face_gray, dsize=(40,40), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('./markdown_res/Amelia_Vega_0001_resized.bmp', resized)