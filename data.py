import os, cv2, re, numpy as np
x=[]
y=[]
files=os.listdir('images')
np.random.seed(202105)
np.random.shuffle(files) # 打亂
for file in files:
    name='.'.join(re.split('[.]', file)[:-1])
    img=cv2.imread('images/'+file)
    try:  # skip *.mat, gif data
        img[0]
    except:
        continue
    mask=cv2.imread('annotations/trimaps/'+name+'.png', cv2.IMREAD_UNCHANGED)
    mask=np.where((mask==1)|(mask==3), 255, 0).astype('uint8') # 取1和3的標記
    img=cv2.resize(img, (224,224))
    mask=cv2.resize(mask, (224,224))
    mask=cv2.threshold(mask,-1,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)[-1]
    mask=np.expand_dims(mask, axis=2)
    x.append(img)
    y.append(mask)
x=np.array(x)
y=np.array(y)
print(x.shape, y.shape) # (7384, 224, 224, 3) (7384, 224, 224, 1)

x_train=x[int(len(x)*0.2):]
y_train=y[int(len(x)*0.2):]
x_valid=x[:int(len(x)*0.2)]
y_valid=y[:int(len(x)*0.2)]

x_train=x_train.astype('float32')/255.0
y_train=y_train.astype('float32')/255.0
x_valid=x_valid.astype('float32')/255.0
y_valid=y_valid.astype('float32')/255.0

print(x_train.shape, y_train.shape) # (5908, 224, 224, 3) (5908, 224, 224, 1)
print(x_train.dtype, y_train.dtype) # float32 float32
print(x_valid.shape, y_valid.shape) # (1476, 224, 224, 3) (1476, 224, 224, 1)
print(x_valid.dtype, y_valid.dtype) # float32 float32

