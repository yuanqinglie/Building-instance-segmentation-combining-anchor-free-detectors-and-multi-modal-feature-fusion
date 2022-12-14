

# read bounding box coordinates

import numpy as np

txt_path = '/Building instance segmentation/boxes.txt'	# txt path
image_path = '/Building instance segmentation/label/' 	# Image path
saveBasePath = '/Building instance segmentation/'
f = open(txt_path)
ftrain = open(os.path.join(saveBasePath,'train1.txt'), 'w')
data_lists = f.readlines()	#str

dataset= [ ]
j=int(data_lists[0].split(',')[0])
k=0

while k!=len(data_lists):

  ftrain.write(data_lists[k].split(',')[0])
  if data_lists[k].split(',')[1] == '\n':
    ftrain.write(" "+"0,0,0,0,0"+"\n")
    k=k+1
    j=int(data_lists[k].split(',')[0])
  else:

     while j ==int(data_lists[k].split(',')[0]) :

       data1 = data_lists[k].strip('\n').strip(',').split(',') #Remove the line breaks at the beginning and "，" as spacers
       data2 = data_lists[k+1].strip('\n').strip(',').split(',')
       x1,y1=np.array(data1[2:4]).astype("float")
       x2,y2=np.array(data2[2:4]).astype("float")
       xmin, xmax= np.sort([x1,x2])
       ymin, ymax= np.sort([y1,y2])


       img_label=rasterio.open(image_path+data2[0]+".TIF")
       affineT=np.linalg.inv(np.asarray(img_label.transform).reshape(3,3))
       Y=np.array( [ [xmin, xmax],[ymin, ymax] ] )
       YT=np.dot(affineT,np.vstack(( Y,np.array([1,1]))) ) # d = np.hstack((a,b))
    
       ftrain.write(" "+",".join( [item for item in YT[0:2,:].reshape(1,4)[0].astype(np.str)])+ ','+"1" ) 

       k=k+2
       if k==len(data_lists):
         break


     if k==len(data_lists):
         break
     else:
       ftrain.write("\n")
       j=int(data_lists[k].split(',')[0])

 

ftrain.close()


#dataset = np.array(dataset).astype("float").reshape(-1,4)
#print('Bounding box coordinates shape:',dataset.shape)

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# Test transformed data
from utils.utils import draw_gaussian, gaussian_radius
import math
img=rasterio.open("/Building instance segmentation/image/1944.TIF") #247 185 146 206 173
old_img=np.moveaxis(img.read()[0:3],0,2).astype(np.int32)
#old_img = np.uint8( (cv2.cvtColor(old_img[:,:,0:3], cv2.COLOR_BGR2RGB))*255 )
Boxex=open('/Building instance segmentation/train1.txt').readlines()
line=Boxex[1765].split()

box=np.zeros((len(line[1:]),4))
j=0
fig, ax = plt.subplots(1, figsize=(10,10))
colors =  random_colors(box.shape[0])
heatmap=np.zeros((512,512))
for i in line[1:]:
   box[j]=np.asarray(i.split(',')).astype(float).reshape(1,5)[0][0:4]
   x1, x2, y1, y2 = box[j]
   #print(x1, x2, y1, y2 )
   color = colors[2]

   p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
   ax.scatter([x1,x2], [y1,y2], c=color, marker='o')
   
   ax.add_patch(p)

   # get heatmaps
   h=y2-y1
   w=x2-x1
   ct = np.array([(x1+x2) / 2, (y1+y2) / 2], dtype=np.float32)
   ct_int = ct.astype(np.int32)
   radius = gaussian_radius((math.ceil(h), math.ceil(w)))
   radius = max(0, int(radius))
   heatmap = draw_gaussian(heatmap, ct_int, radius)

   
   j=j+1


ax.imshow(old_img)
ax.axis('off')

plt.figure(figsize=(5,5))
plt.imshow(heatmap)
plt.axis('off')
#plt.scatter([(box[:,0]+box[:,1])/2], [(box[:,2]+box[:,3])/2], c='r', marker='o',linewidths=0.001)

heatmap1 = np.uint8(heatmap*255)
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
alpha=0.7
fused=alpha*old_img+(1-alpha)*heatmap1

cv2_imshow(fused)
