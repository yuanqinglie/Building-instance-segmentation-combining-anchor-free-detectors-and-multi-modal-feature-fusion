

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def apply_mask(image, mask, color, alpha=0.7):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0 ,image[:, :, c],
                    image[:, :, c] *(1 - alpha) + alpha * color[c] * 255)
                        
    return image

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

def get_box(  line ):
       j=0
       box=np.zeros((len(line[1:]),4))
       for i in line[1:]:
          a =np.asarray(i.split(',')).astype(np.float32).reshape(1,5)[0]
          box[j]=[a[0],a[3],a[1],a[2]]
          j=j+1
       return box

# read bounding box coordinates

import numpy as np

txt_path = '/Building instance segmentation/train1.txt'	# txt文本路径 /content/drive/MyDrive/test3/train.txt
f = open(txt_path)
data_lists = f.readlines()	#读出的是str类型

line=data_lists[1082].strip('\n').split(' ')
dataset=get_box(  line )
print('Bounding box coordinates shape:',dataset.shape)

# transform bounding box coordinates
img=rasterio.open("/Building instance segmentation/image/1186.TIF") #247 185 146 206
img_label=rasterio.open("/Building instance segmentation/label/1186.TIF")
img_label=np.moveaxis(img_label.read(),0,2)[:,:,0]
scores=np.random.randint(96,100,size= dataset.shape[0] )/100

old_img=np.moveaxis(img.read()[0:3],0,2).astype(np.int32)
Image=old_img
#old_img = np.uint8( (cv2.cvtColor(old_img[:,:,0:3], cv2.COLOR_BGR2RGB))*255 )

colors =  random_colors(dataset.shape[0])
red=(1.0, 0.0, 0.1200000000000001)

height, width = old_img.shape[:2]
masks=np.zeros((old_img.shape[0],old_img.shape[1], dataset.shape[0] ))
masked_image = old_img 

fig, ax = plt.subplots(1, figsize=(10,10))

for i in range(dataset.shape[0]):
   color =red #colors[i]
   x1, y1, x2, y2 = dataset[i]

   
   p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4,
                                alpha=0.8, linestyle="dashed",
                                edgecolor=colors[i], facecolor='none')
   ax.add_patch(p)
   
   ax.scatter([x1,x2], [y1,y2], c=color, marker='o')
   ax.scatter([(x1+x2)/2], [(y1+y2)/2], c=color, marker='^')
             
   label = 'Bulding'+str(i+1)
   caption = "{}  {:.2f}".format(label, scores[i])
   """
   ax.text(x1, y2-2, caption,
                color='w', size=9, backgroundcolor='none',alpha=1,rotation=90,bbox ={'facecolor': color,  
                   'alpha':0.3, 'pad':1})
   """
   a= np.trunc(x1 ).astype(int)
   b= np.trunc(y1 ).astype(int)
   c= np.trunc(x2 ).astype(int)
   d= np.trunc(y2 ).astype(int)
   #print(a,b,c,d)

   masks[b:d+1,a:c+1,i]= img_label[b:d+1,a:c+1]/255

   masked_image = apply_mask(masked_image, masks[:, :, i], colors[i],alpha=0.5)

   """
   padded_mask = np.zeros(
            (masks.shape[0] + 2, masks.shape[1] + 2), dtype=np.uint8)
   padded_mask[1:-1, 1:-1] = masks[:, :, i]
   contours = find_contours(padded_mask, 0.5)
   for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
   """


ax.set_ylim(height + 8, -8)
ax.set_xlim(-8, width + 8)
ax.axis('on')
#ax.set_title('Building Instance Segmentation')
ax.imshow(masked_image.astype(np.uint8))
ax.set_xticks([])
ax.set_yticks([])
#plt.scatter(datasetaffine[:,0],datasetaffine[:,2])
#plt.scatter(datasetaffine[:,1],datasetaffine[:,3])
#fig.savefig('/content/drive/MyDrive/146.jpg', dpi=150)
