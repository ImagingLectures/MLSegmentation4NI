import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def buildCMap(plots) :
    cmaplist = []

    for p0 in plots :
        cmaplist.append(p0.get_color())
        
    return ListedColormap(cmaplist)

def magnifyRegion(img,roi, figsize, cmap='gray',vmin=0,vmax=0,title='Original') :
    if vmin==vmax:
        vmin=img.min()
        vmax=img.max()
    fig, ax = plt.subplots(1,2,figsize=figsize)
    
    ax[0].imshow(img,cmap=cmap,vmin=vmin, vmax=vmax)
    ax[0].plot([roi[1],roi[3]],[roi[0],roi[0]],'r')
    ax[0].plot([roi[3],roi[3]],[roi[0],roi[2]],'r')
    ax[0].plot([roi[1],roi[3]],[roi[2],roi[2]],'r')
    ax[0].plot([roi[1],roi[1]],[roi[0],roi[2]],'r')
    ax[0].set_title(title)
    subimg=img[roi[0]:roi[2],roi[1]:roi[3]]
    ax[1].imshow(subimg,cmap=cmap,extent=[roi[0],roi[2],roi[1],roi[3]],vmin=vmin, vmax=vmax)
    ax[1].set_title('Magnified ROI')
