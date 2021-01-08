from matplotlib.colors import ListedColormap

def buildCMap(plots) :
    cmaplist = []

    for p0 in plots :
        cmaplist.append(p0.get_color())
        
    return ListedColormap(cmaplist)
