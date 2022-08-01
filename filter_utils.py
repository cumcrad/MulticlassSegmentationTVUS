# ---------------------------------------------------------------------------------------------------------------------------- #
# Utility functions, called within AMF_func.py
# ---------------------------------------------------------------------------------------------------------------------------- #

# Main Adaptive Median Filtering utility function:
def AdaptiveMedianFilter(sMax, width, height, img, newimg, init_filtersize=3):
    borderSize = sMax // 2
    #find image pixel with maximum value:
    imgMax = img.getpixel((0,0))
    for i in range(width):
        for j in range(height):
            if(imgMax < img.getpixel((i,j))):
                imgMax = img.getpixel((i,j))

    #increase filter size until upper bound is reached, apply all filters to everything except border, return filtered image
    for i in range(borderSize,width-borderSize):
        for j in range(borderSize,height-borderSize):
            members = [imgMax] * (sMax*sMax)
            filterSize = init_filtersize
            zxy = img.getpixel((i,j))
            result = zxy
            while(filterSize<=sMax):
                borderS = filterSize // 2
                for k in range(filterSize):
                    for t in range(filterSize):
                        members[k*filterSize+t] = img.getpixel((i+k-borderS,j+t-borderS))
                members.sort()
                med  = (filterSize*filterSize)//2
                zmin = members[0]
                zmax = members[(filterSize-1)*(filterSize+1)]
                zmed = members[med]
                if(zmed<zmax and zmed > zmin):
                    if(zxy>zmin and zxy<zmax):
                        result = zxy
                    else: 
                        result = zmed
                    break
                else:
                    filterSize += 2
            newimg.putpixel((i,j),(result))
    return newimg

# Add noise to border of image:
def renoiseInBorder(borderSize, width, height, newimg):
    for i in range(1,width):
        for j in range(borderSize):
            newimg.putpixel((i,j),newimg.getpixel((i,borderSize)))
            newimg.putpixel((i,height-j-1),newimg.getpixel((i,height-borderSize-1)))
    return newimg