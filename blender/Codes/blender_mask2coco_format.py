"""
Sharpening mask images
Please run this code on ImageJ/Fiji (Jython)
"""


from ij import IJ
from ij.plugin import ChannelSplitter, RGBStackMerge
from ij.io import DirectoryChooser
import os


srcDir = DirectoryChooser("Choose Folder").getDirectory()
IJ.log("directory: " + srcDir)

num_img_dict = {'train': 10000, 'val': 2000}

for phase in ['train', 'val']:
    for i in range(num_img_dict[phase]):
        filename = 'img_{:05d}.jpg'.format(i+1)
        img = IJ.openImage(os.path.join(srcDir, phase, filename))
    	im = ChannelSplitter.split(img)
    	for j in range(3):
	    	im[j].getProcessor().threshold(100)
	    	im[j].show()
    	
    	mer_img = RGBStackMerge.mergeChannels([im[0], im[1], im[2]], True)
    	IJ.saveAs(mer_img, "PNG", srcDir+"/{}/mask_{:05d}.PNG".format(phase, i+1))

        img.close()
        im[0].close()
        im[1].close()
        im[2].close()

IJ.log("Finish")
