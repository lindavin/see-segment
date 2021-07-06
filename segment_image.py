import random
import matplotlib.pylab as plt
import imageio

from see import JupyterGUI

import argparse

parser = argparse.ArgumentParser(description="Create some csv data files.")

parser.add_argument(
    "--num-gen",
    default=20,
    type=int,
    help="number of generations to run genetic search (default: 20)",
)

parser.add_argument(
    "--pop-size",
    default=20,
    type=int,
    help="population size of each generation to run genetic search (default: 20)",
)

parser.add_argument(
    "--filename",
    required=True,
    type=str,
    help="name of image file to save (i.e. test). File will be saved as png.",
)

#parser.add_argument(
#       "--image-name",
#       type=str,
#       help="name of image found in Image_data/Examples. File extensions is needed."
#        )

args = parser.parse_args()

import imageio
from see.base_classes import pipedata
#img = imageio.imread('Image_data/Examples/Emma.jpg')
#gmask = imageio.imread('Image_data/Examples/Emma_GT.png')
img = imageio.imread('Image_data/Examples/Eboru.png')
gmask = imageio.imread('Image_data/Examples/Eboru_GT.png')
data = pipedata()
data.img = img
data.gmask = gmask

from see import base_classes
from see.Segmentors import segmentor
from see.ColorSpace import colorspace
from see.Workflow import workflow
from see.Segment_Fitness import segment_fitness

workflow.addalgos([colorspace, segmentor, segment_fitness])
wf = workflow()
print(wf)

individual = segmentor()
d = pipedata()
d.img = data.img
d.gmask = data.gmask
individual.runAlgo(d)

from see import GeneticSearch

mydata = base_classes.pipedata()
mydata.img = data.img
mydata.gmask = data.gmask

my_evolver = GeneticSearch.Evolver(workflow, mydata, pop_size=args.pop_size)

# warnings may appear when this runs
population = my_evolver.run(ngen=args.num_gen)

params = my_evolver.hof[0]

print('Best Individual:\n', params)

seg = workflow(params)
data = seg.pipe(data)

plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(data.inputimage)
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(data.mask)
plt.title("Segmentation")
plt.axis('off')

plt.tight_layout
#plt.show()

print("fName: {}".format(args.filename))
plt.savefig("{}.png".format(args.filename))

print('Fitness Value: ', segment_fitness().evaluate(data.mask, data.gmask)[0])
