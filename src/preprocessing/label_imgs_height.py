import cv2
import random
import argparse
import sys 
import os
import numpy as np

#provides supervised learning labels for our images


def main(input_dir, output_dir):
    if not os.path.exists(str(input_dir)):
        print("Please provide a valid input diretory")
        print(str(input_dir) + " does not exist")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filepath = subdir + os.sep + file
            img = cv2.imread(filepath)
            print(filepath)
            hsv =cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #convert image to hsv
            S=hsv[:,:,1] #take the saturation out of the hsv
            (ret,T)=cv2.threshold(S,42,255,cv2.THRESH_BINARY) #threshold the image
            contours,hierarchy = cv2.findContours(T, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #show contours
            ws = [] #width of all bricks in the image
            hs = [] #height of all bricks in the image
            for c in contours:
                (x,y,w,h) = cv2.boundingRect(c)
                # ws.append(w)
                hs.append(h)
            # avg_brick_width = int(np.mean(ws))
            avg_brick_height = int(np.mean(hs))
            # if not os.path.exists(output_dir + "/" + str(avg_brick_width)):
            #     os.makedirs(output_dir + "/" + str(avg_brick_width))
            save_loc = output_dir + "/" + str(avg_brick_height)
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
            path,dirs,files = next(os.walk(save_loc))
            file_count = '{0}'.format(str(len(files)).zfill(4))
            cv2.imwrite(save_loc + "/" + file_count + ".png", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify data to be labeled')
    parser.add_argument("-i", default=1, help="Specify input data directory")
    parser.add_argument("-o", default=1, help="Specify output data directory")
    args = parser.parse_args()
    i = args.i
    o = args.o
    if i == None:
        print("Please specify an input directory")
        sys.exit(1)
    if o == None:
        print("Please specify an output directory")
        sys.exit(1)

    main(i,o)