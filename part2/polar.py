#!/usr/local/bin/python3
#
# Authors: dchodaba - jushah - pyacham
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
import numpy as np
from scipy.ndimage import filters, uniform_filter
import sys
import imageio

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = image.copy()
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)
    
    
## Emission probability calculation by normalizing edge strength for each column    
def get_emission2(column):
    p_emission = column / np.max(column)
    
    return np.clip(p_emission, 1e-9, 1).reshape(p_emission.shape[0],1)

## Initial Probability  calculation. 
## Reasonable assumption is made for air and rock layer boundary problem
def get_initial(image_array):
    
    a = np.arange(image_array.shape[0])
    air_condlist = [a < 50,  a<100, a<175]
    air_choicelist= [0.66/50, 0.33/50, 1e-9]
    
    rock_condlist = [a < 30,  a<120, a<175]
    rock_choicelist= [0.33/50, 0.66/50, 1e-9]
    
    air_initial = np.select(air_condlist, air_choicelist) 
    rock_initial = np.select(rock_condlist, rock_choicelist) 
    
    return  air_initial, rock_initial


## Returns simplified net probablities. 
def get_simplified2(edge_strength):
    
    norm_edge_strength = uint8(255 * edge_strength / (amax(edge_strength))) 
    
    air_ice_simple = []
    rock_ice_simple = []
    for i in range(edge_strength.shape[1]):
        a_i = np.argmax(norm_edge_strength[:,i])
        r_i = np.argmax(norm_edge_strength[:, i]*get_p_rock(edge_strength[:, i], a_i).reshape(-1))
        air_ice_simple.append(a_i)
        rock_ice_simple.append(r_i)  
    return air_ice_simple, rock_ice_simple

## Functions to calculate Viterbi Max State for each pixel

def get_max_state(row, viter_p, lower_limit, upper_limit):

    if row == lower_limit+5:
        return row+np.argmax(viter_p[5:])-5
    elif row==lower_limit+6:
        return row+np.argmax(viter_p[4:])-6
    elif row==lower_limit+7:
        return row+np.argmax(viter_p[3:])-7
    elif row==lower_limit+8:
        return row+np.argmax(viter_p[2:])-8
    elif row==lower_limit+9:
        return row+np.argmax(viter_p[1:])-9
    elif row == upper_limit:
        return row+np.argmax(viter_p[:-1])-10
    elif row == upper_limit+1:
        return row+np.argmax(viter_p[:-2])-10
    elif row == upper_limit+2:
        return row+np.argmax(viter_p[:-3])-10
    elif row == upper_limit+3:
        return row+np.argmax(viter_p[:-4])-10   
    elif row == upper_limit+4:
        return row+np.argmax(viter_p[:-5])-10      
    else:
        return row+np.argmax(viter_p)-10

## Step wise probability function to predict air and rock boundary.
## It is assumed that rock boundary is conditionally NOT independent of Air boundary.
def get_p_rock(col, air_id):
    p_rock = np.zeros(len(col))
    p_rock[air_id:air_id+12] = 0.01
    p_rock[air_id+12:] =1
    return p_rock.reshape(-1,1)
  
def hmm_viterbi3(norm_edge_strength):
    #############
    ###  AIR  ###
    #############
    ## WE will calcualte air first

    ## I am declaring Viterbi array 5 rows above and below the image. I am using smoothness factor for pixel with 5 pixel 
    ## above, same row and below. In order to take advantage of numpy broadcasting I am creating array with 5 rows of pixel
    ## above and below the image. I will clip the viterbi array before starting backtracking.

    V1 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))
    max_state1 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))
    smoothness_factor = np.array([0.1 , 0.1 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.1 , 0.1])

    ## Declare dummy value for additional pixels we added
    max_state1[5:-5,0] = np.arange(norm_edge_strength.shape[0])
    V1[[0,1,2,3,4,-5,-4,-3,-2,-1]] = 1e-100

    # Starting Air probability
    V1[5:-5,0] = norm_edge_strength[:,0]*get_initial(norm_edge_strength)[0]

    #calculating state probabilities of each node using viterbi
    for col in range(1, norm_edge_strength.shape[1]):
        for row in range(5,norm_edge_strength.shape[0]+5):

            viter_p = V1[row-5:row+6, col-1]*smoothness_factor
            maxi = max(1e-100, np.max(viter_p))
            max_state1[row][col] = get_max_state(row, viter_p, 0, norm_edge_strength.shape[0])

            ## We will divide by 100 to avoid overflow
            V1[row][col] = norm_edge_strength[row-5,col]*maxi /100  

    ## Clipping viterbi array        
    V1 = V1[5:-5, :]   
    max_state1 = max_state1[5:-5, :] 

    ## Declaring empty sequence
    opt_air = np.zeros(norm_edge_strength.shape[1]).astype(int)

    ## Fiud best row from last column (end of image)
    best_st = int(np.argmax(V1[:, -1]))
    
    ### Backtracking now
    for col in range(norm_edge_strength.shape[1]-1,-1,-1 ):
        opt_air[col] = int(best_st)
        best_st = int(max_state1[int(best_st)][col])

    ##############
    ###  Rock  ###
    ##############
    ## WE will calcualte rock now

    V2 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))
    max_state2 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))

    max_state2[5:-5,0] = np.arange(norm_edge_strength.shape[0])
    V2[[0,1,2,3,4,-5,-4,-3,-2,-1]] = 1e-100


    V2[opt_air[0]+15:-5, 0] = get_initial(norm_edge_strength)[1][opt_air[0]+10:]*norm_edge_strength[opt_air[0]+10:,0] 
    
    #calculating state probabilities of each node using viterbi

    for col in range(1, norm_edge_strength.shape[1]):

        for row in range(opt_air[col]+20,norm_edge_strength.shape[0]+5):

            viter_p = V2[row-5:row+6, col-1]*smoothness_factor
            maxi = max(1e-100, np.max(viter_p))
            max_state2[row][col] = get_max_state(row, viter_p, opt_air[col]+10, norm_edge_strength.shape[0])

            V2[row][col] = norm_edge_strength[row-5,col]* maxi/100   

    ## Clipping viterbi array        
    V2 = V2[5:-5, :]   
    max_state2 = max_state2[5:-5, :] 

    ## Declaring empty sequence
    opt_rock = np.zeros(norm_edge_strength.shape[1])

    ## Find best row from last column (end of image)
    best_st = int(np.argmax(V2[:, -1]))

    ### Backtracking now
    for col in range(norm_edge_strength.shape[1]-1,-1,-1 ):
        opt_rock[col] = int(best_st)
        best_st = int(max_state2[int(best_st)][col])
    
    return opt_air, opt_rock


## This is similar to Viterbi but with human feedback
## Instead of using initial probability, we are given exact coordinate for 
## one point in rock layer and air layer. We will perform forward and backward propogation from that column.
## It will follow exact same Viterbi logic, except here starting column we are 100% sure of pixel coordiantes

def hmm_viterbi3_human(norm_edge_strength, gt_airice, gt_icerock):
    
    gt_aircol, gt_airrow = gt_airice
    gt_rockcol, gt_rockrow = gt_icerock
    
    #############
    ###  AIR  ###
    #############
    ## WE will calcualte air first

    ## I am declaring Viterbi array 2 rows above and below the image. I am using transition probaility for pixel with 2 pixel 
    ## above, same row and below. In order to take advantage of numpy broadcasting I am creating array with 2 rows of pixel
    ## above and below the image. I will clip the viterbi array before starting backtracking.

    V1 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))
    max_state1 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))
    smoothness_factor = np.array([0.1 , 0.1 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.1 , 0.1])

    ## Declare dummy value for additional pixels we added
    max_state1[5:-5,gt_aircol] = np.arange(norm_edge_strength.shape[0])
    V1[[0,1,2,3,4,-5,-4,-3,-2,-1]] = 1e-100
    V1[gt_airrow+5, gt_aircol] = 1.0

    if gt_aircol > 0:
        #calculating state probabilities of each node using backward for human input
        for col in range(gt_aircol-1, -1, -1):
            for row in range(norm_edge_strength.shape[0], 5, -1):

                viter_p = V1[row-5:row+6, col+1]*smoothness_factor
                maxi = max(1e-100, np.max(viter_p))
                
                max_state1[row][col] = get_max_state(row, viter_p, 0, norm_edge_strength.shape[0])
                V1[row][col] = norm_edge_strength[row-5,col]*maxi /100  
            
            
    #calculating state probabilities of each node using forward for human input
    for col in range(gt_aircol+1, norm_edge_strength.shape[1]):
        for row in range(5,norm_edge_strength.shape[0]+5):

            viter_p = V1[row-5:row+6, col-1]*smoothness_factor
            maxi = max(1e-100, np.max(viter_p))
            max_state1[row][col] = get_max_state(row, viter_p, 0, norm_edge_strength.shape[0])

            ## We will divide by 100 to avoid overflow
            V1[row][col] = norm_edge_strength[row-5,col]*maxi /100  
            
            

    ## Clipping viterbi array        
    V1 = V1[5:-5, :]   
    max_state1 = max_state1[5:-5, :] 

    ## Declaring empty sequence
    opt_air_human = np.zeros(norm_edge_strength.shape[1]).astype(int)

    ## Fiud best row from last column (end of image)
    best_st = int(np.argmax(V1[:, -1]))
    
    ### Backtracking now
    for col in range(norm_edge_strength.shape[1]-1,-1,-1 ):
        opt_air_human[col] = int(best_st)
        best_st = int(max_state1[int(best_st)][col])
        

    ##############
    ###  Rock  ###
    ##############
    ## WE will calcualte rock now

    V2 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))
    max_state2 = np.zeros((norm_edge_strength.shape[0]+10, norm_edge_strength.shape[1]))

    max_state2[5:-5,gt_rockcol] = np.arange(norm_edge_strength.shape[0])
    V2[[0,1,2,3,4,-5,-4,-3,-2,-1]] = 1e-100


    V2[gt_rockrow+5, gt_rockcol] = 1.0
    #calculating state probabilities of each node using viterbi

    if gt_rockcol > 0:
         #calculating state probabilities of each node using backward for human input
        for col in range(gt_rockcol-1, -1, -1):
            for row in range(norm_edge_strength.shape[0], opt_air_human[col]+10, -1):

                viter_p = V2[row-5:row+6, col+1]*smoothness_factor
                maxi = max(1e-100, np.max(viter_p))
                max_state2[row][col] = get_max_state(row, viter_p, 0, norm_edge_strength.shape[0])

                ## We will divide by 100 to avoid overflow
                V2[row][col] = norm_edge_strength[row-5,col]*maxi /100  
            
            
    #calculating state probabilities of each node using forward for human input
    for col in range(gt_rockcol+1, norm_edge_strength.shape[1]):
        for row in range(opt_air_human[col]+20,norm_edge_strength.shape[0]+5):

            viter_p = V2[row-5:row+6, col-1]*smoothness_factor
            maxi = max(1e-100, np.max(viter_p))
            max_state2[row][col] = get_max_state(row, viter_p, 0, norm_edge_strength.shape[0])

            ## We will divide by 100 to avoid overflow
            V2[row][col] = norm_edge_strength[row-5,col]*maxi /100   

    ## Clipping viterbi array        
    V2 = V2[5:-5, :]   
    max_state2 = max_state2[5:-5, :] 

    ## Declaring empty sequence
    opt_rock_human = np.zeros(norm_edge_strength.shape[1])

    ## Find best row from last column (end of image)
    best_st = int(np.argmax(V2[:, -1]))

    ### Backtracking now
    for col in range(norm_edge_strength.shape[1]-1,-1,-1 ):
        opt_rock_human[col] = int(best_st)
        best_st = int(max_state2[int(best_st)][col])
    
    return opt_air_human, opt_rock_human



# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    
    ## We are smoothing the image using uniform filter function and window of 3x3 pixels
    norm_edge_strength = uniform_filter(edge_strength, size = 3, mode = 'nearest')
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    simple_result = get_simplified2(norm_edge_strength)
    hmm_viterbi_result = hmm_viterbi3(norm_edge_strength)
    hmm_viterbi_human_result = hmm_viterbi3_human(norm_edge_strength, gt_airice, gt_icerock)
    

    airice_simple = simple_result[0]
    airice_hmm = hmm_viterbi_result[0]
    airice_feedback= hmm_viterbi_human_result[0]


    icerock_simple = simple_result[1]
    icerock_hmm = hmm_viterbi_result[1]
    icerock_feedback= hmm_viterbi_human_result[1]

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
