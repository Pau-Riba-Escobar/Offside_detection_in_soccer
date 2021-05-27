import numpy as np
import cv2
from numpy.core.numeric import cross

def draw_lines(im, lines):
    for l in lines:
        r,t = l[0]
        p1,p2 = hough_to_projective(r,t)
        cv2.line(im,p1[0:2],p2[0:2],(255,0,0),1)
    return im

def hough_to_projective(ro,theta):
    # sinus y cosinus
    s,c = [np.sin(theta),np.cos(theta)]
    # Central point of the line
    xb,yb = [c*ro, s*ro]
    # Necessitem dos punts extrems per a construir la línia
    p1 = [int(xb + 10000*(-s)),int(yb + 10000*c)]
    p2 = [int(xb - 10000*(-s)),int(yb - 10000*c)]
    return [p1[0],p1[1],1], [p2[0],p2[1],1]

# a partir de las ecuaciones de las dos rectas establecemos si intersectan, y en que punto intersectan
# utilizamos geometría proyectiva
def intersection_point(l1_eq,l2_eq):
    # producto vectorial de dos vectores da el vector perpendicular a ambos, en este caso el resultado será
    # el punto de intersección
    p = np.cross(l1_eq,l2_eq) # p = [0,0,0] ----> no hay intersección entre las rectas
    p =  p if p.all() == 0 else p/p[2] # dividimos ya que tenemos un espacio tridimensional(proyectivo) y así lo pasamos a coord homogeneas
    return np.array([p[0], p[1]])


def get_lines(im):
    # list to hold the  selected lines
    ultimate_lines = []
    # thresholds for the lines
    r_thres = 80
    # Canny to get the edges
    edges = cv2.Canny(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY), 150, 200, 3)
    # T,edges = cv2.threshold(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),200, 255, cv2.THRESH_BINARY)
    # then we will be applying hough transform to the edged image in order to obtain the lines
    lines = cv2.HoughLines(edges,1,np.pi/180, 200) # we only take the lines that have over 200 points
    # we take the first line as a reference
    ref_line = lines[0]
    # getting the line in an euclidean EXPLICIT representation
    p1,p2 = hough_to_projective(ref_line[0][0], ref_line[0][1])
    ref_eq = np.cross(p1,p2)
    ultimate_lines.append(ref_line)
    for l in lines[1:]:
        r,t = l[0]
        t_deg = t*180/np.pi    
        # getting the line in an euclidean EXPLICIT representation
        p1,p2 = hough_to_projective(r, t)
        l_eq = np.cross(p1,p2)
        # testing wether the lines intersect
        p = intersection_point(ref_eq, l_eq)
        # getting the ro parameters of all lines selected until now
        all_ro = np.array(ultimate_lines)[:,0,0]
        # if r verifies the threshold for all the other ro's
        if ((abs(all_ro - r) > r_thres)).all() == True and (p.all() != 0) :# and (t_deg < 80 and t_deg > 95):# reference_line[0][0] = ro
            ultimate_lines.append(l)
            break
    return np.array(ultimate_lines)

def get_vanishing_point(lines):
    # getting the vanishing point as the mean point of the intersecting points of the lines
    intersections = []
    for l1 in lines:
        # getting the equation of the first line
        p1,p2 = hough_to_projective(l1[0][0], l1[0][1])
        l1_eq = np.cross(p1,p2)
        for l2 in lines:
            if np.allclose(l1,l2) == False:
                # getting the equation of the second line
                p3,p4 = hough_to_projective(l2[0][0], l2[0][1])
                l2_eq = np.cross(p3,p4)
                # calculate the intersecting point of the lines
                p = intersection_point(l1_eq,l2_eq)
                # if the point is not in intersections
                if np.array(intersections == p).any() == False:
                    intersections.append(p)
    print(intersections)
    return np.mean(intersections,axis=0)

