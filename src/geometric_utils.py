import numpy as np
import cv2

def draw_lines(im, lines):
    for l in lines:
        r,t = l[0]
        p1,p2 = hough_to_projective(r,t)
        cv2.line(im,p1,p2,(255,0,0),1)
    return im

def hough_to_projective(ro,theta):
    # sinus y cosinus
    s,c = [np.sin(theta),np.cos(theta)]
    # Central point of the line
    xb,yb = [c*ro, s*ro]
    # Necessitem dos punts extrems per a construir la línia
    p1 = [int(xb + 10000*(-s)),int(yb + 10000*c)]
    p2 = [int(xb - 10000*(-s)),int(yb - 10000*c)]
    return p1,p2

# a partir de las ecuaciones de las dos rectas establecemos si intersectan, y en que punto intersectan
# utilizamos geometría proyectiva
def intersection_point(l1_eq,l2_eq):
    # producto vectorial de dos vectores da el vector perpendicular a ambos, en este caso el resultado será
    # el punto de intersección
    p1 = np.cross(l1_eq,l2_eq) # p1 = [0,0,0] ----> no hay intersección entre las rectas
    return p1/p1[2]# dividimos ya que tenemos un espacio tridimensional(proyectivo) y así lo pasamos a proyectivo 


def get_lines(im):
    # list to hold the  selected lines
    ultimate_lines = []
    # thresholds for the lines
    r_thres = 100
    # Canny to get the edges
    edges = cv2.Canny(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY), 150, 200, 3)
    # then we will be applying hough transform to the edged image in order to obtain the lines
    lines = cv2.HoughLines(edges,1,np.pi/180, 200) # we only take the lines that have over 200 points
    # we take the first line as a reference
    reference_line = lines[0]
    ultimate_lines.append(reference_line)
    for l in lines[1:]:
        r,t = l[0]
        if abs(r-reference_line[0][0]) > r_thres:# reference_line[0][0] = ro
            ultimate_lines.append(l)
    return ultimate_lines

def get_vanishing_point(lines):
    # A partir de las diferentes lineas obtenidas con Hough obtendremos el punto de Fuga
    pass
    """
    vanishing_points = []
    for line1 in lines:
        for line2 in lines:
            if line1 != line2:"""

