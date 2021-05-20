import numpy as np

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

def get_vanishing_point(lines):
    # A partir de las diferentes lineas obtenidas con Hough obtendremos el punto de Fuga
    """
    vanishing_points = []
    for line1 in lines:
        for line2 in lines:
            if line1 != line2:"""

