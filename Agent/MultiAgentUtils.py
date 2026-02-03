import numpy as np

def perpendicular(x:np.array, a:np.array, b:np.array):
    d_ab = np.linalg.norm(a-b)
    d_ax = np.linalg.norm(a-x)
    d_bx = np.linalg.norm(b-x)

    if d_ab != 0:
        if np.dot(a-b,x-b)*np.dot(b-a,x-a) >= 0: # x is between a and b
            px = b[0]-a[0]; py = b[1]-a[1]; dAB = px*px + py*py
            u = ((x[0] - a[0]) * px + (x[1] - a[1]) * py) / dAB
            p = [a[0] + u * px, a[1] + u * py]
        else:
            if d_ax < d_bx:
                p = a
            else:
                p = b
    else: # if a and b are identical
        p = a
    return p

def nearest_point_to_obstacle(pose, obstacle):
    nearest_point = []
    nearest_dis= float(np.inf)
    for i in range(len(obstacle)):
        per = perpendicular(pose, np.array(obstacle[i]), np.array(obstacle[(i+1) % len(obstacle)]))
        dis_per = np.linalg.norm(pose - per)
        if dis_per < nearest_dis:
            nearest_dis = dis_per
            nearest_point = per
    return nearest_point
    