import numpy as np
import math

def getSquare(path):
	f = open(path, 'r')
	lines = f.readlines()
	car_pos = lines.pop(0).replace("\n", "").split(",")
	start = lines.pop(0).replace("\n", "").split(",")
	end = lines.pop(0).replace("\n", "").split(",")

	square = []
	for line in lines:
		l = line.replace("\n", "")
		square.append(l.split(","))

	square = np.array(square, dtype=np.float)
	car_pos = np.array(car_pos, dtype=np.float)
	endline = np.array([start,end], dtype=np.float)

	return square, car_pos, endline

def nextPos(x, y, ang, theta):
	r_ang = math.radians(ang)
	r_theta = math.radians(theta)

	nx = x + math.cos( math.radians(ang+theta) ) + (math.sin( math.radians(theta) )*math.sin( math.radians(ang) ))
	ny = y + math.sin( math.radians(ang+theta) ) - (math.sin( math.radians(theta) )*math.cos( math.radians(ang) ))

	nang = math.radians(ang) - math.asin( (math.sin( math.radians(theta) )*2)/6 )
	nang = nang*(180/math.pi)

	#print(nx-x, ny-y, nang)

	return np.array([nx, ny, nang], dtype=np.float)

def draw_sensors(x, y, angle, walls):

    sensor_angles = [angle-45, angle, angle+45]
    sensors=[]
    min_ds=[]
    min_ds_point=[]
    for sensor_angle in sensor_angles:
        sensor_end_x = x + 50*math.cos(math.radians(sensor_angle))
        sensor_end_y = y + 50*math.sin(math.radians(sensor_angle))
        s_line = [[x, y], [sensor_end_x, sensor_end_y]]

        #找出焦點
        points = []
        for i in range(len(walls)-1):
        	w_line = [walls[i], walls[i+1]]
        	#print(s_line, w_line)
        	p = line_intersection(s_line, w_line)
        	if not p==0:
        		points.append(p)
        #計算距離
        distances=[]
        for p in points:
        	distances.append(get_distance([x, y], p))
        ds = np.array(distances)
        minds = np.min(ds)
        minidx = np.argmin(ds)
        	
        min_ds.append(minds)
        min_ds_point.append(points[minidx])
        sensors.append([sensor_end_x, sensor_end_y])

    #print("sensors", sensors)
    #print("min_ds", min_ds)
    #print("min_ds_point", sensors)

    return sensors, min_ds, min_ds_point

def line_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # 計算線段的向量表示
    vec1 = [x2 - x1, y2 - y1]
    vec2 = [x4 - x3, y4 - y3]
    vec3 = [x3 - x1, y3 - y1]

    # 計算線段的交點
    denominator = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if denominator == 0:
        return 0  # 無交點
    t = (vec3[0] * vec2[1] - vec3[1] * vec2[0]) / denominator
    if t < 0 or t > 1:
        return 0  # 交點不在線段上
    u = (vec3[0] * vec1[1] - vec3[1] * vec1[0]) / denominator
    if u < 0 or u > 1:
        return 0  # 交點不在線段上
    p = [x1 + t * vec1[0], y1 + t * vec1[1]]
    return p

def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx*dx + dy*dy)

def inBox(x, y):
	return 18 <= x <= 30 and 37 <= y <= 40