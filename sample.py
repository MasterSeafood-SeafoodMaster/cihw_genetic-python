import Car_toolkit as ct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import shutil

from rbfn import RBFN, load_data
from ga import GeneticAlgorithm, fitness_func, getbestDNA
import os

population_size = 10
num_genes = 40

try:
	os.mkdir("./index/")
except:
	print("folder exist")
for f in os.listdir("./index/"):
    os.remove(os.path.join("./index/", f))

X, y = load_data("./train4dAll.txt")
wList=[]
cList=[]
for i in range(population_size):
	rbfn = RBFN(input_dim=3, hidden_dim=10, output_dim=1)
	rbfn.train(X, y)
	wList.append(rbfn.getDNA().tolist())

w_arr = np.array(wList)
best_w = getbestDNA(w_arr, population_size, 40)
rbfn = RBFN(input_dim=3, hidden_dim=10, output_dim=1)
rbfn.setDNA(best_w)


fig, ax = plt.subplots(figsize=(5, 5))
square, car_pos, endline = ct.getSquare("軌道座標點.txt")
frame=0
while (not(ct.inBox(car_pos[0], car_pos[1])))and frame<100:
	
	plt.xlim([-20, 50])
	plt.ylim([-10, 60])
	sX = square[:, 0]; sY = square[:, 1]
	plt.plot(sX, sY)
	rect = Rectangle((18, 37), 12, 3, linewidth=1, edgecolor='r', facecolor='none')
	circle = Circle((car_pos[0], car_pos[1]), 3, fill=False)
	try:
		sensors, min_ds, min_ds_point = ct.draw_sensors(car_pos[0], car_pos[1], car_pos[2], square)
	except:
		print("failed")
		break
	rs, fs, ls = sensors
	rd, fd, ld = min_ds
	rp, fp, lp = min_ds_point

	sensors_np = np.array([[fd, rd, ld]])
	theta = rbfn.predict(sensors_np)[0]
	#print(theta)

	ax.plot([car_pos[0], fs[0]], [car_pos[1], fs[1]], 'r-', linewidth=2)
	ax.plot([car_pos[0], ls[0]], [car_pos[1], ls[1]], 'g-', linewidth=2)
	ax.plot([car_pos[0], rs[0]], [car_pos[1], rs[1]], 'b-', linewidth=2)

	for i in range(3):
		sp = np.array(min_ds_point[i])
		plt.scatter(sp[0], sp[1])
		plt.annotate(str(round(min_ds[i], 2)), xy=(sp[0], sp[1]), xytext=(sp[0]+0.1, sp[1]+0.1))

	ax.add_artist(circle)
	ax.add_artist(rect)
	car_pos = ct.nextPos(car_pos[0], car_pos[1], car_pos[2], theta)
	
	file_name="./index/"+str(frame)+".png"
	fig.savefig(file_name)
	text = "step "+str(frame)+" success!"
	print(text)
	ax.cla()
	frame += 1

import cv2

# 設定影片參數
fps = 10  # 影片每秒的幀數
size = (500, 500)  # 影片的解析度

# 設定要讀取的圖片資料夾路徑
folder_path = './index'

# 讀取資料夾中所有的圖片路徑
img_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

# 根據檔名的數字排序圖片路徑
img_paths = sorted(img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# 建立影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output.mp4', fourcc, fps, size)

# 依序讀取圖片並寫入影片
for img_path in img_paths:
    img = cv2.imread(img_path)
    cv2.imshow("live", img)
    video_writer.write(img)
    cv2.waitKey(100)

# 釋放影片寫入器資源
video_writer.release()
cv2.destroyAllWindows()