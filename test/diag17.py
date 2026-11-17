# -*- coding: utf-8 -*-
"""调霍夫参数让它检测到 r9c1 和 r9c7"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.recognize import detect_pieces, find_grid

img_path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(img_path)
h_img, w_img = img.shape[:2]
crop = img[int(h_img*0.10):int(h_img*0.90), int(w_img*0.05):int(w_img*0.95)]

pieces = detect_pieces(crop)
row0, col0, cell, score, hit = find_grid(crop, pieces)

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 试不同的霍夫参数组合
configs = [
    ('默认 param2=22', 22, 50),
    ('param2=18', 18, 50),
    ('param2=15', 15, 50),
    ('param2=12', 12, 50),
    ('param2=15, param1=40', 15, 40),
    ('param2=12, param1=40', 12, 40),
    ('param2=10, param1=30', 10, 30),
]

for name, p2, p1 in configs:
    r_min = max(8, int(cell * 0.35))
    r_max = int(cell * 0.60)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=int(cell * 0.55),
        param1=p1, param2=p2, minRadius=r_min, maxRadius=r_max
    )
    hough_set = set()
    if circles is not None:
        for cx, cy, r in np.round(circles[0, :]).astype("int"):
            c = round((cx - col0) / cell)
            ri = round((cy - row0) / cell)
            if 0 <= c < 9 and 0 <= ri < 10:
                hough_set.add((ri, c))
    
    r9_hough = sorted([c for (r, c) in hough_set if r == 9])
    r0_hough = sorted([c for (r, c) in hough_set if r == 0])
    print(f'{name:25s}: 总{len(hough_set):2d}个  row0={r0_hough}({len(r0_hough)})  row9={r9_hough}({len(r9_hough)})')
    print(f'                          r9c1 hough={ (9,1) in hough_set}  r9c7 hough={(9,7) in hough_set}')
