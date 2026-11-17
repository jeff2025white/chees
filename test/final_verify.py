# -*- coding: utf-8 -*-
"""最终验证：无补漏的纯旧 classify + annotate"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import engine.recognize as rec
from engine.decide import decide_position

img_dir = r'g:\chessCn\images'
out_dir = r'g:\chessCn\images\annotated'
os.makedirs(out_dir, exist_ok=True)

imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))

print('=== 最终识别结果（无补漏）===')
for p in imgs:
    name = os.path.basename(p)
    a = rec.analyze_image(p)
    d = decide_position(grid=a.grid_str, side='red', view_side=a.view_side, max_depth=7, time_limit=4.0, top_n=10)
    
    # 生成标注图
    vis = rec.annotate(a, d)
    out_path = os.path.join(out_dir, name.replace('.png', '_annotated.png'))
    cv2.imwrite(out_path, vis)
    
    bm = d.get('best_move', {})
    print(f'{name:30s} R={a.red_count:2d} B={a.black_count:2d}  best={bm.get("chinese","?"):12s} score={bm.get("score_text","?"):>8s}  -> 已保存annotated')

print('\n检查 annotate 目录:')
for fn in sorted(os.listdir(out_dir)):
    fp = os.path.join(out_dir, fn)
    sz = os.path.getsize(fp)
    print(f'  {fn}: {sz:,} bytes')
