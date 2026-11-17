import json, os
work = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work"
for fn in sorted(os.listdir(work)):
    if fn.startswith("result_") and fn.endswith(".json"):
        with open(os.path.join(work, fn), "r", encoding="utf-8") as f:
            r = json.load(f)
        b = r.get("best_move")
        alt = r.get("alternatives", [])
        print(f"\n{'='*60}")
        print(fn.replace('result_','').replace('.json',''))
        print(f"  ok={r.get('ok')}, legal={r.get('input',{}).get('legal_move_count')}, side={r.get('input',{}).get('side_to_move')}")
        if b:
            print(f"  🎯 Best: {b['chinese']} ({b['iccs']}) score={b['score_text']}")
            print(f"     from={b['from']} → to={b['to']}  piece={b['piece']}  cap={b['captured']}  check={b['gives_check']}  mate={b['is_mate']}")
            print(f"     reason: {b['reason'][:100]}")
            for i, a in enumerate(alt):
                print(f"  #{i+2}: {a['chinese']} ({a['iccs']}) score={a['score_text']}")
        else:
            print(f"  ⚠️ 无着法: {r.get('message')}")
