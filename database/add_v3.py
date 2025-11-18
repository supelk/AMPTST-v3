#!/usr/bin/env python3
"""
一键把实验结果 txt 解析并写入 MySQL
用法:  python load_results.py  [results.txt]
"""
import os
import re
import sys
import pymysql
from pathlib import Path

# ---------- 数据库配置 ----------
DB_CFG = dict(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='supelk',   # 改成你的
    database='final',
    charset='utf8mb4'
)

# ---------- 正则 ----------
# BLOCK_SEP = re.compile(r'\n\s*\n')
# BLOCK_SEP = re.compile(r'(?:\r?\n){2,}')
BLOCK_SEP = re.compile(r'(?:\r?\n)\s*(?:\r?\n)')  # 一个空行即可，兼容 CRLF
METRIC_RE = re.compile(r'(mse|mae|mape_i|r2|r2_i):([\d\.e\-]+)', re.I)
TOKEN_RE  = re.compile(r'([a-z]+)(\d+|[A-Z][a-z]*)')  # 字母+数字/True/False

# ---------- 解析 ----------
def parse_block(block: str):
    print('【RAW BLOCK】', repr(block))
    lines = [L.strip() for L in block.splitlines() if L.strip()]
    if len(lines) != 2:
        return None
    param_line, metric_line = lines

    parts = param_line.split('_')
    # print('【PARTS】', len(parts), parts)
    # print('【metric_line】', repr(metric_line))
    # if len(parts) != 15:          # 期望正好 15 段
    #     return None

    # 字段映射表：idx -> (字段名, 转换函数)
    mapping = {
        0:  ('dataset',  lambda v: v),
        1:  ('sl',       int),
        2:  ('pl',       int),
        3:  ('model',    lambda v: v),
        5:  ('ft',       lambda v: v),
        6:  ('ll',       int),
        7:  ('dm',       int),
        8:  ('nh',       int),
        9:  ('el',       int),
        10: ('dl',       int),
        11: ('df',       int),
        12: ('fc',       int),
        13: ('ebtime',   lambda v: v),
        14: ('dt',       lambda v: v),
        15: ('bs',      int),
        16: ('lr',      lambda v: v),
        17: ('describe_',lambda v: v),
    }

    row = {}
    for idx, (field, caster) in mapping.items():
        raw = parts[idx]
        m = TOKEN_RE.fullmatch(raw)
        val = m.group(2) if m else raw
        print(val)
        row[field] = caster(val)

    # 特殊处理 dtTrue
    row['dt'] = 'dtTrue' in param_line

    # 解析指标
    for m in METRIC_RE.finditer(metric_line):
        row[m.group(1).lower()] = float(m.group(2))

    need = {'dataset','sl','pl','mse','mae','mape_i','r2','r2_i'}
    return row if need.issubset(row) else None

# ---------- 批量写入 ----------
def insert_many(rows):
    if not rows:
        return
    cols = ['dataset','sl','pl','model','ft','ll','dm','nh','el','dl','df','fc',
            'ebtime','dt','bs','lr','describe_','mse','mae','mape_i','r2','r2_i']
    for r in rows:
        for c in cols:
            r.setdefault(c, None)
    placeholders = ','.join(['%s']*len(cols))
    upd = ','.join([f'{c}=VALUES({c})' for c in ['mse','mae','mape_i','r2','r2_i']])
    sql = f"""
    INSERT INTO autoDL ({','.join(cols)})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE {upd}
    """
    args = [[row[c] for c in cols] for row in rows]
    with pymysql.connect(**DB_CFG) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, args)
        conn.commit()

# ---------- 主流程 ----------
def main(txt_path: str):
    with open(txt_path, encoding='utf-8') as f:
        content = f.read()
    blocks = BLOCK_SEP.split(content)
    rows = [r for b in blocks if (r := parse_block(b))]
    insert_many(rows)
    print(f'>>> 成功写入 {len(rows)} 条记录')

if __name__ == '__main__':
    # 在 main() 里插入调试
    # with open('test.txt', encoding='utf-8') as f:
    #     content = f.read()
    # blocks = BLOCK_SEP.split(content)
    # rows = [r for b in blocks if (r := parse_block(b))]
    # print('>>> 解析到记录数:', len(rows))  # ← 新增
    # print('>>> 样例:', rows[:2])  # ← 新增

    file = sys.argv[1] if len(sys.argv) > 1 else '../../v100_TL/result_final.txt'
    if not Path(file).exists():
        sys.exit(f'文件不存在: {file}')
    main(file)