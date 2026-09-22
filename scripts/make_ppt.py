#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景拾光 —— 360° 多人专属高光生成系统 · 产品演示 PPT 生成器
风格:影石 Insta360 式黑色高级感(纯黑底 / 白字 / 品牌黄点缀 / 大图占位)

用法:
    python scripts/make_ppt.py
输出:
    全景拾光_产品演示.pptx (16:9)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------- palette
BG     = RGBColor(0x0A, 0x0A, 0x0C)   # 近纯黑
PANEL  = RGBColor(0x13, 0x13, 0x17)   # 卡片底
PANEL2 = RGBColor(0x1A, 0x1A, 0x20)   # 浅一档
LINE   = RGBColor(0x2E, 0x2E, 0x36)   # 发丝线
WHITE  = RGBColor(0xF5, 0xF5, 0xF7)
GREY   = RGBColor(0x9C, 0x9C, 0xA6)
DIM    = RGBColor(0x5C, 0x5C, 0x66)
GHOST  = RGBColor(0x16, 0x16, 0x1A)   # 背景幽灵字
YELLOW = RGBColor(0xFF, 0xD5, 0x00)   # 影石品牌黄
FONT   = "PingFang SC"

EMU_W, EMU_H = Inches(13.333), Inches(7.5)
MARGIN = 0.9
CW = 13.333 - MARGIN * 2              # 内容宽 11.53

prs = Presentation()
prs.slide_width, prs.slide_height = EMU_W, EMU_H
BLANK = prs.slide_layouts[6]

# ---------------------------------------------------------------- helpers
def _noshadow(sp):
    sp.shadow.inherit = False
    return sp

def _ea(run, name=FONT):
    """同时设置中文字体(east asian)。"""
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', name)

def _dash(line_fmt, val="dash"):
    ln = line_fmt._get_or_add_ln()
    for el in ln.findall(qn('a:prstDash')):
        ln.remove(el)
    ln.append(ln.makeelement(qn('a:prstDash'), {'val': val}))

def slide_new():
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, EMU_W, EMU_H)
    r.fill.solid(); r.fill.fore_color.rgb = BG
    r.line.fill.background()
    return _noshadow(r) and s

def shp(slide, kind, x, y, w, h, fill=PANEL, line=None, lw=1.0, dash=False, rot=0):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    _noshadow(s)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(lw)
        if dash:
            _dash(s.line)
    if rot:
        s.rotation = rot
    return s

def card(slide, x, y, w, h, fill=PANEL, line=None, radius=0.06, dash=False):
    s = shp(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h,
            fill=fill, line=line, lw=1.0, dash=dash)
    try:
        s.adjustments[0] = radius
    except Exception:
        pass
    return s

def tb(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf

def para(tf, first=True):
    return tf.paragraphs[0] if first else tf.add_paragraph()

def run(p, text, size=14, color=WHITE, bold=False, spc=None):
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color
    _ea(r)
    if spc is not None:
        r._r.get_or_add_rPr().set('spc', str(spc))
    return r

def text(slide, x, y, w, h, s, size=14, color=WHITE, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spc=None,
         leading=None, after=None):
    tf = tb(slide, x, y, w, h, anchor)
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        p = para(tf, first=(i == 0))
        p.alignment = align
        if leading: p.line_spacing = leading
        if after is not None: p.space_after = Pt(after)
        run(p, ln, size, color, bold, spc)
    return tf

def kicker(slide, x, y, s, color=YELLOW, size=11):
    return text(slide, x, y, 6, 0.3, s, size=size, color=color, bold=True, spc=320)

def hline(slide, x, y, w, color=LINE, weight=0.75):
    ln = shp(slide, MSO_SHAPE.RECTANGLE, x, y, w, 0.001, fill=None, line=color, lw=weight)
    return ln

def header(slide, idx, section, title, sub=None):
    """内容页统一页眉:黄块 + 序号栏目 + 大标题 + 发丝线 + 幽灵序号。"""
    text(slide, 8.2, 0.28, 4.6, 1.4, idx, size=110, color=GHOST, bold=True,
         align=PP_ALIGN.RIGHT)
    shp(slide, MSO_SHAPE.RECTANGLE, MARGIN, 0.62, 0.09, 0.09, fill=YELLOW)
    kicker(slide, MARGIN + 0.22, 0.52, f"{idx} · {section}")
    text(slide, MARGIN, 0.86, 9.5, 0.75, title, size=30, bold=True)
    if sub:
        text(slide, MARGIN, 1.44, 10.5, 0.35, sub, size=12, color=GREY)
    hline(slide, MARGIN, 1.86, CW)

def footer(slide, num):
    hline(slide, MARGIN, 7.06, CW, color=RGBColor(0x1E, 0x1E, 0x24))
    text(slide, MARGIN, 7.14, 5, 0.25, "全景拾光 · PANORAMA MOMENTS",
         size=8, color=DIM, spc=200)
    text(slide, 12.0, 7.14, 0.43, 0.25, f"{num:02d}", size=8, color=DIM,
         align=PP_ALIGN.RIGHT)

# ---------------- 图片 / 视频占位 ----------------
def img_icon(slide, cx, cy, s, color=DIM):
    """极简图片符号:方框 + 太阳 + 山。"""
    b = card(slide, cx - s / 2, cy - s * 0.39, s, s * 0.78,
             fill=None, line=color, radius=0.18)
    b.line.width = Pt(1.2)
    shp(slide, MSO_SHAPE.OVAL, cx - s * 0.24, cy - s * 0.2, s * 0.15, s * 0.15,
        fill=color)
    shp(slide, MSO_SHAPE.ISOSCELES_TRIANGLE, cx - s * 0.3, cy - s * 0.03,
        s * 0.6, s * 0.3, fill=color)

def img_ph(slide, x, y, w, h, label="图片占位", hint="", fill=PANEL):
    card(slide, x, y, w, h, fill=fill, line=LINE, dash=True, radius=0.05)
    img_icon(slide, x + w / 2, y + h / 2 - (0.14 if hint else 0.06), 0.52)
    ty = y + h / 2 + (0.18 if hint else 0.24)
    text(slide, x + 0.1, ty, w - 0.2, 0.3, label, size=11, color=GREY,
         align=PP_ALIGN.CENTER)
    if hint:
        text(slide, x + 0.1, ty + 0.26, w - 0.2, 0.25, hint, size=8.5,
             color=DIM, align=PP_ALIGN.CENTER)

def video_ph(slide, x, y, w, h, label="视频占位", dur="60s", hint=""):
    card(slide, x, y, w, h, fill=PANEL, line=LINE, dash=True, radius=0.05)
    cy = y + h / 2 - 0.28
    ring = shp(slide, MSO_SHAPE.OVAL, x + w / 2 - 0.31, cy - 0.31, 0.62, 0.62,
               fill=None, line=YELLOW, lw=1.5)
    shp(slide, MSO_SHAPE.ISOSCELES_TRIANGLE, x + w / 2 - 0.09, cy - 0.13,
        0.24, 0.26, fill=YELLOW, rot=90)
    text(slide, x + 0.1, cy + 0.44, w - 0.2, 0.3, label, size=11, color=GREY,
         align=PP_ALIGN.CENTER)
    if hint:
        text(slide, x + 0.1, cy + 0.7, w - 0.2, 0.25, hint, size=8.5,
             color=DIM, align=PP_ALIGN.CENTER)
    chip = card(slide, x + w - 0.78, y + 0.14, 0.62, 0.26, fill=PANEL2,
                line=LINE, radius=0.5)
    tf = tb(slide, x + w - 0.78, y + 0.155, 0.62, 0.22, MSO_ANCHOR.MIDDLE)
    p = para(tf); p.alignment = PP_ALIGN.CENTER
    run(p, dur, 9, YELLOW, True)

def chip(slide, x, y, w, s, color=GREY, border=LINE, size=9.5):
    card(slide, x, y, w, 0.3, fill=None, line=border, radius=0.5)
    text(slide, x, y + 0.025, w, 0.25, s, size=size, color=color,
         align=PP_ALIGN.CENTER)

# ================================================================ S1 封面
s = slide_new()
# 装饰:右上黄色细环 + 小圆点
ring = shp(s, MSO_SHAPE.OVAL, 9.1, -2.3, 6.4, 6.4, fill=None, line=YELLOW, lw=1.2)
shp(s, MSO_SHAPE.OVAL, 11.9, 5.6, 0.14, 0.14, fill=YELLOW)
shp(s, MSO_SHAPE.OVAL, 0.7, 6.4, 0.07, 0.07, fill=DIM)
kicker(s, MARGIN, 0.7, "INSTA360 × SOLODIRECTOR · AI 影像创新赛道")
text(s, MARGIN - 0.06, 1.55, 7.5, 1.7, "全景拾光", size=76, bold=True, spc=60)
text(s, MARGIN, 3.05, 7.0, 0.5, "360° 多人专属高光生成系统", size=21,
     color=WHITE, bold=False)
text(s, MARGIN, 3.62, 7.0, 0.4, "「 一段全景,人人都是主角 」", size=13,
     color=YELLOW)
hline(s, MARGIN, 4.25, 3.2, color=LINE)
for i, t in enumerate(["多人物检测与持续跟踪", "精彩事件智能识别",
                       "一键生成专属合集"]):
    y = 4.5 + i * 0.42
    shp(s, MSO_SHAPE.OVAL, MARGIN + 0.02, y + 0.09, 0.07, 0.07, fill=YELLOW)
    text(s, MARGIN + 0.24, y, 5.5, 0.35, t, size=12.5, color=GREY)
video_ph(s, 7.75, 2.35, 4.68, 3.55, label="产品概念视频 · 建议 60s",
         dur="60s", hint="替换:360° 骑行 / 婚礼现场环绕画面")
text(s, MARGIN, 6.85, 8, 0.3, "汇报人:XXX    |    2026", size=10, color=DIM)
text(s, 10.2, 6.85, 2.2, 0.3, "PANORAMA MOMENTS", size=8, color=DIM,
     align=PP_ALIGN.RIGHT, spc=250)

# ================================================================ S2 目录
s = slide_new()
text(s, 8.6, 0.1, 4.4, 1.6, "MENU", size=100, color=GHOST, bold=True,
     align=PP_ALIGN.RIGHT, spc=100)
shp(s, MSO_SHAPE.RECTANGLE, MARGIN, 0.75, 0.09, 0.09, fill=YELLOW)
kicker(s, MARGIN + 0.22, 0.65, "CONTENTS")
text(s, MARGIN, 1.0, 6, 0.8, "目录", size=34, bold=True)
items = [
    ("01", "产品概述", "痛点 · 方案 · 核心价值"),
    ("02", "用户场景", "五大场景 · 使用旅程"),
    ("03", "功能演示", "三大功能 · 界面流程"),
    ("04", "技术实现", "系统架构 · 关键技术"),
    ("05", "商业潜力", "市场机会 · 商业模式"),
    ("06", "团队介绍", "成员 · 分工"),
]
for i, (no, t, d) in enumerate(items):
    col, row = i % 2, i // 2
    x = MARGIN + col * 5.9
    y = 2.35 + row * 1.5
    card(s, x, y, 5.6, 1.24, fill=PANEL, radius=0.09)
    text(s, x + 0.35, y + 0.24, 1.0, 0.75, no, size=30, color=YELLOW, bold=True)
    text(s, x + 1.35, y + 0.26, 3.9, 0.45, t, size=17, bold=True)
    text(s, x + 1.35, y + 0.72, 4.0, 0.35, d, size=10.5, color=GREY)
    shp(s, MSO_SHAPE.RECTANGLE, x + 5.28, y + 0.52, 0.14, 0.02, fill=LINE)
footer(s, 2)

# ================================================================ S3 产品概述 · 痛点
s = slide_new()
header(s, "01", "产品概述", "360° 记录很美好,回看整理是灾难")
pains = [
    ("素材过长", "1 小时全景素材 = 1 小时回看\n无人有耐心看完全部画面"),
    ("人物难找", "360° 画面信息爆炸\n在全景里找一个孩子如同大海捞针"),
    ("精彩难筛", "欢呼、拥抱、冲刺埋在长片中\n精彩片段难以被定位"),
    ("剪辑耗时", "全景重取景 + 多机位剪辑门槛高\n专业后期动辄数小时"),
]
for i, (t, d) in enumerate(pains):
    x = MARGIN + i * (2.74 + 0.19)
    card(s, x, 2.2, 2.74, 2.5, fill=PANEL, radius=0.07)
    text(s, x + 0.26, 2.44, 1.2, 0.5, f"{i+1:02d}", size=22, color=YELLOW,
         bold=True)
    hline(s, x + 0.27, 3.0, 0.5, color=YELLOW, weight=1.5)
    text(s, x + 0.26, 3.14, 2.2, 0.4, t, size=16, bold=True)
    text(s, x + 0.26, 3.62, 2.28, 1.0, d, size=10, color=GREY, leading=1.35)
bar = card(s, MARGIN, 5.1, CW, 1.15, fill=PANEL2, radius=0.1)
shp(s, MSO_SHAPE.RECTANGLE, MARGIN, 5.1, 0.07, 1.15, fill=YELLOW)
text(s, MARGIN + 0.4, 5.32, 2.6, 0.4, "我们的答案", size=13, color=YELLOW,
     bold=True)
text(s, MARGIN + 0.4, 5.72, 9.8, 0.45,
     "无需提前决定拍摄主角 —— 一段 360° 视频,为每个人生成专属高光合集",
     size=15, bold=True)
text(s, 9.3, 5.32, 3.0, 0.4, "PANORAMA MOMENTS", size=8, color=DIM,
     align=PP_ALIGN.RIGHT, spc=220)
footer(s, 3)

# ================================================================ S4 产品概述 · 产品是什么
s = slide_new()
header(s, "01", "产品概述", "一段视频进,N 个专属合集出",
       sub="全景拾光 = AI 时代下的 360° 影像整理引擎")
vals = [
    ("无需预设主角", "拍摄时不用对着某个人\n事后为每位参与者各剪一版"),
    ("一人一集", "自动检测并跟踪每个人\n生成个人专属高光合集"),
    ("互动不遗漏", "保留全体欢呼、拥抱、庆祝\n生成全体互动合集与时间轴"),
]
for i, (t, d) in enumerate(vals):
    y = 2.25 + i * 1.42
    card(s, MARGIN, y, 5.35, 1.24, fill=PANEL, radius=0.09)
    text(s, MARGIN + 0.32, y + 0.2, 0.6, 0.5, f"0{i+1}", size=20,
         color=YELLOW, bold=True)
    text(s, MARGIN + 1.05, y + 0.19, 4.1, 0.4, t, size=15, bold=True)
    text(s, MARGIN + 1.05, y + 0.58, 4.1, 0.6, d, size=10, color=GREY,
         leading=1.3)
video_ph(s, 6.7, 2.25, 5.73, 3.62, label="产品宣传视频 · 建议 60–90s",
         dur="90s", hint="替换:App 实际操作录屏")
for i, (num, unit, lab) in enumerate([("1", "段素材", "一次导入"),
                                      ("N+1", "个合集", "个人 + 全体"),
                                      ("3", "分钟", "全程自动出片")]):
    x = 6.7 + i * 1.97
    text(s, x, 6.1, 1.8, 0.5, num, size=26, color=YELLOW, bold=True)
    text(s, x, 6.55, 1.8, 0.3, f"{unit} · {lab}", size=9.5, color=GREY)
text(s, MARGIN, 6.35, 5.35, 0.5, "对比传统人工剪辑:2 小时 → 3 分钟",
     size=12, color=GREY)
text(s, MARGIN, 6.62, 5.35, 0.4, "效率提升约 40×", size=12, color=YELLOW,
     bold=True)
footer(s, 4)

# ================================================================ S5 用户场景
s = slide_new()
header(s, "02", "用户场景", "哪里有团聚与热爱,哪里就有全景拾光")
scenes = [
    ("亲子活动", "运动会、毕业典礼\n不错过孩子的每个瞬间"),
    ("旅行出游", "朋友圈旅行大片\n人人都有自己的版本"),
    ("骑行运动", "冲刺、爬坡、过弯\n自动生成运动高光"),
    ("朋友聚会", "生日、露营、跨年\n留住每一次欢笑"),
    ("婚礼庆典", "新人、父母、伴郎伴娘\n每位主角一集回忆"),
]
cw_, gap = 2.14, 0.2
for i, (t, d) in enumerate(scenes):
    x = MARGIN + i * (cw_ + gap)
    card(s, x, 2.2, cw_, 4.35, fill=PANEL, radius=0.06)
    img_ph(s, x + 0.13, 2.33, cw_ - 0.26, 2.35, label="场景图",
           hint="16:10", fill=PANEL2)
    text(s, x + 0.22, 4.92, cw_ - 0.4, 0.3, f"0{i+1}", size=12,
         color=YELLOW, bold=True)
    text(s, x + 0.22, 5.24, cw_ - 0.4, 0.4, t, size=14.5, bold=True)
    text(s, x + 0.22, 5.68, cw_ - 0.4, 0.75, d, size=9, color=GREY,
         leading=1.35)
footer(s, 5)

# ================================================================ S6 用户旅程
s = slide_new()
header(s, "02", "用户场景", "从拍摄到分享,五步走完")
steps = [("拍摄", "Insta360 全景\n自由拍摄"), ("导入", "App 一键导入\n相机素材"),
         ("AI 分析", "检测 · 跟踪\n事件识别"), ("选人", "点选「我」\n或全体"),
         ("生成", "合集 / 时间轴\n一键分享")]
bw, gap = 2.02, 0.36
for i, (t, d) in enumerate(steps):
    x = MARGIN + i * (bw + gap)
    card(s, x, 2.25, bw, 1.7, fill=PANEL, radius=0.08)
    text(s, x + 0.24, 2.44, 1.0, 0.4, f"STEP {i+1}", size=9, color=YELLOW,
         bold=True, spc=150)
    text(s, x + 0.24, 2.74, 1.6, 0.45, t, size=16, bold=True)
    text(s, x + 0.24, 3.22, 1.66, 0.6, d, size=9.5, color=GREY, leading=1.3)
    if i < 4:
        text(s, x + bw + 0.015, 2.85, gap, 0.5, "→", size=16, color=YELLOW,
             align=PP_ALIGN.CENTER)
text(s, MARGIN, 4.35, 6, 0.4, "传统方式 vs 全景拾光", size=14, bold=True)
rows = [("整理 1 小时素材", "人工回看 ≥ 60 分钟", "AI 分析 ≈ 3 分钟"),
        ("定位某个人的镜头", "逐帧拖动寻找", "点选人物即得全部镜头"),
        ("出片", "剪辑软件 2 小时起步", "一键生成 3 类成果")]
tx, tw1, tw2, tw3 = MARGIN, 3.3, 4.1, 4.13
for j, htxt in enumerate(["环节", "传统方式", "全景拾光"]):
    w = [tw1, tw2, tw3][j]
    x = tx + sum([tw1, tw2, tw3][:j])
    text(s, x + 0.2, 4.82, w, 0.3, htxt, size=10, color=DIM, bold=True)
hline(s, MARGIN, 5.12, CW)
for i, (a, b, c) in enumerate(rows):
    y = 5.22 + i * 0.52
    text(s, tx + 0.2, y, tw1, 0.35, a, size=11.5, bold=True)
    text(s, tx + tw1 + 0.2, y, tw2, 0.35, b, size=11, color=GREY)
    text(s, tx + tw1 + tw2 + 0.2, y, tw3, 0.35, c, size=11, color=YELLOW)
    if i < 2:
        hline(s, MARGIN, y + 0.42, CW, color=RGBColor(0x1E, 0x1E, 0x24))
footer(s, 6)

# ================================================================ S7 功能演示 · 三大功能
s = slide_new()
header(s, "03", "功能演示", "三大核心功能,一次看懂")
feats = [
    ("多人物检测 · 跟踪 · 命名",
     "YOLO 检测 + ByteTrack 持续跟踪,人脸聚类关联同一人物;\n点按头像即可手动命名,生成「人物墙」。"),
    ("精彩事件识别",
     "多模态大模型 + 音频分析双通道,识别运动、互动、\n庆祝、欢笑等高光事件,并给出可解释高光评分。"),
    ("一键生成三类成果",
     "个人专属合集 / 全体互动合集 / 精彩片段时间轴,\n自动配乐与转场,直接分享社交平台。"),
]
for i, (t, d) in enumerate(feats):
    y = 2.25 + i * 1.52
    card(s, MARGIN, y, 6.1, 1.34, fill=PANEL, radius=0.08)
    shp(s, MSO_SHAPE.RECTANGLE, MARGIN, y, 0.06, 1.34, fill=YELLOW)
    text(s, MARGIN + 0.3, y + 0.17, 0.7, 0.45, f"F{i+1}", size=17,
         color=YELLOW, bold=True)
    text(s, MARGIN + 1.0, y + 0.16, 5.0, 0.4, t, size=14.5, bold=True)
    text(s, MARGIN + 1.0, y + 0.56, 4.95, 0.7, d, size=9.5, color=GREY,
         leading=1.32)
video_ph(s, 7.35, 2.25, 5.08, 4.4, label="功能演示视频 · 建议 90s",
         dur="90s", hint="替换:三大功能完整演示录屏")
footer(s, 7)

# ================================================================ S8 功能演示 · 界面流程
s = slide_new()
header(s, "03", "功能演示", "App 界面一览", sub="以下为界面截图占位,请替换为实际 UI 截图")
uis = [("素材导入", "相机直连 / 相册导入"), ("人物墙", "检测头像 · 手动命名"),
       ("精彩时间轴", "事件标记 · 类型筛选"), ("合集预览", "一键导出 · 分享")]
pw, gap = 2.42, 0.62
total = 4 * pw + 3 * gap
x0 = (13.333 - total) / 2
for i, (t, d) in enumerate(uis):
    x = x0 + i * (pw + gap)
    card(s, x, 2.3, pw, 3.7, fill=PANEL2, line=LINE, radius=0.14)
    img_ph(s, x + 0.12, 2.42, pw - 0.24, 3.46, label="UI 截图",
           hint="9:19.5", fill=PANEL)
    text(s, x, 6.18, pw, 0.32, f"{i+1}. {t}", size=12.5, bold=True,
         align=PP_ALIGN.CENTER)
    text(s, x, 6.5, pw, 0.3, d, size=9, color=GREY, align=PP_ALIGN.CENTER)
    if i < 3:
        text(s, x + pw + 0.06, 3.85, gap - 0.12, 0.5, "→", size=15,
             color=YELLOW, align=PP_ALIGN.CENTER)
footer(s, 8)

# ================================================================ S9 技术架构
s = slide_new()
header(s, "04", "技术实现", "端到端 AI 高光流水线",
       sub="感知 → 认知 → 生成,三层架构,全自动运行")
layers = [("感知层", 0, 2), ("认知层", 2, 4), ("生成层", 4, 6)]
nodes = [
    ("360° 采集", "Insta360 SDK\n相机直连导入"),
    ("预处理", "等距柱状展开\n关键帧采样"),
    ("检测与跟踪", "YOLO 目标检测\nByteTrack 多目标"),
    ("事件理解", "多模态大模型\n音频情感分析"),
    ("智能剪辑", "高光评分\n最佳视角重取景"),
    ("成果输出", "个人/全体合集\nFFmpeg 渲染"),
]
bw, gap = 1.72, 0.242
for i, (t, d) in enumerate(nodes):
    x = MARGIN + i * (bw + gap)
    card(s, x, 2.7, bw, 1.62, fill=PANEL, radius=0.09)
    text(s, x + 0.16, 2.86, 1.0, 0.3, f"0{i+1}", size=11, color=YELLOW,
         bold=True)
    text(s, x + 0.16, 3.16, bw - 0.3, 0.35, t, size=13, bold=True)
    text(s, x + 0.16, 3.56, bw - 0.3, 0.65, d, size=8.5, color=GREY,
         leading=1.3)
    if i < 5:
        text(s, x + bw - 0.02, 3.25, gap + 0.06, 0.5, "→", size=13,
             color=YELLOW, align=PP_ALIGN.CENTER)
for name, a, b in layers:
    xa = MARGIN + a * (bw + gap)
    xb = MARGIN + b * (bw + gap) - gap
    hline(s, xa, 2.42, xb - xa, color=YELLOW, weight=1.2)
    text(s, xa, 2.1, 2.0, 0.3, name, size=10.5, color=YELLOW, bold=True,
         spc=120)
text(s, MARGIN, 4.75, 8, 0.35, "核心技术栈", size=12, bold=True)
stack = ["YOLOv8 / v11", "ByteTrack", "多模态 LLM", "音频分析 librosa",
         "FFmpeg", "FastAPI", "OpenCV", "PyTorch"]
xx = MARGIN
for t in stack:
    wch = 0.32 + len(t) * 0.105
    chip(s, xx, 5.14, wch, t)
    xx += wch + 0.18
card(s, MARGIN, 5.75, CW, 0.95, fill=PANEL2, radius=0.1)
text(s, MARGIN + 0.35, 5.95, 2.2, 0.4, "工程亮点", size=12, color=YELLOW,
     bold=True)
text(s, MARGIN + 2.1, 5.93, 9.2, 0.5,
     "本地端侧推理,隐私不出设备 · 可解释高光评分 · 事件 JSON 统一前后端契约",
     size=11, color=GREY)
footer(s, 9)

# ================================================================ S10 关键技术
s = slide_new()
header(s, "04", "技术实现", "四项关键技术,支撑三大功能")
tech = [
    ("多目标持续跟踪", "ByteTrack 低分检测关联,遮挡、出画再入画不丢 ID;\n人脸聚类辅助身份合并,人物关联准确率 ≥ 95%。", "ID 保持率", "≥95%"),
    ("多模态事件识别", "画面 + 音频双通道:动作幅度 × 欢呼/掌声峰值 ×\n大模型语义理解,覆盖 8+ 类精彩事件。", "事件类型", "8+ 类"),
    ("可解释高光评分", "动作幅度、构图、清晰度、情感强度多维加权,\n每一分都能说出理由,筛选结果可信可控。", "评分维度", "5 维"),
    ("最佳视角重取景", "360° 自由度变成优势:虚拟运镜自动对准主角,\n平滑轨迹规划,输出电影感平面视频。", "输出规格", "1080P/4K"),
]
for i, (t, d, kl, kv) in enumerate(tech):
    x = MARGIN + (i % 2) * (5.68 + 0.17)
    y = 2.25 + (i // 2) * (2.1 + 0.2)
    card(s, x, y, 5.68, 2.1, fill=PANEL, radius=0.07)
    text(s, x + 0.3, y + 0.22, 4.0, 0.4, t, size=15, bold=True)
    hline(s, x + 0.31, y + 0.62, 0.45, color=YELLOW, weight=1.5)
    text(s, x + 0.3, y + 0.76, 3.95, 1.2, d, size=9.5, color=GREY,
         leading=1.35)
    text(s, x + 4.28, y + 0.55, 1.25, 0.6, kv, size=19, color=YELLOW,
         bold=True, align=PP_ALIGN.RIGHT)
    text(s, x + 4.28, y + 1.1, 1.25, 0.3, kl, size=9, color=DIM,
         align=PP_ALIGN.RIGHT)
footer(s, 10)

# ================================================================ S11 商业潜力
s = slide_new()
header(s, "05", "商业潜力", "一个功能,三种生意")
biz = [
    ("C 端订阅", "相机 App 内增值服务:高级模板、更长时长、云空间\n¥18/月,面向千万级全景相机用户"),
    ("B 端授权", "婚礼摄影机构 / 文旅景区 / 赛事运营方\n按项目或年费授权 SaaS 版"),
    ("硬件协同", "作为旗舰相机的差异化卖点内置\n拉动硬件销量与品牌粘性"),
]
for i, (t, d) in enumerate(biz):
    y = 2.3 + i * 1.5
    card(s, MARGIN, y, 5.9, 1.32, fill=PANEL, radius=0.08)
    text(s, MARGIN + 0.32, y + 0.18, 1.2, 0.45, f"B{i+1}", size=16,
         color=YELLOW, bold=True)
    text(s, MARGIN + 1.15, y + 0.17, 4.6, 0.4, t, size=14.5, bold=True)
    text(s, MARGIN + 1.15, y + 0.57, 4.6, 0.65, d, size=9.5, color=GREY,
         leading=1.3)
img_ph(s, 7.15, 2.3, 5.28, 3.0, label="市场数据图表占位",
       hint="建议:全景相机市场规模 / 短视频用户增长曲线", fill=PANEL)
stats = [("千万级", "全球全景相机\n用户基数"), ("10亿+", "短视频\n月活用户"),
         ("40×", "剪辑效率\n提升")]
for i, (v, l) in enumerate(stats):
    x = 7.15 + i * 1.85
    text(s, x, 5.55, 1.7, 0.45, v, size=20, color=YELLOW, bold=True)
    text(s, x, 6.02, 1.7, 0.55, l, size=8.5, color=GREY, leading=1.3)
text(s, 7.15, 6.62, 5.28, 0.3, "注:数据为公开资料估算,仅供演示",
     size=8, color=DIM)
footer(s, 11)

# ================================================================ S12 团队介绍
s = slide_new()
header(s, "06", "团队介绍", "让每段全景都被重温的人")
team = [("姓名", "队长 · 算法负责人", "目标检测 / 多目标跟踪\n高光评分算法设计"),
        ("姓名", "工程负责人", "相机 SDK 接入\n流水线与渲染工程化"),
        ("姓名", "产品负责人", "场景定义 · 交互设计\n商业模式设计"),
        ("姓名", "视觉与演示", "品牌视觉 · 视频制作\n路演呈现")]
cw2, gap2 = 2.68, 0.27
for i, (n, r, d) in enumerate(team):
    x = MARGIN + i * (cw2 + gap2)
    card(s, x, 2.25, cw2, 4.3, fill=PANEL, radius=0.07)
    img_ph(s, x + 0.14, 2.39, cw2 - 0.28, 2.15, label="成员照片",
           hint="1:1", fill=PANEL2)
    text(s, x + 0.24, 4.72, cw2 - 0.4, 0.4, n, size=15.5, bold=True)
    text(s, x + 0.24, 5.12, cw2 - 0.4, 0.3, r, size=10, color=YELLOW)
    hline(s, x + 0.24, 5.5, cw2 - 0.48, color=RGBColor(0x24, 0x24, 0x2B))
    text(s, x + 0.24, 5.62, cw2 - 0.44, 0.75, d, size=9, color=GREY,
         leading=1.35)
footer(s, 12)

# ================================================================ S13 结尾
s = slide_new()
ring = shp(s, MSO_SHAPE.OVAL, -2.6, 3.4, 7.2, 7.2, fill=None,
           line=RGBColor(0x24, 0x24, 0x1E), lw=1.2)
shp(s, MSO_SHAPE.OVAL, 2.2, 5.9, 0.12, 0.12, fill=YELLOW)
kicker(s, 2.0, 2.05, "THANKS FOR WATCHING")
text(s, 1.94, 2.4, 8.5, 1.3, "谢谢观看", size=60, bold=True, spc=40)
text(s, 2.0, 3.75, 8.0, 0.45, "全景拾光 · 让每一段 360° 回忆,都值得被重温",
     size=15, color=GREY)
hline(s, 2.0, 4.45, 2.6, color=LINE)
text(s, 2.0, 4.7, 7.5, 0.35, "联系:team@example.com   |   微信:xxxxxxx",
     size=11, color=DIM)
img_ph(s, 10.35, 2.3, 1.9, 1.9, label="二维码", hint="1:1", fill=PANEL)
text(s, 9.9, 4.32, 2.8, 0.3, "扫码观看 Demo", size=9, color=DIM,
     align=PP_ALIGN.CENTER)
text(s, 9.35, 6.85, 3.0, 0.3, "PANORAMA MOMENTS", size=8, color=DIM,
     align=PP_ALIGN.RIGHT, spc=250)

# ---------------------------------------------------------------- save
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "全景拾光_产品演示.pptx")
prs.save(out)
print(f"OK -> {out}  ({len(prs.slides._sldIdLst)} slides)")
