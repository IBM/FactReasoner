from pptx.util import Pt, Inches
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from palette import *

W, H = Inches(13.333), Inches(7.5)

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def tb(slide, x, y, w, h, text, size=18, bold=False, color=INK, font=FONT,
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, spacing=1.0):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run(); r.text = ln
        f = r.font
        f.size = Pt(size); f.bold = bold; f.italic = italic
        f.color.rgb = color; f.name = font
    return box

def rect(slide, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         adj=None, lw=1.0):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(lw)
    if adj is not None:
        try: s.adjustments[0] = adj
        except Exception: pass
    s.text_frame.text = ""
    return s

def label(shape, text, size=13, bold=True, color=RGBColor(0xFF,0xFF,0xFF), font=FONT):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = font
    return shape

def arrow(slide, x, y, w, h=Inches(0.14), color=HAIR):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    a.shadow.inherit = False
    a.fill.solid(); a.fill.fore_color.rgb = color
    a.line.fill.background()
    return a

# ---- syntax-highlighted code block -----------------------------------------
KW = {"from","import","def","return","await","async","with","as","for","in",
      "if","else","None","True","False","print","class"}

def code(slide, x, y, w, h, src, size=11.5, title=None):
    """Dark code panel with light keyword/string/comment highlighting."""
    if title:
        tb(slide, x, y - Inches(0.28), w, Inches(0.24), title, size=11.5,
           bold=True, color=SUBTLE)
    panel = rect(slide, x, y, w, h, fill=CODEBG, adj=0.03)
    tf = panel.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.14)
    tf.margin_top = tf.margin_bottom = Inches(0.11)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, ln in enumerate(src.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.16
        stripped = ln.strip()
        if stripped.startswith("#"):
            r = p.add_run(); r.text = ln
            r.font.size = Pt(size); r.font.name = MONO; r.font.color.rgb = CODECOM
            continue
        # tokenize on quotes first, then keywords
        for seg, isstr in _split_strings(ln):
            if isstr:
                r = p.add_run(); r.text = seg
                r.font.size = Pt(size); r.font.name = MONO; r.font.color.rgb = CODESTR
            else:
                for word, iskw in _split_words(seg):
                    r = p.add_run(); r.text = word
                    r.font.size = Pt(size); r.font.name = MONO
                    r.font.color.rgb = CODEKEY if iskw else CODEFG
    return panel

def _split_strings(s):
    out, buf, i = [], "", 0
    while i < len(s):
        ch = s[i]
        if ch in "\"'":
            q = ch; j = i + 1
            while j < len(s) and s[j] != q: j += 1
            if buf: out.append((buf, False)); buf = ""
            out.append((s[i:j+1], True)); i = j + 1
        else:
            buf += ch; i += 1
    if buf: out.append((buf, False))
    return out or [("", False)]

def _split_words(s):
    import re
    out = []
    for tok in re.split(r"(\W)", s):
        if tok == "": continue
        out.append((tok, tok in KW))
    return out

def title_bar(slide, kicker, title, sub=None):
    tb(slide, Inches(0.72), Inches(0.42), Inches(11.9), Inches(0.24),
       kicker.upper(), size=11.5, bold=True, color=ACCENT)
    tb(slide, Inches(0.72), Inches(0.70), Inches(11.9), Inches(0.52),
       title, size=31, bold=True, color=INK)
    y = Inches(1.30)
    if sub:
        tb(slide, Inches(0.72), y, Inches(11.9), Inches(0.34), sub,
           size=15.5, color=SUBTLE)
        y = Inches(1.72)
    return y

TOTAL_SLIDES = [21]
_SLIDE_N = [1]


def footer(slide, n=None, total=None):
    """Page footer. Auto-numbers in call order when n is omitted."""
    if n is None:
        # Derive the page number from the deck itself, so it can never drift
        # from slide order regardless of call nesting.
        n = len(slide.part.package.presentation_part.presentation.slides._sldIdLst)
    if total is None:
        total = TOTAL_SLIDES[0]
    tb(slide, Inches(0.72), Inches(7.02), Inches(6.0), Inches(0.22),
       "FactReasoner", size=10, color=RGBColor(0x9A,0xA3,0xB0))
    tb(slide, Inches(11.6), Inches(7.02), Inches(1.0), Inches(0.22),
       f"{n} / {total}", size=10, color=RGBColor(0x9A,0xA3,0xB0), align=PP_ALIGN.RIGHT)


def picture(slide, path, x, y, w=None, h=None):
    """Place an image, preserving aspect ratio when only one dimension is given."""
    if w is not None and h is None:
        return slide.shapes.add_picture(path, x, y, width=w)
    if h is not None and w is None:
        return slide.shapes.add_picture(path, x, y, height=h)
    return slide.shapes.add_picture(path, x, y, width=w, height=h)


def divider(slide, kicker, title, blurb=None, accent=None):
    """Full-bleed section break."""
    from palette import INK, ACCENT as A, BG
    col = accent or A
    rect(slide, 0, 0, W, H, fill=INK, shape=MSO_SHAPE.RECTANGLE)
    rect(slide, 0, 0, Inches(0.09), H, fill=col, shape=MSO_SHAPE.RECTANGLE)
    tb(slide, Inches(1.10), Inches(2.72), Inches(10.6), Inches(0.3),
       kicker.upper(), size=13, bold=True, color=col)
    tb(slide, Inches(1.10), Inches(3.14), Inches(11.0), Inches(0.8),
       title, size=44, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    if blurb:
        tb(slide, Inches(1.10), Inches(4.18), Inches(10.4), Inches(0.7),
           blurb, size=17, color=RGBColor(0xA8, 0xB2, 0xC2), spacing=1.26)


def promptbox(slide, x, y, w, h, text, size=10.5, title=None, accent=None):
    """Light panel holding verbatim prompt text (not syntax-highlighted)."""
    from palette import INK, SUBTLE, ACCENT as A, PANEL as P
    col = accent or A
    if title:
        tb(slide, x, y - Inches(0.27), w, Inches(0.24), title, size=11, bold=True,
           color=col)
    p = rect(slide, x, y, w, h, fill=RGBColor(0xFA, 0xFB, 0xFD), line=RGBColor(0xDD, 0xE3, 0xEB),
             adj=0.03, lw=1.0)
    rect(slide, x, y, Inches(0.045), h, fill=col, shape=MSO_SHAPE.RECTANGLE)
    tf = p.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.20); tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.10); tf.margin_bottom = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, ln in enumerate(text.split("\n")):
        pr = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        pr.alignment = PP_ALIGN.LEFT
        pr.line_spacing = 1.14
        r = pr.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.name = MONO
        r.font.color.rgb = SUBTLE if ln.strip().startswith(("-", "\u2022")) else INK
    return p


def logo(slide, path, x, y, h, caption=None):
    """Mellea logo at a fixed height, with an optional caption to its right."""
    pic = slide.shapes.add_picture(path, x, y, height=h)
    if caption:
        tb(slide, x + pic.width + Inches(0.14), y + h / 2 - Inches(0.13),
           Inches(3.0), Inches(0.26), caption, size=12, bold=True, color=SUBTLE)
    return pic
