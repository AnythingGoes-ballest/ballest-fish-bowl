"""Tile pictures drawn from a model file as the host builds it: meshes read by the plugin manager's own reader
(tools/tests/meshfile_test.cpp), shapes made into triangles here, flat-shaded and drawn farthest first. Groups are
drawn at rest (no spin, swing or bob); animated meshes at 0.2 s into their animation."""
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODELS = TOOL = None

SIZE, SCALE = 256, 4


def turn(rot, p):
    """The engine's rotator turn (FRotationMatrix), as models.cpp applies "rot"."""
    sp, cp = math.sin(math.radians(rot[0])), math.cos(math.radians(rot[0]))
    sy, cy = math.sin(math.radians(rot[1])), math.cos(math.radians(rot[1]))
    sr, cr = math.sin(math.radians(rot[2])), math.cos(math.radians(rot[2]))
    x = np.array([cp * cy, cp * sy, sp])
    y = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    z = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return p[:, 0:1] * x + p[:, 1:2] * y + p[:, 2:3] * z


def load_obj(path):
    v, vt, faces, mat, materials = [], [], [], None, {}
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "mtllib":
            for ml in open(path.parent / " ".join(p[1:])):
                q = ml.split()
                if q and q[0] == "newmtl":
                    cur = materials.setdefault(q[1], {"Kd": (0.8, 0.8, 0.8), "tex": None})
                elif q and q[0] == "Kd":
                    cur["Kd"] = tuple(float(x) for x in q[1:4])
                elif q and q[0] == "map_Kd":
                    cur["tex"] = np.asarray(Image.open(ml.split(None, 1)[1].strip()).convert("RGB"), float) / 255
        elif p[0] == "v":
            v.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == "usemtl":
            mat = p[1]
        elif p[0] == "f":
            faces.append(([int(c.split("/")[0]) - 1 for c in p[1:4]], [int(c.split("/")[1]) - 1 for c in p[1:4]], mat))
    return np.array(v), np.array(vt) if vt else np.zeros((len(v), 2)), faces, materials


def mesh_tris(line, work):
    """A mesh line's triangles and colours, posed and placed as models.cpp does."""
    words = line.split()
    opts = dict(w.split("=", 1) for w in words[2:] if "=" in w)
    num = lambda k, d: [float(x) for x in opts[k].split(",")] if k in opts else d
    size = float(opts.get("size", 0))
    anim = opts.get("anim", "")
    obj = work / (Path(words[1]).stem + ".obj")
    subprocess.run([str(TOOL), str(MODELS / words[1]), anim, "0.2", str(obj)], check=True, capture_output=True)
    rest = work / (Path(words[1]).stem + "_rest.obj")
    subprocess.run([str(TOOL), str(MODELS / words[1]), "", "0", str(rest)], check=True, capture_output=True)
    v, vt, faces, mats = load_obj(obj)
    rv = load_obj(rest)[0]
    k = size / (rv.max(0) - rv.min(0)).max() if size else 1
    scale = np.array(num("scale", [1, 1, 1]))
    rot = num("rot", [0, 0, 0])
    at = np.array(num("at", [0, 0, 0]))
    placed_rest = turn(rot, rv * k * scale)
    lo, hi = placed_rest.min(0), placed_rest.max(0)
    offset = at - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    p = turn(rot, v * k * scale) + offset
    out = []
    for idx, tidx, mat in faces:
        m = mats.get(mat, {"Kd": (0.8, 0.8, 0.8), "tex": None})
        if m["tex"] is not None:
            u, w = vt[tidx].mean(0)
            tex = m["tex"]
            colour = tex[min(int(w % 1 * tex.shape[0]), tex.shape[0] - 1), min(int(u % 1 * tex.shape[1]), tex.shape[1] - 1)] ** 2.2
        else:
            colour = np.array(m["Kd"])
        out.append((p[idx], colour, "plastic", 1.0))
    return out


def shape_tris(words, mats):
    """A few of the model format's shapes as triangles: sphere, box, cylinder, cone, disc, bowl."""
    kind, mat = words[0], words[1]
    o = dict(w.split("=", 1) for w in words[2:] if "=" in w)
    num = lambda k, d: [float(x) for x in o[k].split(",")] if k in o else d
    at, rot, sc = np.array(num("at", [0, 0, 0])), num("rot", [0, 0, 0]), np.array(num("scale", [1, 1, 1]))
    kind_m, colour, alpha = mats[mat]
    seg = 28
    tris = []
    ring = lambda r, z: [np.array([r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), z]) for i in range(seg)]
    if kind in ("sphere", "bowl"):
        r = float(o["r"])
        rows = 12
        top = 0.5 if kind == "bowl" else 0.0
        for j in range(rows):
            t0, t1 = top * math.pi + (1 - top) * math.pi * j / rows, top * math.pi + (1 - top) * math.pi * (j + 1) / rows
            a, b = ring(r * math.sin(t0), r * math.cos(t0)), ring(r * math.sin(t1), r * math.cos(t1))
            for i in range(seg):
                tris += [[a[i], b[i], a[(i + 1) % seg]], [a[(i + 1) % seg], b[i], b[(i + 1) % seg]]]
    elif kind in ("cylinder", "cone"):
        r = float(o["r"])
        top = float(o.get("top", r)) if kind == "cone" else r
        h = float(o["h"])
        a, b = ring(r, 0), ring(top, h)
        for i in range(seg):
            tris += [[a[i], a[(i + 1) % seg], b[i]], [b[i], a[(i + 1) % seg], b[(i + 1) % seg]]]
            tris += [[np.zeros(3), a[i], a[(i + 1) % seg]], [np.array([0, 0, h]), b[i], b[(i + 1) % seg]]]
    elif kind == "capsule":
        r, length = float(o["r"]), float(o["len"])
        a, b = ring(r, r), ring(r, r + length)
        for i in range(seg):
            tris += [[a[i], a[(i + 1) % seg], b[i]], [b[i], a[(i + 1) % seg], b[(i + 1) % seg]]]
        for cz, sign in ((r, -1), (r + length, 1)):
            rows = 6
            for j in range(rows):
                t0, t1 = math.pi / 2 * j / rows, math.pi / 2 * (j + 1) / rows
                c0 = ring(r * math.cos(t0), cz + sign * r * math.sin(t0))
                c1 = ring(r * math.cos(t1), cz + sign * r * math.sin(t1))
                for i in range(seg):
                    tris += [[c0[i], c1[i], c0[(i + 1) % seg]], [c0[(i + 1) % seg], c1[i], c1[(i + 1) % seg]]]
    elif kind == "disc":
        r = float(o["r"])
        a = ring(r, 0)
        tris += [[np.zeros(3), a[i], a[(i + 1) % seg]] for i in range(seg)]
    elif kind == "box":
        s = np.array(num("size", [1, 1, 1])) / 2
        c = [np.array([x, y, z]) * s for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        for f in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)):
            tris += [[c[f[0]], c[f[1]], c[f[2]]], [c[f[0]], c[f[2]], c[f[3]]]]
    out = []
    for t in tris:
        p = turn(rot, np.array(t) * sc) + at
        out.append((p, np.array(colour), kind_m, alpha))
    return out


def model_tris(text, work):
    mats, out = {}, []
    for raw in text.splitlines():
        s = raw.split("# ")[0].strip()
        if not s or s.startswith("#"):
            continue
        w = s.split()
        if w[0] == "material":
            opts = dict(x.split("=", 1) for x in w[3:] if "=" in x)
            hexes = [x for x in w[3:] if x.startswith("#")]
            h = hexes[0][1:] if hexes else "ffffff"
            colour = [(int(h[i:i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4)]
            mats[w[1]] = (w[2], colour, float(opts.get("opacity", 1)) if w[2] == "glass" else 1.0)
        elif w[0] == "mesh":
            out += mesh_tris(s, work)
        elif w[0] in ("sphere", "bowl", "cylinder", "cone", "disc", "box", "capsule"):
            out += shape_tris(w, mats)
    return out


def srgb(c):
    return np.clip(np.power(np.clip(c, 0, 1), 1 / 2.2), 0, 1)


def draw(tris, out, ball=None, azimuth=30, elevation=15, extent=58, centre_z=0):
    """Flat-shaded triangles, farthest first. `ball`: None, "glass" (a pale disc behind and a rim in front) or a
    texture array for the ball itself."""
    az, el = math.radians(azimuth), math.radians(elevation)
    view = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    f = -view
    right = np.array([-f[1], f[0], 0.0])
    right /= np.linalg.norm(right)
    up = np.array([0.0, 0.0, 1.0]) - f[2] * f
    up /= np.linalg.norm(up)
    light = view * 0.6 + np.array([0.0, 0.0, 0.8])
    light /= np.linalg.norm(light)
    px = SIZE * SCALE
    to_screen = lambda p: np.stack([px / 2 + (p @ right) / extent * px / 2, px / 2 - ((p @ up) - centre_z) / extent * px / 2], -1)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    r = 50 / extent * px / 2
    c = to_screen(np.zeros((1, 3)))[0]
    if isinstance(ball, np.ndarray):
        img.alpha_composite(textured_ball(ball, view, right, up, int(2 * r)), (int(c[0] - r), int(c[1] - r)))
    elif isinstance(ball, str):
        ImageDraw.Draw(img, "RGBA").ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(150, 205, 240, 110))
    layer = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    depth = [t[0].mean(0) @ view for t in tris]
    for n in np.argsort(depth):
        p, colour, kind, alpha = tris[n]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        if kind == "glow":
            rgb = srgb(np.array(colour) / max(max(colour), 1e-3))
        else:
            rgb = srgb(np.array(colour) * (0.45 + 0.55 * abs(normal @ light) / length))
        a = 255 if kind != "glass" else int(255 * min(1.0, alpha * 1.3))
        d.polygon([tuple(q) for q in to_screen(p)], fill=tuple(int(v * 255) for v in rgb) + (a,))
    img.alpha_composite(layer)
    if isinstance(ball, str):
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=(230, 240, 250, 200), width=SCALE * 5)
        d.arc([c[0] - r * 0.8, c[1] - r * 0.8, c[0] + r * 0.8, c[1] + r * 0.8], 200, 250, fill=(255, 255, 255, 170), width=SCALE * 6)
    img.resize((SIZE, SIZE), Image.LANCZOS).save(out)
    


def textured_ball(tex, view, right, up, n):
    """The ball's own sphere with a texture, as seen along `view`, lit from above, with a shine."""
    y, x = np.mgrid[0:n, 0:n]
    sx, sy = (x + 0.5) / n * 2 - 1, 1 - (y + 0.5) / n * 2
    r2 = sx * sx + sy * sy
    inside = r2 < 1
    sz = np.sqrt(np.clip(1 - r2, 0, 1))
    p = sx[..., None] * right + sy[..., None] * up + sz[..., None] * view
    lon = np.arctan2(p[..., 1], p[..., 0])
    lat = np.arcsin(np.clip(p[..., 2], -1, 1))
    u = ((lon / (2 * math.pi) + 0.5) * tex.shape[1]).astype(int) % tex.shape[1]
    v = np.clip(((0.5 - lat / math.pi) * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
    col = tex[v, u] ** 2.2
    light = view * 0.6 + np.array([0.0, 0.0, 0.8])
    light /= np.linalg.norm(light)
    col = col * (0.45 + 0.55 * np.clip(p @ light, 0, 1))[..., None]
    col = srgb(col)
    shine = np.exp(-(((sx + 0.42) ** 2 + (sy - 0.45) ** 2) / 0.02))
    col = col + (1 - col) * (0.7 * shine)[..., None]
    rgba = np.dstack([col, inside.astype(float)])
    return Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")




def draw_model(text, models, tool, work, out):
    global MODELS, TOOL
    MODELS, TOOL = Path(models), Path(tool)
    # the water sphere is left out: drawn flat it would hide everything; the glass disc behind is tinted instead
    text = chr(10).join(line for line in text.splitlines() if not line.startswith("sphere water"))
    draw(model_tris(text, work), out, ball="glass", azimuth=200, elevation=16)
