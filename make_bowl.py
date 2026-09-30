"""Makes the Fish Bowl's files:
  - models/sand.glb: the sand floor, the bottom of a sphere just inside the ball with a gently rippled top
  - models/bubbles.glb: a column of bubbles rising, wobbling and shrinking away at the top (an animation that loops)
  - models/coral_*.glb: single corals taken from MiniPoly's Coral Reef Set (sources/coral_reef.glb, eight in a row)
  - models/*.glb: the other downloaded models, copied from sources/
  - models/fish_bowl.txt: the model, in the plugin manager's model format
  - preview.png and icon.png: the tile picture, drawn from the model as the host builds it (meshes through the plugin
    manager's own reader, tools/tests/meshfile_test.cpp)
    python make_bowl.py <path to ballest-plugin-manager>
Needs Python with NumPy and Pillow."""
import json
import math
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

import preview

HERE = Path(__file__).resolve().parent
SOURCES, MODELS = HERE / "sources", HERE / "models"


# --- glTF files --------------------------------------------------------------------------------------------------------
def pad4(b, fill=b"\0"):
    return b + fill * (-len(b) % 4)


class Glb:
    """A small glTF 2.0 writer: meshes of positions, normals and indices, plain materials, nodes and animations."""

    def __init__(self):
        self.g = {"asset": {"version": "2.0", "generator": "ballest-fish-bowl make_bowl.py"}, "scene": 0, "scenes": [{"nodes": []}],
                  "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": [], "buffers": [{"byteLength": 0}]}
        self.data = bytearray()

    def view(self, raw, target=None):
        v = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            v["target"] = target
        self.data += pad4(raw)
        self.g["bufferViews"].append(v)
        return len(self.g["bufferViews"]) - 1

    def accessor(self, array, kind, target=None, minmax=False):
        array = np.ascontiguousarray(array)
        comp = {np.dtype("float32"): 5126, np.dtype("uint32"): 5125}[array.dtype]
        a = {"bufferView": self.view(array.tobytes(), target), "componentType": comp, "count": len(array), "type": kind}
        if minmax:
            a["min"], a["max"] = array.min(0).tolist(), array.max(0).tolist()
        self.g["accessors"].append(a)
        return len(self.g["accessors"]) - 1

    def material(self, name, rgba, rough=0.5):
        self.g["materials"].append({"name": name, "pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0,
                                                                           "roughnessFactor": rough},
                                    **({"alphaMode": "BLEND"} if rgba[3] < 1 else {})})
        return len(self.g["materials"]) - 1

    def mesh(self, pos, nrm, idx, material):
        prim = {"attributes": {"POSITION": self.accessor(pos.astype(np.float32), "VEC3", 34962, True),
                               "NORMAL": self.accessor(nrm.astype(np.float32), "VEC3", 34962)},
                "indices": self.accessor(idx.astype(np.uint32), "SCALAR", 34963), "material": material}
        self.g["meshes"].append({"primitives": [prim]})
        return len(self.g["meshes"]) - 1

    def node(self, **kw):
        self.g["nodes"].append(kw)
        index = len(self.g["nodes"]) - 1
        self.g["scenes"][0]["nodes"].append(index)
        return index

    def animation(self, name, channels):
        """channels: (node, "translation"|"scale", times, values), each STEP (the host bakes its frames from it)."""
        anim = {"name": name, "samplers": [], "channels": []}
        for node, path, times, values in channels:
            sampler = {"input": self.accessor(np.array(times, np.float32), "SCALAR", minmax=True),
                       "output": self.accessor(np.array(values, np.float32), "VEC3"), "interpolation": "STEP"}
            anim["samplers"].append(sampler)
            anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": node, "path": path}})
        self.g.setdefault("animations", []).append(anim)

    def save(self, path):
        self.g["buffers"][0]["byteLength"] = len(self.data)
        js = pad4(json.dumps(self.g, separators=(",", ":")).encode(), b" ")
        binary = bytes(self.data)
        path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)) +
                         struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary)


def uv_sphere(r, seg=16, rings=10):
    pos, idx = [], []
    for j in range(rings + 1):
        t = math.pi * j / rings
        for i in range(seg + 1):
            p = 2 * math.pi * i / seg
            pos.append((r * math.sin(t) * math.cos(p), r * math.cos(t), -r * math.sin(t) * math.sin(p)))
    for j in range(rings):
        for i in range(seg):
            a, b = j * (seg + 1) + i, (j + 1) * (seg + 1) + i
            idx += [a, b, a + 1, a + 1, b, b + 1]
    pos = np.array(pos)
    return pos, pos / r, np.array(idx)


# glTF is Y-up in metres; the host turns it into the game's Z-up centimetres (x, y, z) -> (100 z, -100 x, 100 y). The
# files below are written in metres, so with no size= they come out at exactly these sizes.
SAND_TOP, SAND_R = -0.20, 0.48          # the sand's top 20 cm below the ball's middle, inside a 48 cm sphere


def sand():
    """The part of a 48 cm sphere below 26 cm under the middle: a curved bottom and a flat top with soft ripples."""
    g = Glb()
    rim = math.sqrt(SAND_R ** 2 - SAND_TOP ** 2)
    seg, rings = 48, 10
    pos, idx = [], []
    # the curved bottom: rings from the rim down to the pole
    t0 = math.acos(-SAND_TOP / SAND_R)               # angle from the bottom pole to the rim
    for j in range(rings + 1):
        t = t0 * (1 - j / rings)
        for i in range(seg + 1):
            p = 2 * math.pi * i / seg
            pos.append((SAND_R * math.sin(t) * math.cos(p), -SAND_R * math.cos(t), SAND_R * math.sin(t) * math.sin(p)))
    for j in range(rings):
        for i in range(seg):
            a, b = j * (seg + 1) + i, (j + 1) * (seg + 1) + i
            idx += [a, a + 1, b, a + 1, b + 1, b]
    bottom = np.array(pos)
    bottom_n = bottom / SAND_R
    # the top: a disc of rings, rippled a little (a centimetre either way), meeting the rim exactly
    top, top_i = [(0.0, SAND_TOP + 0.004, 0.0)], []
    for j in range(1, rings + 1):
        r = rim * j / rings
        for i in range(seg):
            p = 2 * math.pi * i / seg
            x, z = r * math.cos(p), r * math.sin(p)
            ripple = 0.006 * math.sin(x * 45) * math.cos(z * 38) * (1 - (j / rings) ** 4)
            top.append((x, SAND_TOP + ripple, z))
    for i in range(seg):
        top_i += [0, 1 + (i + 1) % seg, 1 + i]
    for j in range(1, rings):
        for i in range(seg):
            a, b = 1 + (j - 1) * seg + i, 1 + j * seg + i
            a2, b2 = 1 + (j - 1) * seg + (i + 1) % seg, 1 + j * seg + (i + 1) % seg
            top_i += [a, a2, b, a2, b2, b]
    top = np.array(top)
    top_n = np.tile([0.0, 1.0, 0.0], (len(top), 1))
    sand_mat = g.material("sand", (0.98, 0.86, 0.58, 1), rough=0.95)
    g.node(mesh=g.mesh(bottom, bottom_n, np.array(idx), sand_mat), name="sand_bottom")
    g.node(mesh=g.mesh(top, top_n, np.array(top_i), sand_mat), name="sand_top")
    g.save(MODELS / "sand.glb")


def bubbles(count=6, height=0.34, period=3.0, steps=48):
    """Bubbles rising `height` from their start in `period` seconds, one after another, wobbling side to side, growing
    as they leave and shrinking to nothing at the top before starting again. Each is stepped (the host bakes the
    animation into frames), so the jump back to the start is never drawn as a bubble sinking."""
    g = Glb()
    mat = g.material("bubble", (0.85, 0.95, 1.0, 0.45), rough=0.05)
    pos, nrm, idx = uv_sphere(0.012)
    mesh = g.mesh(pos, nrm, idx, mat)
    times = [period * s / steps for s in range(steps + 1)]
    channels = []
    for b in range(count):
        size = 0.7 + 0.5 * ((b * 37) % 10) / 10
        node = g.node(mesh=mesh, name=f"bubble{b}", scale=[size] * 3)
        moves, sizes = [], []
        for t in times:
            f = (t / period + b / count) % 1.0
            wobble = 0.006 * math.sin(f * 14 + b)
            moves.append([wobble, f * height, 0.004 * math.cos(f * 11 + b)])
            grow = min(1.0, f / 0.08) * min(1.0, (1 - f) / 0.12) * size * (1 + 0.5 * f)
            sizes.append([grow] * 3)
        channels += [(node, "translation", times, moves), (node, "scale", times, sizes)]
    g.animation("rise", channels)
    g.save(MODELS / "bubbles.glb")


def split_coral(keep):
    """Single corals from the Coral Reef Set: each written as a file of its own with only that coral's mesh, its
    material and the images it uses (the set is eight corals in a row)."""
    b = (SOURCES / "coral_reef.glb").read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    src = json.loads(b[20:20 + n])
    binary = b[20 + n + 8:]
    for mesh_index, name in keep:
        node = next(nd for nd in src["nodes"] if nd.get("mesh") == mesh_index)
        g = Glb()
        mesh = json.loads(json.dumps(src["meshes"][node["mesh"]]))
        remap_acc, remap_mat, remap_img, remap_tex = {}, {}, {}, {}

        def copy_view(i):
            v = src["bufferViews"][i]
            start = v.get("byteOffset", 0)
            return g.view(binary[start:start + v["byteLength"]], v.get("target"))

        def copy_accessor(i):
            if i not in remap_acc:
                a = dict(src["accessors"][i])
                a["bufferView"] = copy_view(a["bufferView"])
                g.g["accessors"].append(a)
                remap_acc[i] = len(g.g["accessors"]) - 1
            return remap_acc[i]

        def copy_texture(t):
            if t not in remap_tex:
                tex = dict(src["textures"][t])
                img = tex["source"]
                if img not in remap_img:
                    im = dict(src["images"][img])
                    im["bufferView"] = copy_view(im["bufferView"])
                    g.g.setdefault("images", []).append(im)
                    remap_img[img] = len(g.g["images"]) - 1
                tex["source"] = remap_img[img]
                tex.pop("sampler", None)
                g.g.setdefault("textures", []).append(tex)
                remap_tex[t] = len(g.g["textures"]) - 1
            return remap_tex[t]

        for prim in mesh["primitives"]:
            prim["attributes"] = {k: copy_accessor(v) for k, v in prim["attributes"].items()}
            if "indices" in prim:
                prim["indices"] = copy_accessor(prim["indices"])
            m = prim.get("material")
            if m is not None:
                if m not in remap_mat:
                    mat = json.loads(json.dumps(src["materials"][m]))
                    for key in ("baseColorTexture", "metallicRoughnessTexture"):
                        if key in mat.get("pbrMetallicRoughness", {}):
                            mat["pbrMetallicRoughness"][key]["index"] = copy_texture(mat["pbrMetallicRoughness"][key]["index"])
                    # plain coloured coral: the set's metallic and emissive maps (metallic 1, emissive white) would make
                    # the host draw it as metal and glowing
                    for key in ("normalTexture", "occlusionTexture", "emissiveTexture", "emissiveFactor", "extras"):
                        mat.pop(key, None)
                    pbr = mat["pbrMetallicRoughness"]
                    pbr.pop("metallicRoughnessTexture", None)
                    pbr["metallicFactor"], pbr["roughnessFactor"] = 0, 0.7
                    g.g["materials"].append(mat)
                    remap_mat[m] = len(g.g["materials"]) - 1
                prim["material"] = remap_mat[m]
        g.g["meshes"].append(mesh)
        kept = {k: node[k] for k in ("rotation", "scale") if k in node}
        g.node(mesh=0, name=name, **kept)
        g.save(MODELS / f"{name}.glb")


# --- the model --------------------------------------------------------------------------------------------------------
def seaweed(name, x, y, height, colour_mat, phase):
    """A strand of seaweed standing on the sand: four flat leaf-like segments leaning one way then the other (a wavy
    kelp), each swaying on the one below, a little more toward the tip."""
    lines, z, parent = [], -20.0, "tank"
    lengths = [height * 0.3, height * 0.27, height * 0.23, height * 0.2]
    widths = [1.6, 1.45, 1.25, 1.0]
    dx = 0.0
    for i, (length, w) in enumerate(zip(lengths, widths)):
        group = f"{name}{i}"
        lean = 12 if i % 2 == 0 else -12
        lines.append(f"group {group} on={parent} pivot={x + dx:.1f},{y},{z:.1f} swing=x angle={6 + 3 * i} phase={phase + 40 * i}")
        lines.append(f"capsule {colour_mat} r={w} len={length:.1f} at={x + dx:.1f},{y},{z:.1f} rot={lean},0,0 scale=2.4,0.4,1")
        # the next segment starts where this one ends, leaning `lean` degrees from upright about the y axis
        top = length + 2 * w - 1.2
        dx += top * math.sin(math.radians(lean))
        z += top * math.cos(math.radians(lean))
        parent = group
    return lines


def model_text():
    lines = [
        "# The Fish Bowl: the ball is an aquarium. A clownfish and a yellow tang (Quaternius, CC0) swim round it, breathing",
        "# little bubbles, over a sand floor with coral (MiniPoly, CC0), a treasure chest puffing bubbles, a little castle",
        "# and rocks (Quaternius, CC0), a shell (Kenney, CC0) and swaying seaweed. Everything inside stays upright as the",
        "# ball rolls. Made by make_bowl.py.",
        "material water glass #5cc8ff opacity=0.09",
        "material weed plastic #2f9e44 rough=0.6",
        "material weed2 plastic #58c24a rough=0.6",
        "material weed3 plastic #1f7a4a rough=0.6",
        "tempo rate=0.35 calm=1 full=1",
        "",
        "# the water fills the tank",
        "group water",
        "sphere water r=48.8",
        "",
        "# the tank: upright, turned the way the ball goes",
        "group tank travel",
        "mesh sand.glb at=0,0,-48",
        "mesh ocean_chest.glb size=22 at=-15,-16,-20.5 rot=0,35,0",
        "mesh tower.glb size=32 at=18,12,-20.5 rot=0,-140,0",
        "mesh rocks.glb size=19 at=-22,13,-20.5 rot=0,70,0",
        "mesh mussel.glb size=9 at=21,-15,-20.3 rot=0,-30,0",
        "mesh coral_pink.glb size=15 at=4,22,-20.5",
        "mesh coral_red.glb size=13 at=-4,-25,-20.5 rot=0,60,0",
        "mesh coral_blue.glb size=14 at=26,0,-20.5 rot=0,20,0",
        "mesh coral_orange.glb size=12 at=-26,-2,-20.5",
        "# bubbles puffing out of the chest",
        "mesh bubbles.glb at=-15,-16,-8 anim=rise rate=0.8",
        "",
        "# seaweed, swaying",
    ]
    lines += seaweed("weedA", -6, 24, 30, "weed", 0)
    lines += seaweed("weedB", 9, -22, 24, "weed2", 120)
    lines += seaweed("weedC", -24, 6, 20, "weed3", 240)
    lines += seaweed("weedD", 26, 8, 18, "weed2", 60)
    lines += [
        "",
        "# the clownfish swims round, a little above the sand, breathing bubbles",
        "group swimA on=tank spin=z speed=40",
        "group fishA on=swimA bob=3",
        "mesh clownfish.glb size=22 at=22,0,-2 rot=0,90,0 anim=Swimming_Normal rate=1",
        "mesh bubbles.glb at=26,0,5 anim=rise rate=0.6",
        "",
        "# the yellow tang swims the other way, higher up",
        "group swimB on=tank spin=z speed=-28",
        "group fishB on=swimB bob=4 phase=150",
        "mesh yellow_tang.glb size=20 at=-16,0,12 rot=0,90,0 anim=Swimming_Normal rate=0.8",
        "mesh bubbles.glb at=-20,0,20 anim=rise rate=0.5",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    tool = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "ballest-plugin-manager") / "build" / "meshfile_test.exe"
    MODELS.mkdir(exist_ok=True)
    for name in ("clownfish", "yellow_tang", "ocean_chest", "rocks", "mussel", "tower"):
        shutil.copy(SOURCES / f"{name}.glb", MODELS / f"{name}.glb")
    sand()
    bubbles()
    split_coral([(3, "coral_pink"), (2, "coral_red"), (0, "coral_blue"), (5, "coral_orange")])
    text = model_text()
    (MODELS / "fish_bowl.txt").write_text(text, newline="\n")
    with tempfile.TemporaryDirectory() as tmp:
        preview.draw_model(text, MODELS, tool, Path(tmp), HERE / "preview.png")
    shutil.copy(HERE / "preview.png", HERE / "icon.png")
    print("wrote models, preview.png, icon.png")
