"""The GPU half of the show: the NPU computes, the Radeon iGPU draws it (moderngl, OpenGL 3.3).

Per frame the host uploads the raw state the NPU sent back (cells, chemical, iteration counts) as a
texture; shaders do everything else:
  - Life & co: the fading trails (a heat texture ping-pongs on the GPU), palettes, glowing cells;
  - Gray-Scott: the chemical as a height map, lit like wet tissue (normals, diffuse, specular, rim);
  - Mandelbrot: a cycling cosine palette;
then bloom (bright pass, blurred at quarter size) and tone mapping, and the HUD text on top.

World textures keep the rows in data order (row 0 at GL y = 0); only the passes that draw on the
screen flip them.
"""

import numpy as np

import moderngl

VS = """
#version 330
in vec2 pos;
out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }
"""

HEAT = """
#version 330
uniform sampler2D cells;
uniform sampler2D prev;
uniform float fade;
in vec2 uv;
out float heat;
void main() {
    float s = floor(texture(cells, uv).r * 255.0 + 0.5);
    heat = max(texture(prev, uv).r * fade, s == 1.0 ? 1.0 : 0.0);
}
"""

CA = """
#version 330
uniform sampler2D cells;
uniform sampler2D heat;
uniform sampler2D lut;
uniform int rule;           // 0 life, 1 brain, 2 wire
in vec2 uv;
out vec4 color;
void main() {
    vec2 p = vec2(uv.x, 1.0 - uv.y);
    float s = floor(texture(cells, p).r * 255.0 + 0.5);
    float h = texture(heat, p).r;
    vec3 c;
    if (rule == 2) {
        c = vec3(0.0);
        if (s == 3.0) c = vec3(0.16, 0.08, 0.03) + h * vec3(0.6, 0.85, 1.2);
        if (s == 2.0) c = vec3(1.4, 0.35, 0.15);
        if (s == 1.0) c = vec3(0.7, 1.4, 2.2);
    } else {
        c = texture(lut, vec2(h, 0.5)).rgb * (0.4 + 0.9 * h);
        if (rule == 1 && s == 2.0) c = vec3(0.3, 0.5, 1.6);
        if (s == 1.0) c = rule == 1 ? vec3(1.8, 2.0, 2.2) : vec3(2.0, 1.9, 1.1);
    }
    color = vec4(c, 1.0);
}
"""

GRAY = """
#version 330
uniform sampler2D chem;     // V, 1.0 = 4096
uniform sampler2D lut;
uniform vec2 texel;
in vec2 uv;
out vec4 color;
float hgt(vec2 p) { return smoothstep(0.0, 0.45, texture(chem, p).r); }
void main() {
    vec2 p = vec2(uv.x, 1.0 - uv.y);
    float h = hgt(p);
    float dx = hgt(p + vec2(texel.x, 0.0)) - hgt(p - vec2(texel.x, 0.0));
    float dy = hgt(p - vec2(0.0, texel.y)) - hgt(p + vec2(0.0, texel.y));
    vec3 n = normalize(vec3(-dx * 3.0, -dy * 3.0, 1.0));
    vec3 l = normalize(vec3(-0.45, 0.55, 0.7));
    vec3 v = vec3(0.0, 0.0, 1.0);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(reflect(-l, n), v), 0.0), 48.0);
    float rim = pow(1.0 - n.z, 2.0);
    vec3 base = texture(lut, vec2(h, 0.5)).rgb;
    vec3 c = base * (0.18 + 0.95 * diff) + vec3(1.0, 0.97, 0.9) * spec * (0.3 + 1.6 * h) + base * rim * 1.5;
    color = vec4(c, 1.0);
}
"""

MANDEL = """
#version 330
uniform sampler2D counts;
uniform float maxit;
uniform float t;
in vec2 uv;
out vec4 color;
void main() {
    vec2 p = vec2(uv.x, 1.0 - uv.y);
    float n = texture(counts, p).r;
    if (n >= maxit - 0.5) { color = vec4(0.0, 0.0, 0.0, 1.0); return; }
    float x = log(n + 1.0) * 0.16 + t;
    vec3 c = 0.5 + 0.5 * cos(6.2831853 * (x + vec3(0.0, 0.1, 0.2)));
    c = pow(c, vec3(1.6)) * (0.25 + 1.3 * smoothstep(0.0, 4.0, log(n + 1.0)));
    color = vec4(c, 1.0);
}
"""

BRIGHT = """
#version 330
uniform sampler2D src;
uniform float threshold;
in vec2 uv;
out vec4 color;
void main() { color = vec4(max(texture(src, uv).rgb - threshold, 0.0), 1.0); }
"""

BLUR = """
#version 330
uniform sampler2D src;
uniform vec2 dir;
in vec2 uv;
out vec4 color;
void main() {
    float w[5] = float[](0.227, 0.194, 0.121, 0.054, 0.016);
    vec3 c = texture(src, uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(src, uv + dir * float(i)).rgb * w[i];
        c += texture(src, uv - dir * float(i)).rgb * w[i];
    }
    color = vec4(c, 1.0);
}
"""

COMPOSE = """
#version 330
uniform sampler2D scene;
uniform sampler2D bloom;
uniform float strength;
in vec2 uv;
out vec4 color;
void main() {
    vec3 c = texture(scene, uv).rgb + texture(bloom, uv).rgb * strength;
    c = vec3(1.0) - exp(-c * 1.1);            // tone mapping
    color = vec4(pow(c, vec3(1.0 / 1.15)), 1.0);
}
"""

TEXT = """
#version 330
uniform sampler2D img;
in vec2 uv;
out vec4 color;
void main() { color = texture(img, vec2(uv.x, 1.0 - uv.y)); }
"""


class Renderer:
    def __init__(self, ctx, size):
        self.ctx = ctx
        self.size = size
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.progs, self.vaos = {}, {}
        for name, fs in (("heat", HEAT), ("ca", CA), ("gray", GRAY), ("mandel", MANDEL), ("bright", BRIGHT),
                         ("blur", BLUR), ("compose", COMPOSE), ("text", TEXT)):
            prog = ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.progs[name] = prog
            self.vaos[name] = ctx.vertex_array(prog, [(quad, "2f", "pos")])
        w, h = size
        self.scene = ctx.texture((w, h), 4, dtype="f2")
        self.scene_fb = ctx.framebuffer([self.scene])
        bw, bh = max(1, w // 4), max(1, h // 4)
        self.blur = [ctx.texture((bw, bh), 4, dtype="f2") for _ in range(2)]
        for t in self.blur:
            t.repeat_x = t.repeat_y = False
        self.blur_fb = [ctx.framebuffer([t]) for t in self.blur]
        self.world = {}  # textures sized to an effect's world, made on first use
        self.luts = {}

    # -- textures ------------------------------------------------------------------------------

    def _tex(self, key, shape, comps, dtype, nearest=True):
        h, w = shape
        t = self.world.get(key)
        if t is None or t.size != (w, h):
            if t is not None:
                t.release()
            t = self.ctx.texture((w, h), comps, dtype=dtype)
            f = moderngl.NEAREST if nearest else moderngl.LINEAR
            t.filter = (f, f)
            self.world[key] = t
        return t

    def lut(self, name, rgb):
        """A 256 x 1 palette texture from a (256, 3) uint8 array."""
        if name not in self.luts:
            t = self.ctx.texture((256, 1), 3, data=np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            self.luts[name] = t
        return self.luts[name]

    def _draw(self, name, target, viewport, **uniforms):
        target.use()
        self.ctx.viewport = viewport
        prog = self.progs[name]
        unit = 0
        for k, v in uniforms.items():
            u = prog.get(k, None)
            if u is None:  # the compiler dropped an unused uniform
                continue
            if isinstance(v, moderngl.Texture):
                v.use(unit)
                u.value = unit
                unit += 1
            else:
                u.value = v
        self.vaos[name].render(moderngl.TRIANGLE_STRIP)

    def _viewport(self, rect):
        """pygame rect (top-left origin) -> GL viewport (bottom-left origin)."""
        x, y, w, h = rect
        return (x, self.size[1] - y - h, w, h)

    # -- the effects ---------------------------------------------------------------------------

    def begin(self):
        self.scene_fb.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.scene_fb.clear(0.0, 0.0, 0.0, 1.0)

    def ca(self, cells, rule, lut_rgb, fade, rect, key="ca"):
        h, w = cells.shape
        tex = self._tex(key + ":cells", (h, w), 1, "f1")
        tex.write(np.ascontiguousarray(cells, dtype=np.uint8).tobytes())
        heat = [self._tex(key + f":heat{i}", (h, w), 1, "f2") for i in range(2)]
        flip = self.world.get(key + ":flip", 0)
        src, dst = heat[flip], heat[1 - flip]
        fb = self.world.get(key + ":fb", {})
        if dst.glo not in fb:
            fb[dst.glo] = self.ctx.framebuffer([dst])
            self.world[key + ":fb"] = fb
        self._draw("heat", fb[dst.glo], (0, 0, w, h), cells=tex, prev=src, fade=fade)
        self.world[key + ":flip"] = 1 - flip
        self._draw("ca", self.scene_fb, self._viewport(rect), cells=tex, heat=dst,
                   lut=self.lut(key + ":" + str(rule), lut_rgb), rule={"life": 0, "brain": 1, "wire": 2}[rule])

    def clear_heat(self, key="ca"):
        for i in range(2):
            t = self.world.get(key + f":heat{i}")
            if t is not None:
                t.write(np.zeros(t.size[0] * t.size[1], np.float16).tobytes())

    def gray(self, v, lut_name, lut_rgb, rect):
        h, w = v.shape
        tex = self._tex("gray:v", (h, w), 1, "f4", nearest=False)
        tex.write((v.astype(np.float32) / 4096.0).tobytes())
        self._draw("gray", self.scene_fb, self._viewport(rect), chem=tex, lut=self.lut(lut_name, lut_rgb),
                   texel=(1.0 / w, 1.0 / h))

    def mandel(self, counts, maxit, t, rect):
        h, w = counts.shape
        tex = self._tex("mandel:n", (h, w), 1, "f4", nearest=False)
        tex.write(counts.astype(np.float32).tobytes())
        self._draw("mandel", self.scene_fb, self._viewport(rect), counts=tex, maxit=float(maxit), t=t)

    def finish(self, target, strength=0.8, threshold=0.9):
        """Bloom, tone mapping, onto `target` (ctx.screen, or a framebuffer when testing)."""
        bw, bh = self.blur[0].size
        self._draw("bright", self.blur_fb[0], (0, 0, bw, bh), src=self.scene, threshold=threshold)
        for _ in range(2):
            self._draw("blur", self.blur_fb[1], (0, 0, bw, bh), src=self.blur[0], dir=(1.5 / bw, 0.0))
            self._draw("blur", self.blur_fb[0], (0, 0, bw, bh), src=self.blur[1], dir=(0.0, 1.5 / bh))
        self._draw("compose", target, (0, 0, *self.size), scene=self.scene, bloom=self.blur[0], strength=strength)

    def text(self, target, surface, x, y):
        """Draw a pygame surface with alpha (the HUD) at pixel (x, y), top-left origin."""
        import pygame
        w, h = surface.get_size()
        t = self.ctx.texture((w, h), 4, data=pygame.image.tobytes(surface, "RGBA"))
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._draw("text", target, self._viewport((x, y, w, h)), img=t)
        self.ctx.disable(moderngl.BLEND)
        t.release()
