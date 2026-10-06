from manim import *

BG = "#0e1117"
GPU_C = "#76b900"
CPU_C = "#4da3ff"
RAM_C = "#8b7cf6"
HOT = "#ff7a45"
DIM = "#8b949e"
WARN = "#f5c542"

config.background_color = BG


def box(label, w=2.4, h=0.9, color=WHITE, fill=None, size=24):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, color=color, stroke_width=3)
    if fill:
        r.set_fill(fill, opacity=0.25)
    t = Text(label, font_size=size, color=color)
    t.move_to(r)
    if t.width > w - 0.2:
        t.scale_to_fit_width(w - 0.2)
    return VGroup(r, t)


def title(text):
    t = Text(text, font_size=38, weight=BOLD)
    t.scale_to_fit_width(min(t.width, 12.4))
    t.to_edge(UP, buff=0.35)
    return t


def caption(text):
    c = Text(text, font_size=24, color=DIM)
    c.scale_to_fit_width(min(c.width, 12.4))
    c.to_edge(DOWN, buff=0.3)
    return c


def swap_caption(scene, old, text):
    new = caption(text)
    if old is None:
        scene.play(FadeIn(new, shift=UP * 0.1))
    else:
        scene.play(ReplacementTransform(old, new))
    return new


class S1_Problem(Scene):
    def construct(self):
        t = title("FreeToken: frontier MoE on a gaming PC")
        self.play(Write(t))

        model = box("290B MoE model  ~ 150+ GB", w=5, h=1.0, color=RAM_C, fill=RAM_C)
        model.shift(UP * 1.6 + LEFT * 3)
        gpu = box("GPU VRAM  24 GB", w=3.2, h=1.0, color=GPU_C, fill=GPU_C)
        gpu.shift(UP * 1.6 + RIGHT * 3.8)
        self.play(FadeIn(model), FadeIn(gpu))
        cross = Cross(VGroup(model, gpu), stroke_width=0)
        no = Text("does not fit", font_size=26, color=HOT).move_to((model.get_center() + gpu.get_center()) / 2 + DOWN * 0.9)
        self.play(FadeIn(no, shift=UP * 0.1))

        cap = swap_caption(self, None, "But a MoE token only touches a few experts per layer")
        self.wait(0.5)

        grid = VGroup(*[Square(0.34, stroke_width=2, color=DIM) for _ in range(64)]).arrange_in_grid(4, 16, buff=0.08)
        grid.shift(DOWN * 0.7)
        lab = Text("one layer: 64 experts", font_size=22, color=DIM).next_to(grid, UP, buff=0.2)
        self.play(FadeIn(grid), FadeIn(lab))
        picks = [3, 17, 22, 40, 41, 58]
        self.play(*[grid[i].animate.set_fill(HOT, opacity=0.9).set_color(HOT) for i in picks])
        topk = Text("router picks top-k = 6", font_size=24, color=HOT).next_to(grid, DOWN, buff=0.2)
        self.play(FadeIn(topk))
        cap = swap_caption(self, cap, "Keep all weights in host RAM, keep only what is hot on the GPU")
        self.wait(1.5)


class S2_Pipeline(Scene):
    def construct(self):
        t = title("Request path through the engine")
        self.play(Write(t))
        names = [("HTTP API", "OpenAI / Anthropic", WHITE), ("Tokenizer", "message/", WHITE),
                 ("Scheduler", "scheduler/", WARN), ("Engine", "engine/ + models/", GPU_C),
                 ("Detokenizer", "stream back", WHITE)]
        blocks = VGroup(*[box(n, w=2.3, h=1.0, color=c, fill=c) for n, _, c in names]).arrange(RIGHT, buff=0.35)
        blocks.scale_to_fit_width(12.4).shift(UP * 1.4)
        subs = VGroup(*[Text(s, font_size=18, color=DIM).next_to(b, DOWN, buff=0.12) for b, (_, s, _) in zip(blocks, names)])
        arrows = VGroup(*[Arrow(blocks[i].get_right(), blocks[i + 1].get_left(), buff=0.05, stroke_width=3, max_tip_length_to_length_ratio=0.3) for i in range(4)])
        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.2), FadeIn(subs))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2))
        cap = swap_caption(self, None, "Separate processes talk over message queues; the GPU never waits on text work")

        tok = Dot(color=HOT, radius=0.12).move_to(blocks[0])
        self.play(FadeIn(tok))
        for b in blocks[1:3]:
            self.play(tok.animate.move_to(b), run_time=0.5)

        cap = swap_caption(self, cap, "Scheduler: radix prefix match, then chunked prefill and batched decode")
        prompt = VGroup(*[Square(0.4, stroke_width=2, color=DIM) for _ in range(16)]).arrange(RIGHT, buff=0.06).shift(DOWN * 0.9)
        self.play(FadeIn(prompt), FadeOut(tok))
        cached = VGroup(*prompt[:6])
        self.play(*[c.animate.set_fill(RAM_C, 0.8).set_color(RAM_C) for c in cached])
        l1 = Text("cached prefix: reused", font_size=20, color=RAM_C).next_to(cached, DOWN, buff=0.15)
        self.play(FadeIn(l1))
        chunk1, chunk2 = VGroup(*prompt[6:11]), VGroup(*prompt[11:16])
        for ch in (chunk1, chunk2):
            self.play(*[c.animate.set_fill(GPU_C, 0.8).set_color(GPU_C) for c in ch], run_time=0.8)
        l2 = Text("two prefill chunks", font_size=20, color=GPU_C).next_to(prompt[8], DOWN, buff=0.15).shift(RIGHT * 1.5)
        self.play(FadeIn(l2))

        decode = VGroup(*[Square(0.4, stroke_width=2, color=HOT, fill_opacity=0.0) for _ in range(5)]).arrange(RIGHT, buff=0.06)
        decode.next_to(prompt, DOWN, buff=1.0)
        dl = Text("decode: one token per step, all running requests batched", font_size=20, color=HOT).next_to(decode, DOWN, buff=0.15)
        self.play(FadeIn(dl))
        for d in decode:
            self.play(FadeIn(d.set_fill(HOT, 0.8)), run_time=0.3)
        self.wait(1.5)


class S3_ExpertCache(Scene):
    def construct(self):
        t = title("Decode: a GPU slot cache over host-RAM experts")
        self.play(Write(t))
        ram = Rectangle(width=5.2, height=3.4, color=RAM_C, stroke_width=3).shift(LEFT * 3.4 + DOWN * 0.3)
        ram_l = Text("Host RAM (pinned banks)", font_size=22, color=RAM_C).next_to(ram, UP, buff=0.1)
        gpu = Rectangle(width=5.2, height=3.4, color=GPU_C, stroke_width=3).shift(RIGHT * 3.4 + DOWN * 0.3)
        gpu_l = Text("GPU slot cache (LRU)", font_size=22, color=GPU_C).next_to(gpu, UP, buff=0.1)
        self.play(FadeIn(ram), FadeIn(ram_l), FadeIn(gpu), FadeIn(gpu_l))

        experts = VGroup(*[Square(0.42, stroke_width=2, color=RAM_C, fill_opacity=0.3) for _ in range(40)]).arrange_in_grid(5, 8, buff=0.1).move_to(ram)
        slots = VGroup(*[Square(0.42, stroke_width=2, color=GPU_C) for _ in range(16)]).arrange_in_grid(2, 8, buff=0.1).move_to(gpu).shift(UP * 0.5)
        self.play(FadeIn(experts), FadeIn(slots))
        sl = Text("slot_for_id: expert -> slot", font_size=20, color=DIM).next_to(slots, DOWN, buff=0.3)
        self.play(FadeIn(sl))

        cap = swap_caption(self, None, "Router output -> ensure_experts -> hits stay put, misses get a slot")
        resident = {}
        free = list(range(16))
        order = []

        def load(ei, si):
            src = experts[ei]
            c = src.copy().set_fill(GPU_C, 0.9).set_color(GPU_C)
            self.play(c.animate.move_to(slots[si]), run_time=0.45)
            resident[ei] = (si, c)
            order.append(ei)

        for step, routed in enumerate([[0, 5, 9, 12], [0, 5, 14, 20], [0, 9, 14, 31]]):
            hdr = Text(f"decode step {step + 1}: routed {routed}", font_size=22, color=HOT).next_to(sl, DOWN, buff=0.25)
            self.play(FadeIn(hdr), *[experts[e].animate.set_color(HOT) for e in routed], run_time=0.4)
            misses = [e for e in routed if e not in resident]
            hits = [e for e in routed if e in resident]
            if hits:
                self.play(*[Indicate(resident[h][1], color=WHITE) for h in hits], run_time=0.6)
            for m in misses:
                load(m, free.pop(0))
            self.play(FadeOut(hdr), *[experts[e].animate.set_color(RAM_C) for e in routed], run_time=0.3)
        cap = swap_caption(self, cap, "Misses cross PCIe, so skew in routing (hot experts) is what makes the cache pay off")
        self.wait(1.5)


class S4_Hybrid(Scene):
    def construct(self):
        t = title("Hybrid decode: split misses by bandwidth (q*)")
        self.play(Write(t))
        cap = swap_caption(self, None, "Pure offload waits on PCIe; the CPU is idle but has fast DRAM")

        gpu = box("GPU", w=2.6, h=1.2, color=GPU_C, fill=GPU_C).shift(RIGHT * 4 + UP * 0.6)
        cpu = box("CPU", w=2.6, h=1.2, color=CPU_C, fill=CPU_C).shift(LEFT * 4 + UP * 0.6)
        ram = box("Host RAM experts", w=3.2, h=1.0, color=RAM_C, fill=RAM_C).shift(LEFT * 1.0 + DOWN * 1.2)
        self.play(FadeIn(gpu), FadeIn(cpu), FadeIn(ram))
        a_cpu = Arrow(ram.get_left(), cpu.get_bottom(), buff=0.1, color=CPU_C)
        a_pcie = Arrow(ram.get_right(), gpu.get_bottom(), buff=0.1, color=GPU_C)
        l_cpu = Text("CPU reads DRAM ~ 100 GB/s", font_size=20, color=CPU_C)
        l_pcie = Text("GPU fetch over PCIe ~ 25 GB/s", font_size=20, color=GPU_C)
        VGroup(l_cpu, l_pcie).arrange(RIGHT, buff=0.8).next_to(t, DOWN, buff=0.25)
        self.play(GrowArrow(a_cpu), GrowArrow(a_pcie), FadeIn(l_cpu), FadeIn(l_pcie))

        cap = swap_caption(self, cap, "fraction fetched = pcie / (pcie + cpu): both sides finish together")
        eq = Text("misses = 10   ->   fetch 2 over PCIe, compute 8 on CPU", font_size=24, color=WARN).shift(UP * 2.0)
        self.play(FadeIn(eq))
        misses = VGroup(*[Square(0.35, stroke_width=2, color=HOT, fill_opacity=0.8) for _ in range(10)]).arrange(RIGHT, buff=0.06).move_to(ram).shift(DOWN * 0.85)
        self.play(FadeIn(misses))
        to_gpu = misses[:2].copy()
        to_cpu = misses[2:].copy()
        self.play(
            to_gpu.animate.arrange(RIGHT, buff=0.06).next_to(gpu, DOWN, buff=0.6).set_color(GPU_C),
            to_cpu.animate.arrange(RIGHT, buff=0.06).next_to(cpu, DOWN, buff=0.6).shift(LEFT * 0.5).set_color(CPU_C),
            run_time=1.5,
        )
        bar_g = Rectangle(width=0.8, height=0.18, color=GPU_C, fill_opacity=0.9).next_to(gpu, UP, buff=0.3)
        bar_c = Rectangle(width=0.8, height=0.18, color=CPU_C, fill_opacity=0.9).next_to(cpu, UP, buff=0.3)
        self.play(GrowFromEdge(bar_g, LEFT), GrowFromEdge(bar_c, LEFT), run_time=1.5)
        merge = Text("partial outputs merge -> next layer", font_size=24, color=WHITE).shift(DOWN * 3.0)
        self.play(FadeIn(merge))
        cap2 = swap_caption(self, cap, "Handshake via GPU stream memory ops: no host callback, graph-capturable")
        self.wait(1.5)


class S5_PrefillStream(Scene):
    def construct(self):
        t = title("Prefill: stream whole layers, double-buffered")
        self.play(Write(t))
        cap = swap_caption(self, None, "Long prompts touch every expert, so copy layer i+1 while computing layer i")
        n = 6
        layers = VGroup(*[box(f"L{i}", w=1.3, h=0.8, color=RAM_C, fill=RAM_C) for i in range(n)]).arrange(RIGHT, buff=0.25).shift(UP * 1.6)
        rl = Text("Host RAM layers", font_size=20, color=RAM_C).next_to(layers, UP, buff=0.1)
        self.play(FadeIn(layers), FadeIn(rl))
        bufA = box("buffer A", w=2.4, h=1.0, color=GPU_C, fill=GPU_C).shift(LEFT * 2 + DOWN * 0.8)
        bufB = box("buffer B", w=2.4, h=1.0, color=GPU_C, fill=GPU_C).shift(RIGHT * 2 + DOWN * 0.8)
        self.play(FadeIn(bufA), FadeIn(bufB))
        gemm = Text("GEMM", font_size=22, color=HOT)
        cs = Text("copy stream", font_size=20, color=DIM).next_to(bufB, DOWN, buff=0.7).shift(LEFT * 2)
        ms = Text("compute stream", font_size=20, color=DIM).next_to(cs, DOWN, buff=0.4)
        self.play(FadeIn(cs), FadeIn(ms))

        bufs = [bufA, bufB]
        prev = None
        for i in range(n):
            b = bufs[i % 2]
            cp = layers[i].copy().set_color(GPU_C)
            self.play(cp.animate.move_to(b), run_time=0.5)
            if prev is not None:
                pass
            bar_c = Rectangle(width=1.1, height=0.22, color=GPU_C, fill_opacity=0.9).move_to(cs.get_right() + RIGHT * (0.9 + i * 0.5) + RIGHT * 0.2)
            bar_m = Rectangle(width=1.1, height=0.22, color=HOT, fill_opacity=0.9).move_to(ms.get_right() + RIGHT * (0.9 + i * 0.5) + RIGHT * 0.2 + RIGHT * 0.5)
            self.play(FadeIn(bar_c), FadeIn(bar_m), Indicate(b, color=HOT), run_time=0.6)
            self.play(FadeOut(cp), run_time=0.2)
        cap = swap_caption(self, cap, "Experts already in the slot cache are gathered on-device; only misses cross PCIe")
        self.wait(1.5)


class S6_Memory(Scene):
    def construct(self):
        t = title("One VRAM budget: experts first, KV gets the rest")
        self.play(Write(t))
        total = Rectangle(width=10, height=1.0, color=WHITE, stroke_width=3).shift(UP * 0.8)
        lab = Text("memory_ratio x free VRAM  -  weights  =  budget", font_size=22, color=DIM).next_to(total, UP, buff=0.15)
        self.play(Create(total), FadeIn(lab))
        reserve = Rectangle(width=1.6, height=1.0, color=WARN, fill_opacity=0.5, stroke_width=0).align_to(total, LEFT).align_to(total, UP)
        moe = Rectangle(width=5.4, height=1.0, color=GPU_C, fill_opacity=0.5, stroke_width=0).next_to(reserve, RIGHT, buff=0)
        kv = Rectangle(width=3.0, height=1.0, color=CPU_C, fill_opacity=0.5, stroke_width=0).next_to(moe, RIGHT, buff=0)
        self.play(FadeIn(reserve), run_time=0.6)
        self.play(FadeIn(moe), run_time=0.8)
        self.play(FadeIn(kv), run_time=0.8)
        ls = VGroup(
            Text("KV reserve", font_size=20, color=WARN).next_to(reserve, DOWN, buff=0.15),
            Text("MoE expert slots", font_size=20, color=GPU_C).next_to(moe, DOWN, buff=0.15),
            Text("KV pages", font_size=20, color=CPU_C).next_to(kv, DOWN, buff=0.15),
        )
        self.play(FadeIn(ls))
        cap = swap_caption(self, None, "plan_cache_budget: reserve KV, fill experts, give KV the remainder")
        self.wait(1)
        newmoe = Rectangle(width=3.4, height=1.0, color=GPU_C, fill_opacity=0.5, stroke_width=0).next_to(reserve, RIGHT, buff=0)
        newkv = Rectangle(width=5.0, height=1.0, color=CPU_C, fill_opacity=0.5, stroke_width=0).next_to(newmoe, RIGHT, buff=0)
        cap = swap_caption(self, cap, "At runtime the boundary moves with no restart and no weight reload")
        self.play(Transform(moe, newmoe), Transform(kv, newkv), ls[1].animate.next_to(newmoe, DOWN, buff=0.15), ls[2].animate.next_to(newkv, DOWN, buff=0.15), run_time=2)
        self.wait(1)


class S7_SemanticCache(Scene):
    def construct(self):
        t = title("Semantic-aware caching for agent loops")
        self.play(Write(t))
        cap = swap_caption(self, None, "Radix tree reuses KV for shared prefixes; hybrid models also checkpoint recurrent state")
        row = VGroup(*[Square(0.45, stroke_width=2, color=DIM) for _ in range(20)]).arrange(RIGHT, buff=0.06).shift(UP * 1.2)
        self.play(FadeIn(row))
        names = {4: "system", 9: "tool call", 13: "tool result"}
        self.play(*[row[i].animate.set_fill(RAM_C, 0.7).set_color(RAM_C) for i in range(0, 10)])
        flag = Triangle(color=HOT, fill_opacity=1).scale(0.15).rotate(PI).next_to(row[9], UP, buff=0.1)
        fl = Text("anchor at tool-call opener: GDN state snapshot", font_size=20, color=HOT).next_to(flag, UP, buff=0.15)
        self.play(FadeIn(flag), FadeIn(fl))
        snap = box("state snapshot", w=2.6, h=0.7, color=HOT, fill=HOT, size=20).next_to(row[9], DOWN, buff=0.6)
        self.play(TransformFromCopy(row[9], snap))
        cap = swap_caption(self, cap, "The agent edits context after the tool call: thinking block dropped, result appended")
        self.play(*[row[i].animate.set_fill(GPU_C, 0.7).set_color(GPU_C) for i in range(10, 20)], run_time=0.8)
        edit = VGroup(*[row[i] for i in range(10, 14)])
        self.play(*[e.animate.set_fill(WARN, 0.8).set_color(WARN) for e in edit])
        cap = swap_caption(self, cap, "Restore from the anchor and recompute only the edited tail, not the whole context")
        self.play(Indicate(snap, color=WHITE), *[row[i].animate.set_fill(RAM_C, 0.7).set_color(RAM_C) for i in range(0, 10)])
        self.wait(1.5)


class S0_Title(Scene):
    def construct(self):
        t = Text("FreeToken", font_size=64, weight=BOLD, color=GPU_C)
        s = Text("how the engine works", font_size=30, color=DIM).next_to(t, DOWN)
        self.play(Write(t), FadeIn(s))
        self.wait(1)
