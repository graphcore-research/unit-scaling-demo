# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# pylint: disable=attribute-defined-outside-init
import pandas as pd
from colour import hsl2rgb
from manim import *

config.max_files_cached = 500


class Op(VMobject):
    def __init__(
        self, text, subtext=None, narrow_left=False, narrow_right=False, screen_height=4
    ):
        super().__init__()
        points = [-4, -5, 0], [-4, 5, 0], [4, 5, 0], [4, -5, 0]
        if narrow_left:
            points[0][1] += 2
            points[1][1] -= 2
        if narrow_right:
            points[2][1] -= 2
            points[3][1] += 2
        points = ([c * screen_height / 16 for c in point] for point in points)
        trapezium = Polygon(*points, color=LIGHTER_GRAY, fill_opacity=1)
        if not narrow_left and not narrow_right:
            trapezium.round_corners(0.2)
        self.text = Text(text, color=DARKER_GRAY, font_size=42)
        self.add(trapezium, self.text)
        if subtext:
            self.subtext = Text(subtext, color=DARKER_GRAY, font_size=20).next_to(
                self.text, DOWN
            )
            self.add(self.subtext)


class ExpHist(BarChart):
    def __init__(self, data, bar_colors, y_length=3):
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()
        bar_names = [str(x) if x % 3 == 0 else "" for x in range(15, -14 - 1, -1)]
        super().__init__(
            data,
            bar_names=bar_names,
            y_range=[0, 0.4, 0.1],
            x_length=3.2,
            y_length=y_length,
            bar_colors=bar_colors,
            bar_width=1.0,
            bar_fill_opacity=1.0,
            bar_stroke_width=0,
            x_axis_config={
                "font_size": 28,
                "tick_size": 0.04,
                "color": DARKER_GRAY,
            },
            y_axis_config={
                "font_size": 28,
                "color": DARKER_GRAY,
            },
        )
        self.rotate(-PI / 2)
        for n in self.x_axis.labels:
            n.rotate(PI / 2).shift(DOWN * 0.045)
        self.x_axis.labels.set_color(DARK_GRAY)
        self.x_axis.shift(UP * 0.05)
        self.x_axis.ticks.shift(LEFT * 0.05)
        for n in self.y_axis.numbers:
            n.rotate(PI / 2)
        self.y_axis.numbers.set_color(DARK_GRAY)


class ExpHeat(BarChart):
    def __init__(self, data, tensor_name, col, heat_width, add_axis=False):
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        l_min = col.hsl[2]
        l_max = 0.9
        v_min = data.min()
        v_max = 0.4
        interp = lambda v: l_min + (l_max - l_min) * (1 - (v - v_min) / (v_max - v_min))
        bar_colors = [
            rgb_to_color(hsl2rgb((col.hsl[0], col.hsl[1], interp(v)))) for v in data
        ]

        if add_axis:
            bar_names = [str(x) if x % 3 == 0 else "" for x in range(15, -14 - 1, -1)]
        else:
            bar_names = None

        super().__init__(
            np.ones_like(data),
            bar_names=bar_names,
            y_range=[0, 1, 0.1],
            x_length=3.2,
            y_length=heat_width,
            bar_colors=bar_colors,
            bar_width=1.0,
            bar_fill_opacity=1.0,
            bar_stroke_width=0,
            x_axis_config={
                "font_size": 28,
                "tick_size": 0.04,
                "color": DARKER_GRAY,
            },
        )
        self.rotate(-PI / 2)

        l = self.bars[0].get_left()
        fp8_max_short = (
            DashedLine(
                l,
                l + RIGHT * heat_width,
                stroke_width=2,
                dash_length=0.1,
                color=DARKER_GRAY,
            )
            .next_to(self.x_axis.ticks[7], RIGHT, buff=0)
            .shift(DOWN * 0.008, LEFT * 0.05)
            # .shift(DOWN * 0.03, LEFT * 0.1)
        )
        fp8_min_short = (
            DashedLine(
                l,
                l + RIGHT * heat_width,
                stroke_width=2,
                dash_length=0.1,
                color=DARKER_GRAY,
            )
            .next_to(self.x_axis.ticks[22], RIGHT, buff=0)
            .shift(DOWN * 0.008, LEFT * 0.05)
            # .shift(DOWN * 0.03, LEFT * 0.1)
        )
        fp8_max_short.z_index = 1
        fp8_min_short.z_index = 1
        self.add(fp8_max_short, fp8_min_short)

        label = MathTex(tensor_name, color=DARKER_GRAY).move_to(self.bars[-3])
        label.z_index = 1
        self.add(label)

        self.remove(self.y_axis)
        if add_axis:
            for n in self.x_axis.labels:
                n.rotate(PI / 2).shift(DOWN * 0.045)
            self.x_axis.labels.set_color(DARK_GRAY)
            self.x_axis.shift(UP * 0.05)
            self.x_axis.ticks.shift(LEFT * 0.05)
        else:
            self.remove(self.x_axis)


class USBox(BackgroundRectangle):
    def __init__(self, text, base, col, direction):
        self.us_text = (
            MathTex(text, color=DARKER_GRAY)
            .scale(0.7)
            .move_to(base)
            .shift(direction * 0.5)
        )
        self.us_text.z_index = 3
        super().__init__(
            self.us_text,
            color=lighten(col),
            fill_opacity=1.0,
            buff=0.1,
        )
        self.add(self.us_text)


def lighten(col):
    return rgb_to_color(hsl2rgb((col.hsl[0], col.hsl[1], 0.8)))


class Main(Scene):
    def setup(self):
        self.camera.background_color = WHITE
        self.hist_width = 1.2815
        self.heat_width = 1.0
        self.base_colors = {
            "x": rgb_to_color([0.424, 0.557, 0.749]),
            "dx": rgb_to_color([1, 0.6, 0.2]),
            "w": rgb_to_color([0.51, 0.702, 0.40]),
            "dw": rgb_to_color([1, 0.463, 0.38]),
        }
        self.reg_df = pd.read_pickle("./reg.pkl").iloc[::-1]
        self.us_df = pd.read_pickle("./us.pkl").iloc[::-1]

    def define_confidential(self):
        TOP_LEFT = UP * 3.85 + LEFT * 7
        gc_dark = rgb_to_color(hex_to_rgb("#292C31"))
        conf_text = Text("Confidential", font_size=100, font="Graphik", color=gc_dark)
        logo = SVGMobject("gc-dev-icon-01.svg")
        conf_text.next_to(logo, RIGHT, buff=0.4)
        self.conf_logo = VGroup(logo, conf_text).scale(0.2).move_to(TOP_LEFT, LEFT + UP)

    def animate_confidential(self):
        self.add(self.conf_logo)

    def define_opening_text(self):
        self.problem_txt = (
            Tex(
                "for most models, tensor scales\n\nare inconsistent at initialisation",
                color=DARKER_GRAY,
                tex_environment="flushleft",
            )
            .scale(1.4)
            .shift(RIGHT * 1.5 + UP * 2)
        )
        self.problem_prompt = (
            Tex("Problem: ", color=self.base_colors["dw"])
            .scale(1.4)
            .next_to(self.problem_txt, LEFT, buff=0.4)
        )
        self.example_prompt = (
            Tex("Example: ", color=self.base_colors["x"])
            .scale(1.4)
            .align_to(self.problem_prompt, RIGHT)
            .shift(DOWN * 2)
        )
        self.example_txt = (
            Tex(
                "Consider the FFN\n\nlayer in BERT\\textsubscript{LARGE}...",
                color=DARKER_GRAY,
                tex_environment="flushleft",
            ).scale(1.4)
        ).next_to(self.example_prompt, RIGHT, buff=0.4)

    def animate_opening_text(self):
        self.play(FadeIn(self.problem_prompt, self.problem_txt, run_time=0.7))
        self.wait(2.25)
        self.play(FadeIn(self.example_prompt, self.example_txt, run_time=0.7))
        self.wait(2)
        self.play(FadeOut(self.problem_prompt, self.problem_txt))
        self.wait(0.5)

    def define_ffn_diagram(self):
        self.matmul_a = Op("matmul", "(1024 × 4096)", narrow_left=True)
        self.gelu = Op("GeLU")
        self.matmul_b = Op("matmul", "(4096 × 1024)", narrow_right=True)
        self.matmul_a.shift(LEFT * 4)
        self.matmul_b.shift(RIGHT * 4)
        self.ffn_diagram = VGroup(self.matmul_a, self.gelu, self.matmul_b).shift(
            UP * 1.5
        )

    def animate_ffn_diagram(self):
        self.play(FadeIn(self.ffn_diagram))

    def define_arrows(self):
        self.x1_arrow = Arrow(1.45 * LEFT, RIGHT, color=self.base_colors["x"])
        self.x1_arrow.next_to(self.matmul_a, LEFT, buff=0.02).shift(UP * 0.6)
        self.x1_text = MathTex("x_1", color=DARKER_GRAY).next_to(
            self.x1_arrow, UP, buff=0.03
        )
        self.w1_arrow = Arrow(
            0.55 * UP,
            DOWN,
            color=self.base_colors["w"],
            max_tip_length_to_length_ratio=1,
        )
        self.w1_arrow.next_to(self.matmul_a, UP, buff=0.02).shift(
            DOWN * 0.4 + LEFT * 0.6
        )
        self.w1_text = (
            MathTex("w_1", color=DARKER_GRAY)
            .next_to(self.w1_arrow, LEFT, buff=0.03)
            .shift(UP * 0.1)
        )
        self.x2_arrow = Arrow(1.45 * LEFT, RIGHT, color=self.base_colors["x"])
        self.x2_arrow.next_to(self.gelu, LEFT, buff=0.02).shift(UP * 0.6)
        self.x2_text = MathTex("x_2", color=DARKER_GRAY).next_to(
            self.x2_arrow, UP, buff=0.03
        )
        self.x3_arrow = Arrow(1.45 * LEFT, RIGHT, color=self.base_colors["x"])
        self.x3_arrow.next_to(self.matmul_b, LEFT, buff=0.02).shift(UP * 0.6)
        self.x3_text = MathTex("x_3", color=DARKER_GRAY).next_to(
            self.x3_arrow, UP, buff=0.03
        )
        self.x4_arrow = Arrow(1.45 * LEFT, RIGHT, color=self.base_colors["x"])
        self.x4_arrow.next_to(self.matmul_b, RIGHT, buff=0.02).shift(UP * 0.6)
        self.x4_text = MathTex("x_4", color=DARKER_GRAY).next_to(
            self.x4_arrow, UP, buff=0.03
        )
        self.w2_arrow = Arrow(
            0.55 * UP,
            DOWN,
            color=self.base_colors["w"],
            max_tip_length_to_length_ratio=1,
        )
        self.w2_arrow.next_to(self.matmul_b, UP, buff=0.02).shift(
            DOWN * 0.4 + RIGHT * 0.6
        )
        self.w2_text = MathTex("w_2", color=DARKER_GRAY).next_to(
            self.w2_arrow, LEFT, buff=0.03
        )
        self.dx4_arrow = Arrow(1.45 * RIGHT, LEFT, color=self.base_colors["dx"])
        self.dx4_arrow.next_to(self.matmul_b, RIGHT, buff=0.02).shift(DOWN * 0.6)
        self.dx4_text = MathTex("\\nabla_{x_4}", color=DARKER_GRAY).next_to(
            self.dx4_arrow, DOWN, buff=0.03
        )
        self.dx3_arrow = Arrow(1.45 * RIGHT, LEFT, color=self.base_colors["dx"])
        self.dx3_arrow.next_to(self.matmul_b, LEFT, buff=0.02).shift(DOWN * 0.6)
        self.dx3_text = MathTex("\\nabla_{x_3}", color=DARKER_GRAY).next_to(
            self.dx3_arrow, DOWN, buff=0.03
        )
        self.dw2_arrow = Arrow(
            0.55 * UP,
            DOWN,
            color=self.base_colors["dw"],
            max_tip_length_to_length_ratio=1,
        )
        self.dw2_arrow.next_to(self.matmul_b, DOWN, buff=0.02).shift(
            UP * 0.4 + RIGHT * 0.6
        )
        self.dw2_text = MathTex("\\nabla_{w_2}", color=DARKER_GRAY).next_to(
            self.dw2_arrow, LEFT, buff=0.03
        )
        self.dx2_arrow = Arrow(1.45 * RIGHT, LEFT, color=self.base_colors["dx"])
        self.dx2_arrow.next_to(self.gelu, LEFT, buff=0.02).shift(DOWN * 0.6)
        self.dx2_text = MathTex("\\nabla_{x_2}", color=DARKER_GRAY).next_to(
            self.dx2_arrow, DOWN, buff=0.03
        )
        self.dx1_arrow = Arrow(1.45 * RIGHT, LEFT, color=self.base_colors["dx"])
        self.dx1_arrow.next_to(self.matmul_a, LEFT, buff=0.02).shift(DOWN * 0.6)
        self.dx1_text = (
            MathTex("\\nabla_{x_1}", color=DARKER_GRAY)
            .next_to(self.dx1_arrow, DOWN, buff=0.03)
            .shift(LEFT * 0.25)
        )
        self.dw1_arrow = Arrow(
            0.55 * UP,
            DOWN,
            color=self.base_colors["dw"],
            max_tip_length_to_length_ratio=1,
        )
        self.dw1_arrow.next_to(self.matmul_a, DOWN, buff=0.02).shift(
            UP * 0.4 + LEFT * 0.6
        )
        self.dw1_text = MathTex("\\nabla_{w_1}", color=DARKER_GRAY).next_to(
            self.dw1_arrow, LEFT, buff=0.03
        )

    def define_reg_sigmas(self):
        self.x1_sigma = (
            MathTex("\sigma = 1", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.x1_arrow, DOWN, buff=0.02)
        )
        self.w1_sigma = (
            MathTex("\sigma = 0.02", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.w1_arrow, RIGHT, buff=0.03)
            .shift(UP * 0.1)
        )
        self.w2_sigma = (
            MathTex("\sigma = 0.02", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.w2_arrow, RIGHT, buff=0.03)
            .shift(UP * 0.1)
        )

    def animate_x1_w1(self):
        self.wait(0.5)
        self.play(
            GrowArrow(self.x1_arrow),
            FadeIn(self.x1_text),
            GrowArrow(self.w1_arrow),
            FadeIn(self.w1_text),
        )
        self.wait(1)
        self.play(
            FadeIn(self.x1_sigma),
            FadeIn(self.w1_sigma),
        )
        self.wait(1)

    def define_histograms_base(self):
        self.w1_bar_colors = [self.base_colors["w"] for _ in range(30)]
        self.w1_hist_pos = DOWN * 2 + LEFT * 3
        self.w1_hist_true_bars = ExpHist(self.reg_df["w1"], self.w1_bar_colors).move_to(
            self.w1_hist_pos
        )
        self.w1_hist = ExpHist(np.zeros(30), self.w1_bar_colors).move_to(
            self.w1_hist_pos
        )
        self.x1_bar_colors = [self.base_colors["x"] for _ in range(30)]
        self.x1_hist_pos = DOWN * 2 + RIGHT * 3
        self.x1_hist_true_bars = ExpHist(self.reg_df["x1"], self.x1_bar_colors).move_to(
            self.x1_hist_pos
        )
        self.x1_hist = ExpHist(np.zeros(30), self.x1_bar_colors).move_to(
            self.x1_hist_pos
        )
        self.w1_text_chart = self.w1_text.copy()
        self.x1_text_chart = self.x1_text.copy()
        self.hist_exp_text = (
            Tex("Exponent value", font_size=32, color=DARKER_GRAY)
            .align_to(self.w1_hist.x_axis.labels[0], RIGHT)
            .shift(DOWN * 0.4)
        )
        self.hist_freq_text = (
            Tex("Frequency", font_size=32, color=DARKER_GRAY)
            .next_to(self.w1_hist.y_axis.numbers, UP, buff=0.1)
            .shift(RIGHT * 0.5)
        )

    def animate_histograms_base(self):
        self.play(FadeOut(self.example_prompt), FadeOut(self.example_txt))
        self.wait(1)
        self.play(
            self.w1_text_chart.animate.next_to(self.w1_hist, LEFT),
            self.x1_text_chart.animate.next_to(self.x1_hist, LEFT),
        )
        self.play(
            FadeIn(self.w1_hist, self.x1_hist, self.hist_exp_text, self.hist_freq_text),
        )
        self.wait(0.5)
        self.play(
            Transform(self.w1_hist.bars, self.w1_hist_true_bars.bars),
            Transform(self.x1_hist.bars, self.x1_hist_true_bars.bars),
        )
        self.wait(2.5)

    def define_histogram_fp_lines(self):
        l = self.w1_hist.y_axis.ticks.get_left()
        r = self.w1_hist.y_axis.ticks.get_right()
        _r = l + (r - l) * 4 / 3
        self.w1_fp16_min = (
            DashedLine(l, _r, stroke_width=2, dash_length=0.05, color=DARKER_GRAY)
            .next_to(self.w1_hist.x_axis.ticks[29], RIGHT, buff=0)
            .shift(DOWN * 0.045)
        )
        self.x1_fp16_min = (
            DashedLine(l, _r, stroke_width=2, dash_length=0.05, color=DARKER_GRAY)
            .next_to(self.x1_hist.x_axis.ticks[29], RIGHT, buff=0)
            .shift(DOWN * 0.045)
        )
        self.w1_fp16_max_text = (
            Tex("FP16 max", color=DARKER_GRAY, font_size=30)
            .next_to(self.w1_hist.y_axis, RIGHT, buff=0.1)
            .shift(DOWN * 0.2 + LEFT * 0.2)
        )
        self.w1_fp16_min_text = (
            Tex("FP16 min", color=DARKER_GRAY, font_size=30)
            .next_to(self.w1_fp16_min, RIGHT, buff=0.1)
            .shift(UP * 0.1)
        )
        self.w1_fp8_max = (
            DashedLine(l, _r, stroke_width=2, dash_length=0.1, color=DARKER_GRAY)
            .next_to(self.w1_hist.x_axis.ticks[7], RIGHT, buff=0)
            .shift(DOWN * 0.08)
        )
        self.w1_fp8_min = (
            DashedLine(l, _r, stroke_width=2, dash_length=0.1, color=DARKER_GRAY)
            .next_to(self.w1_hist.x_axis.ticks[22], RIGHT, buff=0)
            .shift(DOWN * 0.08)
        )
        self.x1_fp8_max = (
            DashedLine(l, _r, stroke_width=2, dash_length=0.1, color=DARKER_GRAY)
            .next_to(self.x1_hist.x_axis.ticks[7], RIGHT, buff=0)
            .shift(DOWN * 0.055)
        )
        self.x1_fp8_min = (
            DashedLine(l, _r, stroke_width=2, dash_length=0.1, color=DARKER_GRAY)
            .next_to(self.x1_hist.x_axis.ticks[22], RIGHT, buff=0)
            .shift(DOWN * 0.055)
        )
        self.w1_fp8_max_text = Tex(
            "FP8 E4\n\nmax",
            color=DARKER_GRAY,
            font_size=30,
            tex_environment="flushleft",
        ).next_to(self.w1_fp8_max, RIGHT, buff=0.1)
        self.fp8_min_text = Tex(
            "FP8 E4\n\nmin",
            color=DARKER_GRAY,
            font_size=30,
            tex_environment="flushleft",
        ).next_to(self.w1_fp8_min, RIGHT, buff=0.1)
        self.w1_fp8_max_short = (
            DashedLine(
                l,
                l + RIGHT * self.heat_width,
                stroke_width=2,
                dash_length=0.1,
                color=DARKER_GRAY,
            )
            .next_to(self.w1_hist.x_axis.ticks[7], RIGHT, buff=0)
            .shift(DOWN * 0.06)
        )
        self.w1_fp8_min_short = (
            DashedLine(
                l,
                l + RIGHT * self.heat_width,
                stroke_width=2,
                dash_length=0.1,
                color=DARKER_GRAY,
            )
            .next_to(self.w1_hist.x_axis.ticks[22], RIGHT, buff=0)
            .shift(DOWN * 0.06)
        )
        self.w1_fp8_min.z_index = 1
        self.w1_fp8_max.z_index = 1
        self.w1_fp8_min_short.z_index = 1
        self.w1_fp8_max_short.z_index = 1
        self.w1_text_chart.z_index = 1
        self.x1_fp8_min.z_index = 1
        self.x1_fp8_max.z_index = 1
        self.x1_text_chart.z_index = 1

    def animate_histogram_fp_lines(self):
        self.play(
            FadeIn(
                self.w1_fp16_min,
                self.w1_fp16_max_text,
                self.w1_fp16_min_text,
                self.x1_fp16_min,
            )
        )
        self.wait(1)
        self.play(
            FadeIn(
                self.w1_fp8_max,
                self.w1_fp8_min,
                self.w1_fp8_max_text,
                self.fp8_min_text,
                self.x1_fp8_max,
                self.x1_fp8_min,
            )
        )
        self.wait(2.5)

    def define_histogram_transform_w1(self):
        col = self.base_colors["w"]
        l_min = col.hsl[2]
        l_max = 0.9
        v_min = self.reg_df["w1"].min()
        v_max = 0.4
        interp = lambda v: l_min + (l_max - l_min) * (1 - (v - v_min) / (v_max - v_min))
        w1_bar_new_colors = [
            rgb_to_color(hsl2rgb((col.hsl[0], col.hsl[1], interp(v))))
            for v in self.reg_df["w1"]
        ]

        self.w1_col_hist = ExpHist(self.reg_df["w1"], w1_bar_new_colors).move_to(
            self.w1_hist_pos
        )
        self.w1_wide_hist = ExpHist(
            np.ones_like(self.reg_df["w1"]) * 0.4,
            w1_bar_new_colors,
            y_length=self.heat_width,
        )
        self.w1_wide_hist.remove(self.w1_wide_hist.y_axis).move_to(
            self.w1_hist.x_axis, LEFT
        )
        self.w1_reg_heat = VGroup(
            self.w1_hist,
            self.w1_fp16_min,
            self.w1_fp8_min,
            self.w1_fp8_max,
            self.w1_text_chart,
        )

    def animate_histogram_transform_w1(self):
        self.play(Transform(self.w1_hist.bars, self.w1_col_hist.bars))
        self.wait(0.6)
        self.play(
            Transform(self.w1_hist.bars, self.w1_wide_hist.bars),
            Transform(self.w1_fp8_max, self.w1_fp8_max_short),
            Transform(self.w1_fp8_min, self.w1_fp8_min_short),
            FadeOut(
                self.w1_hist.y_axis,
                self.w1_fp16_min,
                self.hist_freq_text,
                self.w1_fp16_min_text,
                self.fp8_min_text,
                self.w1_fp8_max_text,
                self.w1_fp16_max_text,
            ),
        )
        self.w1_fp16_min.color = WHITE
        self.w1_fp16_min.shift(DOWN * 10)
        self.w1_hist.remove(self.w1_hist.y_axis)
        self.wait(1)
        self.play(
            self.w1_hist.animate.shift(LEFT * 2.19 + DOWN * 0.025),
            self.w1_fp8_max.animate.shift(LEFT * 2.18),
            self.w1_fp8_min.animate.shift(LEFT * 2.18),
            self.w1_text_chart.animate.move_to(self.w1_hist.bars[-3]).shift(LEFT * 2.2),
            self.hist_exp_text.animate.shift(LEFT * 0.25),
        )

    def define_histogram_transform_x1(self):
        col = self.base_colors["x"]
        l_min = col.hsl[2]
        l_max = 0.9
        v_min = self.reg_df["x1"].min()
        v_max = 0.4
        interp = lambda v: l_min + (l_max - l_min) * (1 - (v - v_min) / (v_max - v_min))
        x1_bar_new_colors = [
            rgb_to_color(hsl2rgb((col.hsl[0], col.hsl[1], interp(v))))
            for v in self.reg_df["x1"]
        ]
        self.x1_col_hist = ExpHist(self.reg_df["x1"], x1_bar_new_colors).move_to(
            self.x1_hist_pos
        )
        self.x1_wide_hist = ExpHist(
            np.ones_like(self.reg_df["x1"]) * 0.4,
            x1_bar_new_colors,
            y_length=self.heat_width,
        )
        self.x1_wide_hist.remove(self.x1_wide_hist.y_axis).move_to(
            self.x1_hist.x_axis, LEFT
        )
        l = self.w1_hist.y_axis.ticks.get_left()
        self.fp8_max_short = (
            DashedLine(
                l,
                l + RIGHT * self.heat_width,
                stroke_width=2,
                dash_length=0.1,
                color=DARKER_GRAY,
            )
            .next_to(self.x1_hist.x_axis.ticks[7], RIGHT, buff=0)
            .shift(DOWN * 0.035)
        )
        self.fp8_min_short = (
            DashedLine(
                l,
                l + RIGHT * self.heat_width,
                stroke_width=2,
                dash_length=0.1,
                color=DARKER_GRAY,
            )
            .next_to(self.x1_hist.x_axis.ticks[22], RIGHT, buff=0)
            .shift(DOWN * 0.035)
        )
        self.heat_base_line = self.w1_hist.bars.get_left() + LEFT * 2.2
        self.x1_reg_heat = VGroup(
            self.x1_hist,
            self.x1_fp16_min,
            self.x1_fp8_min,
            self.x1_fp8_max,
            self.x1_text_chart,
        )

    def animate_histogram_transform_x1(self):
        self.play(Transform(self.x1_hist.bars, self.x1_col_hist.bars))
        self.play(
            Transform(self.x1_hist.bars, self.x1_wide_hist.bars),
            Transform(self.x1_fp8_max, self.fp8_max_short),
            Transform(self.x1_fp8_min, self.fp8_min_short),
            FadeOut(
                self.x1_hist.y_axis,
                self.x1_fp16_min,
            ),
        )
        self.x1_fp16_min.color = WHITE
        self.x1_fp16_min.shift(DOWN * 10)
        self.x1_hist.remove(self.x1_hist.y_axis)
        self.wait(1)
        self.x1_hist_shift = (
            self.heat_base_line
            - self.x1_hist.bars.get_left()
            + 2.01 * RIGHT * self.heat_width
        )
        self.play(
            self.x1_hist.animate.shift(self.x1_hist_shift),
            self.x1_fp8_max.animate.shift(self.x1_hist_shift),
            self.x1_fp8_min.animate.shift(self.x1_hist_shift),
            self.x1_text_chart.animate.move_to(self.x1_hist.bars[-3]).shift(
                self.x1_hist_shift
            ),
        )
        self.play(FadeOut(self.x1_hist.x_axis))
        self.x1_hist.remove(self.x1_hist.x_axis)
        self.wait(1.5)

    def define_reg_x2(self):
        self.x2_sigma = (
            MathTex("\sigma = ", color=DARKER_GRAY)
            .scale(0.75)
            .next_to(self.x2_arrow, DOWN, buff=0.05)
            .shift(LEFT * 0.5)
        )
        self.w1_sigma_clone = (
            MathTex("0.02", color=DARKER_GRAY)
            .scale(0.7)
            .align_to(self.w1_sigma, RIGHT)
            .align_to(self.w1_sigma, DOWN)
        )
        self.times_a = (
            MathTex("\\times ", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.x2_sigma, DOWN, buff=0.2)
            .shift(RIGHT * 0.17)
        )
        self.x1_sigma_clone = (
            MathTex("1", color=DARKER_GRAY)
            .scale(0.7)
            .align_to(self.x1_sigma, RIGHT)
            .align_to(self.x1_sigma, DOWN)
        )
        self.times_b = (
            MathTex("\\times ", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.times_a, DOWN, buff=0.2)
        )
        self.matmul_a_in_dim = (
            MathTex("\\sqrt{1024}", color=DARKER_GRAY)
            .scale(0.7)
            .align_to(self.matmul_a.subtext, LEFT)
            .align_to(self.matmul_a.subtext, DOWN)
        )
        self.x2_sigma_value = (
            MathTex("0.64", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.x2_sigma, RIGHT, buff=0.2)
        )
        self.x2_reg_heat = ExpHeat(
            self.reg_df["x2"], "x_2", self.base_colors["x"], self.heat_width
        ).move_to(self.heat_base_line + RIGHT * 4.5 * self.heat_width, LEFT)

    def animate_reg_x2(self):
        self.play(GrowArrow(self.x2_arrow), FadeIn(self.x2_text))
        self.wait(0.6)
        self.play(FadeIn(self.x2_sigma))
        self.play(self.w1_sigma_clone.animate.next_to(self.x2_sigma, RIGHT, buff=0.2))
        self.wait(0.5)
        self.play(
            FadeIn(self.times_a),
            self.x1_sigma_clone.animate.next_to(self.times_a, RIGHT, buff=0.2),
        )
        self.wait(0.5)
        self.play(
            FadeIn(self.times_b),
            self.matmul_a_in_dim.animate.next_to(self.times_b, RIGHT, buff=0.2),
        )
        self.wait(1.5)
        self.play(
            FadeOut(
                self.w1_sigma_clone,
                self.times_a,
                self.x1_sigma_clone,
                self.times_b,
                self.matmul_a_in_dim,
                shift=DOWN * 0.25,
            ),
            FadeIn(self.x2_sigma_value, shift=UP * 0.25),
        )
        self.wait(1)
        self.play(FadeIn(self.x2_reg_heat), FadeOut(self.x2_sigma, self.x2_sigma_value))
        self.wait(1.5)

    def define_reg_x3(self):
        self.x3_reg_heat = ExpHeat(
            self.reg_df["x3"], "x_3", self.base_colors["x"], self.heat_width
        ).move_to(self.heat_base_line + RIGHT * 7 * self.heat_width, LEFT)

    def animate_reg_x3(self):
        self.play(GrowArrow(self.x3_arrow), FadeIn(self.x3_text, self.x3_reg_heat))
        self.wait(1.5)

    def define_reg_x4_w2(self):
        self.w2_reg_heat = ExpHeat(
            self.reg_df["w2"], "w_2", self.base_colors["w"], self.heat_width
        ).move_to(self.heat_base_line + RIGHT * 9.5 * self.heat_width, LEFT)
        self.x4_reg_heat = ExpHeat(
            self.reg_df["x4"], "x_4", self.base_colors["x"], self.heat_width
        ).move_to(self.heat_base_line + RIGHT * 11.5 * self.heat_width, LEFT)

    def animate_reg_x4_w2(self):
        self.play(
            GrowArrow(self.x4_arrow),
            GrowArrow(self.w2_arrow),
            FadeIn(
                self.x4_text,
                self.w2_text,
                self.w2_sigma,
                self.x4_reg_heat,
                self.w2_reg_heat,
            ),
        )
        self.wait(1.5)

    def define_loss_scaling_dx4(self):
        dx4_data = self.reg_df["x4_grad"].to_numpy()
        ls_sweep_data = [
            np.roll(dx4_data, i)
            for i in list(range(8, -8 - 1, -1)) + list(range(-8, 0 + 1))
        ]
        self.ls_sweep_hists = [
            ExpHeat(
                data, "\\nabla_{x_4}", self.base_colors["dx"], self.heat_width
            ).move_to(self.heat_base_line + RIGHT * 12.5 * self.heat_width, LEFT)
            for data in ls_sweep_data
        ]
        self.ls_text = Tex("Sweep loss scale", color=DARKER_GRAY, font_size=38)
        self.ls_rect = SurroundingRectangle(
            self.ls_text,
            color=self.base_colors["dx"],
            stroke_width=2,
            stroke_opacity=1.0,
            buff=0.125,
        ).move_to(
            self.ls_sweep_hists[0].bars[0].get_right() - RIGHT * 0.1 + UP * 0.5, RIGHT
        )
        self.ls_text.move_to(self.ls_rect)

    def animate_loss_scaling_dx4(self):
        self.play(
            GrowArrow(self.dx4_arrow), FadeIn(self.dx4_text, self.ls_sweep_hists[0])
        )
        self.wait(1)
        self.play(FadeIn(self.ls_rect, self.ls_text))
        self.wait(1)
        h = self.ls_sweep_hists[0]
        for h_new in self.ls_sweep_hists[1:]:
            self.play(Transform(h, h_new, run_time=0.04))
        self.wait(1.5)

    def define_reg_dx3_dw2(self):
        self.dx3_reg_heat = ExpHeat(
            self.reg_df["x3_grad"],
            "\\nabla_{x_3}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 8 * self.heat_width, LEFT)
        self.dw2_reg_heat = ExpHeat(
            self.reg_df["w2_grad"],
            "\\nabla_{w_2}",
            self.base_colors["dw"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 10.5 * self.heat_width, LEFT)

    def animate_reg_dx3_dw2(self):
        self.play(FadeOut(self.ls_rect, self.ls_text))
        self.play(
            GrowArrow(self.dx3_arrow),
            GrowArrow(self.dw2_arrow),
            FadeIn(self.dx3_text, self.dx3_reg_heat, self.dw2_text, self.dw2_reg_heat),
        )
        self.wait(2.5)

    def define_reg_dx2(self):
        self.dx2_reg_heat = ExpHeat(
            self.reg_df["x2_grad"],
            "\\nabla_{x_2}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 5.5 * self.heat_width, LEFT)

    def animate_reg_dx2(self):
        self.play(GrowArrow(self.dx2_arrow), FadeIn(self.dx2_text, self.dx2_reg_heat))
        self.wait(1.5)

    def define_reg_dx1_dw1(self):
        self.dx1_reg_heat = ExpHeat(
            self.reg_df["x1_grad"],
            "\\nabla_{x_1}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 3 * self.heat_width, LEFT)
        self.dw1_reg_heat = ExpHeat(
            self.reg_df["w1_grad"],
            "\\nabla_{w_1}",
            self.base_colors["dw"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 1 * self.heat_width, LEFT)

    def animate_reg_dx1_dw1(self):
        self.play(
            GrowArrow(self.dx1_arrow),
            GrowArrow(self.dw1_arrow),
            FadeIn(self.dx1_text, self.dw1_text, self.dx1_reg_heat, self.dw1_reg_heat),
        )
        self.wait(2.5)

    def define_reg_heats(self):
        self.reg_heats = VGroup(
            self.hist_exp_text,
            self.w1_reg_heat,
            self.x1_reg_heat,
            self.x2_reg_heat,
            self.x3_reg_heat,
            self.w2_reg_heat,
            self.x4_reg_heat,
            self.ls_sweep_hists[0],
            self.dw2_reg_heat,
            self.dx3_reg_heat,
            self.dx2_reg_heat,
            self.dx1_reg_heat,
            self.dw1_reg_heat,
        )

    def define_us_text(self):
        self.solution_prompt = (
            Tex("Solution: ", color=self.base_colors["w"])
            .scale(1.4)
            .move_to(self.problem_prompt, RIGHT)
        )
        self.solution_txt = (
            Tex(
                "\\textit{unit scaling} scales each op to\n\nmaintain local unit variance",
                color=DARKER_GRAY,
                tex_environment="flushleft",
            ).scale(1.4)
        ).next_to(self.solution_prompt, RIGHT, buff=0.4)

    def animate_fade_out_reg_diagram(self):
        self.play(
            FadeOut(
                self.ffn_diagram,
                self.x1_text,
                self.x1_arrow,
                self.x1_sigma,
                self.w1_text,
                self.w1_arrow,
                self.w1_sigma,
                self.x2_text,
                self.x2_arrow,
                self.x3_text,
                self.x3_arrow,
                self.x4_text,
                self.x4_arrow,
                self.w2_text,
                self.w2_arrow,
                self.w2_sigma,
                self.dx4_text,
                self.dx4_arrow,
                self.dx1_text,
                self.dx1_arrow,
                self.dw1_text,
                self.dw1_arrow,
                self.dx2_text,
                self.dx2_arrow,
                self.dx3_text,
                self.dx3_arrow,
                self.dw2_text,
                self.dw2_arrow,
                self.dx4_text,
                self.dx4_arrow,
            )
        )

    def animate_us_text(self):
        self.play(FadeIn(self.problem_prompt, self.problem_txt, run_time=0.7))
        self.wait(1.5)

        self.play(FadeOut(self.problem_prompt, self.problem_txt, run_time=0.7))
        self.play(FadeIn(self.solution_prompt, self.solution_txt, run_time=0.7))
        self.wait(2.5)

    def animate_fade_out_reg_heats(self):
        self.play(FadeOut(self.reg_heats))
        self.wait(1.5)

    def animate_fade_out_us_text(self):
        self.play(FadeOut(self.solution_prompt, self.solution_txt, run_time=0.7))

    def define_us_matmul_a(self):
        self.w1_sigma_us = (
            MathTex("\sigma = 1", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.w1_arrow, RIGHT, buff=0.03)
            .shift(UP * 0.1)
        )
        self.w1_us_heat = ExpHeat(
            self.us_df["w1"],
            "w_1",
            self.base_colors["w"],
            self.heat_width,
            add_axis=True,
        ).move_to(
            self.heat_base_line + RIGHT * 1.015 * self.heat_width + UP * 0.025, RIGHT
        )
        self.x1_us_heat = ExpHeat(
            self.us_df["x1"],
            "x_1",
            self.base_colors["x"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 2 * self.heat_width, LEFT)
        self.us_text = Tex("Add scaling factor", color=DARKER_GRAY, font_size=38)
        self.us_rect = (
            SurroundingRectangle(
                self.us_text,
                color=LIGHTER_GRAY,
                stroke_width=3,
                stroke_opacity=1.0,
                buff=0.125,
            )
            .move_to(self.matmul_a)
            .shift(DOWN * 1.7)
        )
        self.us_text.move_to(self.us_rect)
        self.us_div = (
            MathTex("\\div", color=DARKER_GRAY)
            .scale(0.7)
            .move_to(self.matmul_a)
            .shift(DOWN * 1.7 + LEFT * 1)
        )
        self.us_scale_formula = (
            MathTex("\\sqrt{\\sqrt{1024} \\times \\sqrt{4096}}", color=DARKER_GRAY)
            .scale(0.4)
            .move_to(self.matmul_a.subtext)
        )
        self.us_scale_formula_value = (
            MathTex("45", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.us_div, RIGHT, buff=0.1)
        )
        self.us_div.z_index = 3
        self.us_scale_formula_value.z_index = 3
        self.x2_us_scale_formula_sq = BackgroundRectangle(
            VGroup(self.us_div, self.us_scale_formula_value),
            color=lighten(self.base_colors["x"]),
            fill_opacity=1.0,
            buff=0.1,
        )
        self.x2_us_scale = VGroup(
            self.x2_us_scale_formula_sq, self.us_div, self.us_scale_formula_value
        )

    def animate_us_matmul_a(self):
        self.play(FadeOut(self.w1_sigma, run_time=0.5, shift=DOWN * 0.2))
        self.play(FadeIn(self.w1_sigma_us, run_time=0.5, shift=UP * 0.2))
        self.wait(0.5)
        self.play(FadeIn(self.w1_us_heat, self.x1_us_heat))
        self.wait(1.3)
        self.play(FadeIn(self.us_text, self.us_rect))
        self.wait(1.5)
        self.play(FadeOut(self.us_text, self.us_rect))
        self.wait(0.25)
        self.play(
            FadeIn(self.us_div),
            self.us_scale_formula.animate.scale(0.6 / 0.4).next_to(
                self.us_div, RIGHT, buff=0.1
            ),
        )
        self.wait(1.5)
        self.play(FadeOut(self.us_scale_formula, run_time=0.5, shift=DOWN * 0.2))
        self.play(FadeIn(self.us_scale_formula_value, run_time=0.5, shift=UP * 0.2))
        self.wait(1)
        self.play(FadeIn(self.x2_us_scale_formula_sq))
        self.play(self.x2_us_scale.animate.next_to(self.matmul_a.text, UP, buff=0.1))
        self.wait(1)

    def define_us_x2(self):
        self.x2_us_heat = ExpHeat(
            self.us_df["x2"],
            "x_2",
            self.base_colors["x"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 4.5 * self.heat_width, LEFT)

    def animate_us_x2(self):
        self.play(
            GrowArrow(self.x2_arrow),
            FadeIn(self.x2_text, self.x2_us_heat),
        )
        self.wait(2)

    def define_us_x3(self):
        self.x3_us_heat = ExpHeat(
            self.us_df["x3"],
            "x_3",
            self.base_colors["x"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 7 * self.heat_width, LEFT)
        self.x3_us_scale = USBox("\\times 2.5", self.gelu, self.base_colors["x"], UP)

    def animate_us_x3(self):
        self.play(FadeIn(self.x3_us_scale))
        self.wait(1)
        self.play(GrowArrow(self.x3_arrow), FadeIn(self.x3_text, self.x3_us_heat))
        self.wait(1)

    def define_us_w2_x4(self):
        self.w2_us_heat = ExpHeat(
            self.us_df["w2"],
            "w_2",
            self.base_colors["w"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 9.5 * self.heat_width, LEFT)
        self.w2_sigma_us = (
            MathTex("\sigma = 1", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.w2_arrow, RIGHT, buff=0.03)
            .shift(UP * 0.1)
        )
        self.x4_us_heat = ExpHeat(
            self.us_df["x4"],
            "x_4",
            self.base_colors["x"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 11.5 * self.heat_width, LEFT)
        self.x4_us_scale = USBox("\\div 45", self.matmul_b, self.base_colors["x"], UP)

    def animate_us_w2_x4(self):
        self.play(
            GrowArrow(self.w2_arrow),
            FadeIn(self.w2_text, self.w2_sigma_us, self.w2_us_heat),
        )
        self.wait(1)
        self.play(FadeIn(self.x4_us_scale))
        self.wait(1)
        self.play(GrowArrow(self.x4_arrow), FadeIn(self.x4_text, self.x4_us_heat))
        self.wait(1)

    def define_us_dx4(self):
        self.dx4_us_heat = ExpHeat(
            self.us_df["x4_grad"],
            "\\nabla_{x_4}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 12.5 * self.heat_width, LEFT)
        self.no_sweep = Tex("No sweep required", color=DARKER_GRAY, font_size=38)
        self.no_sweep_rect = SurroundingRectangle(
            self.no_sweep,
            color=self.base_colors["dx"],
            stroke_width=2,
            stroke_opacity=1.0,
            buff=0.125,
        ).move_to(self.dx4_us_heat.bars[0].get_right() - RIGHT * 0.1 + UP * 0.5, RIGHT)
        self.no_sweep.move_to(self.no_sweep_rect)

    def animate_us_dx4(self):
        self.play(
            GrowArrow(self.dx4_arrow),
            FadeIn(self.dx4_text),
        )
        self.wait(0.5)
        self.play(FadeIn(self.no_sweep_rect, self.no_sweep))
        self.wait(0.5)
        self.play(FadeIn(self.dx4_us_heat))
        self.wait(1)
        self.play(FadeOut(self.no_sweep_rect, self.no_sweep))

    def define_us_dx3_dw2(self):
        self.dx3_us_scale = USBox("\\div 45", self.matmul_b, self.base_colors["dx"], UP)
        self.dx3_us_heat = ExpHeat(
            self.us_df["x3_grad"],
            "\\nabla_{x_3}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 8 * self.heat_width, LEFT)
        self.sep_grad = Tex("Separate grad\\_w scale", color=DARKER_GRAY, font_size=38)
        self.sep_grad_rect = SurroundingRectangle(
            self.sep_grad,
            color=self.base_colors["dw"],
            stroke_width=2,
            stroke_opacity=1.0,
            buff=0.125,
        ).move_to(
            self.w2_us_heat.bars[0].get_left() - RIGHT * 0.1 + UP * 0.5 + RIGHT * 0.96,
            RIGHT,
        )
        self.sep_grad.move_to(self.sep_grad_rect)
        self.dw2_us_text = (
            MathTex("\\div \\sqrt{\\mathrm{batch\\_size}}", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.sep_grad_rect)
        )
        self.dw2_us_value = (
            MathTex("\\div 256", color=DARKER_GRAY)
            .scale(0.7)
            .next_to(self.sep_grad_rect)
        )
        self.dw2_us_value.z_index = 3
        self.dw2_us_sq = BackgroundRectangle(
            self.dw2_us_value,
            color=lighten(self.base_colors["dw"]),
            fill_opacity=1.0,
            buff=0.1,
        )
        self.dw2_us_scale = VGroup(self.dw2_us_sq, self.dw2_us_value)
        self.dw2_us_heat = ExpHeat(
            self.us_df["w2_grad"],
            "\\nabla_{w_2}",
            self.base_colors["dw"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 10.5 * self.heat_width, LEFT)

    def animate_us_dx3_dw2(self):
        self.play(FadeIn(self.dx3_us_scale))
        self.play(
            self.dx3_us_scale.animate.move_to(self.matmul_b).shift(
                DOWN * 0.52 + LEFT * 0.52
            ),
            FadeOut(self.matmul_b.subtext),
        )
        self.wait(0.3)
        self.play(GrowArrow(self.dx3_arrow), FadeIn(self.dx3_text, self.dx3_us_heat))
        self.wait(1)
        self.play(FadeIn(self.sep_grad_rect, self.sep_grad))
        self.wait(0.5)
        self.play(FadeIn(self.dw2_us_text))
        self.wait(1.5)
        self.play(FadeOut(self.dw2_us_text, shift=DOWN * 0.2, run_time=0.5))
        self.play(FadeIn(self.dw2_us_value, shift=UP * 0.2, run_time=0.5))
        self.wait(0.5)
        self.play(FadeIn(self.dw2_us_sq))
        self.play(self.dw2_us_scale.animate.shift(UP * 1.2))
        self.wait(0.75)
        self.play(
            FadeOut(self.sep_grad_rect, self.sep_grad),
            GrowArrow(self.dw2_arrow),
            FadeIn(self.dw2_us_heat, self.dw2_text),
        )
        self.wait(1)

    def define_us_dx2(self):
        self.dx2_us_scale = USBox("\\times 2.5", self.gelu, self.base_colors["dx"], UP)
        self.dx2_us_heat = ExpHeat(
            self.us_df["x2_grad"],
            "\\nabla_{x_2}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 5.5 * self.heat_width, LEFT)

    def animate_us_dx2(self):
        self.play(FadeIn(self.dx2_us_scale))
        self.play(self.dx2_us_scale.animate.shift(DOWN * 1.05))
        self.wait(0.3)
        self.play(GrowArrow(self.dx2_arrow), FadeIn(self.dx2_text, self.dx2_us_heat))
        self.wait(1)

    def define_us_dx1_dw1(self):
        self.dx1_us_scale = USBox("\\div 45", self.matmul_a, self.base_colors["dx"], UP)
        self.dx1_us_heat = ExpHeat(
            self.us_df["x1_grad"],
            "\\nabla_{x_1}",
            self.base_colors["dx"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 3 * self.heat_width, LEFT)
        self.dw1_us_scale = USBox(
            "\\div 256", self.matmul_a, self.base_colors["dw"], DOWN * 1.05 + LEFT * 0.9
        )
        self.dw1_us_heat = ExpHeat(
            self.us_df["w1_grad"],
            "\\nabla_{w_1}",
            self.base_colors["dw"],
            self.heat_width,
        ).move_to(self.heat_base_line + RIGHT * 1 * self.heat_width, LEFT)

    def animate_us_dx1_dw1(self):
        self.play(FadeIn(self.dx1_us_scale))
        self.play(
            self.dx1_us_scale.animate.move_to(self.matmul_a).shift(
                DOWN * 0.52 + RIGHT * 0.52
            ),
            FadeOut(self.matmul_a.subtext),
        )
        self.wait(0.3)
        self.play(GrowArrow(self.dx1_arrow), FadeIn(self.dx1_text, self.dx1_us_heat))
        self.wait(1)
        self.play(FadeIn(self.dw1_us_scale))
        self.wait(0.3)
        self.play(GrowArrow(self.dw1_arrow), FadeIn(self.dw1_text, self.dw1_us_heat))
        self.wait(1.5)

    def animate_fade_out_us_diagram(self):
        self.matmul_a.remove(self.matmul_a.subtext)
        self.matmul_b.remove(self.matmul_b.subtext)
        self.play(
            FadeOut(
                self.ffn_diagram,
                self.x1_text,
                self.x1_arrow,
                self.x1_sigma,
                self.w1_text,
                self.w1_arrow,
                self.w1_sigma_us,
                self.x2_text,
                self.x2_arrow,
                self.x2_us_scale,
                self.x3_text,
                self.x3_arrow,
                self.x3_us_scale,
                self.x4_text,
                self.x4_arrow,
                self.x4_us_scale,
                self.w2_text,
                self.w2_arrow,
                self.w2_sigma_us,
                self.dx1_text,
                self.dx1_arrow,
                self.dw1_text,
                self.dw1_arrow,
                self.dw1_us_scale,
                self.dx1_us_scale,
                self.dx2_text,
                self.dx2_arrow,
                self.dx2_us_scale,
                self.dx3_text,
                self.dx3_arrow,
                self.dx3_us_scale,
                self.dw2_text,
                self.dw2_arrow,
                self.dw2_us_scale,
                self.dx4_text,
                self.dx4_arrow,
            )
        )

    def define_result_text(self):
        self.result_prompt = (
            Tex("Result: ", color=self.base_colors["dx"])
            .scale(1.4)
            .move_to(self.problem_prompt, RIGHT)
            .shift(DOWN * 0.7)
        )
        self.result_txt = (
            Tex(
                "out-of-the box\n\nlow-precision training",
                color=DARKER_GRAY,
                tex_environment="flushleft",
            ).scale(1.4)
        ).next_to(self.result_prompt, RIGHT, buff=0.4)

    def animate_result_text(self):
        self.play(FadeIn(self.result_prompt, self.result_txt, run_time=0.7))
        self.wait(2.5)

    def define(self):
        self.define_confidential()
        self.define_opening_text()
        self.define_ffn_diagram()
        self.define_arrows()
        self.define_reg_sigmas()
        self.define_histograms_base()
        self.define_histogram_fp_lines()
        self.define_histogram_transform_w1()
        self.define_histogram_transform_x1()
        self.define_reg_x2()
        self.define_reg_x3()
        self.define_reg_x4_w2()
        self.define_loss_scaling_dx4()
        self.define_reg_dx3_dw2()
        self.define_reg_dx2()
        self.define_reg_dx1_dw1()
        self.define_reg_heats()
        self.define_us_text()
        self.define_us_matmul_a()
        self.define_us_x2()
        self.define_us_x3()
        self.define_us_w2_x4()
        self.define_us_dx4()
        self.define_us_dx3_dw2()
        self.define_us_dx2()
        self.define_us_dx1_dw1()
        self.define_result_text()

    def animate(self):
        self.animate_confidential()
        self.animate_opening_text()
        self.animate_ffn_diagram()
        self.animate_x1_w1()
        self.animate_histograms_base()
        self.animate_histogram_fp_lines()
        self.animate_histogram_transform_w1()
        self.animate_histogram_transform_x1()
        self.animate_reg_x2()
        self.animate_reg_x3()
        self.animate_reg_x4_w2()
        self.animate_loss_scaling_dx4()
        self.animate_reg_dx3_dw2()
        self.animate_reg_dx2()
        self.animate_reg_dx1_dw1()
        self.animate_fade_out_reg_diagram()
        self.animate_us_text()
        self.animate_fade_out_reg_heats()
        self.animate_fade_out_us_text()
        self.animate_ffn_diagram()
        self.animate_x1_w1()
        self.animate_us_matmul_a()
        self.animate_us_x2()
        self.animate_us_x3()
        self.animate_us_w2_x4()
        self.animate_us_dx4()
        self.animate_us_dx3_dw2()
        self.animate_us_dx2()
        self.animate_us_dx1_dw1()
        self.animate_fade_out_us_diagram()
        self.animate_result_text()

    def construct(self):
        self.setup()
        self.define()
        self.animate()
