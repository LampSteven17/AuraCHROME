package main

import (
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Heavy "ANSI Shadow" block glyphs for the splash wordmark. Each glyph is 6
// rows tall with equal-width rows so they assemble by plain concatenation.
// Rendered in the animated magenta->amber->cyan Aerochrome sheen at draw time.
var shadowGlyphs = map[rune][]string{
	'A': {
		" █████╗ ",
		"██╔══██╗",
		"███████║",
		"██╔══██║",
		"██║  ██║",
		"╚═╝  ╚═╝",
	},
	'U': {
		"██╗   ██╗",
		"██║   ██║",
		"██║   ██║",
		"██║   ██║",
		"╚██████╔╝",
		" ╚═════╝ ",
	},
	'R': {
		"██████╗ ",
		"██╔══██╗",
		"██████╔╝",
		"██╔══██╗",
		"██║  ██║",
		"╚═╝  ╚═╝",
	},
	'C': {
		" ██████╗",
		"██╔════╝",
		"██║     ",
		"██║     ",
		"╚██████╗",
		" ╚═════╝",
	},
	'H': {
		"██╗  ██╗",
		"██║  ██║",
		"███████║",
		"██╔══██║",
		"██║  ██║",
		"╚═╝  ╚═╝",
	},
	'O': {
		" ██████╗ ",
		"██╔═══██╗",
		"██║   ██║",
		"██║   ██║",
		"╚██████╔╝",
		" ╚═════╝ ",
	},
	'M': {
		"███╗   ███╗",
		"████╗ ████║",
		"██╔████╔██║",
		"██║╚██╔╝██║",
		"██║ ╚═╝ ██║",
		"╚═╝     ╚═╝",
	},
	'E': {
		"███████╗",
		"██╔════╝",
		"█████╗  ",
		"██╔══╝  ",
		"███████╗",
		"╚══════╝",
	},
}

// splashView renders the animated title card. phase advances on a timer so the
// color sheen, film-strip bar, border glow, and prompt all breathe.
func splashView(width, phase int) string {
	var art []string
	if width >= 96 {
		art = assemble("AURACHROME")
	} else {
		art = []string{spaced("AURACHROME")}
	}

	title := strings.Join(gradient2D(art, phase), "\n")
	bar := gradientBar(maxRunes(art), phase)
	tag := taglineStyle.Render("false-color film engine")
	spec := dimStyle.Render("Aerochrome / EIR  ·  RAW → 16-bit TIFF  ·  CPU / CUDA")
	press := pulseText("press any key to begin", phase)

	body := lipgloss.JoinVertical(lipgloss.Center, title, bar, "", tag, spec, "", press)

	br, bg, bb := ramp(float64(phase) * 0.02)
	box := boxStyle.BorderForeground(lipgloss.Color(rgbHex(br, bg, bb)))
	framed := box.Render(body)
	if width <= 0 {
		return framed
	}
	return lipgloss.Place(width, lipgloss.Height(framed)+2, lipgloss.Center, lipgloss.Center, framed)
}

// assemble concatenates the 6-row block glyphs for word into 6 lines.
func assemble(word string) []string {
	rows := make([]string, 6)
	for _, r := range word {
		g, ok := shadowGlyphs[r]
		if !ok {
			continue
		}
		for i := 0; i < 6; i++ {
			rows[i] += g[i]
		}
	}
	return rows
}

// spaced letter-spaces a word for the compact (narrow-terminal) fallback.
func spaced(s string) string {
	return strings.Join(strings.Split(s, ""), " ")
}

// gradient2D paints each non-space rune along a diagonal position, offset by
// phase so the magenta->amber->cyan sheen sweeps across the block over time.
func gradient2D(lines []string, phase int) []string {
	h := len(lines)
	w := maxRunes(lines)
	den := float64(w-1) + float64(h-1)
	if den <= 0 {
		den = 1
	}
	shift := float64(phase) * 0.015
	out := make([]string, h)
	for y, l := range lines {
		var b strings.Builder
		for x, r := range []rune(l) {
			if r == ' ' {
				b.WriteByte(' ')
				continue
			}
			t := (float64(x) + float64(y)) / den
			cr, cg, cb := ramp(t*0.85 - shift)
			b.WriteString(paint(string(r), cr, cg, cb))
		}
		out[y] = b.String()
	}
	return out
}

// gradientBar is a flowing color strip (a nod to a film color reference) that
// drifts in sync with the title sheen.
func gradientBar(n, phase int) string {
	if n < 1 {
		n = 1
	}
	shift := float64(phase) * 0.015
	var b strings.Builder
	for i := 0; i < n; i++ {
		cr, cg, cb := ramp(float64(i)/float64(n)*0.85 - shift)
		b.WriteString(paint("━", cr, cg, cb))
	}
	return b.String()
}

// pulseText breathes a line between dim and amber.
func pulseText(s string, phase int) string {
	p := 0.5 + 0.5*math.Sin(float64(phase)*0.3)
	r := lerp(107, 255, p)
	g := lerp(114, 180, p)
	bl := lerp(128, 84, p)
	return paint(s, r, g, bl)
}

// ramp maps u (any real; fractional part used) cyclically through the
// Aerochrome palette: IR magenta -> sun amber -> sky cyan -> back.
func ramp(u float64) (int, int, int) {
	u = u - math.Floor(u)
	stops := [4][3]float64{
		{0xff, 0x4f, 0xd8}, // magenta
		{0xff, 0xb4, 0x54}, // amber
		{0x36, 0xe3, 0xff}, // cyan
		{0xff, 0x4f, 0xd8}, // wrap to magenta
	}
	seg := u * 3
	i := int(seg)
	if i > 2 {
		i = 2
	}
	f := seg - float64(i)
	a, b := stops[i], stops[i+1]
	return lerp(int(a[0]), int(b[0]), f),
		lerp(int(a[1]), int(b[1]), f),
		lerp(int(a[2]), int(b[2]), f)
}

func paint(s string, r, g, b int) string {
	return lipgloss.NewStyle().Foreground(lipgloss.Color(rgbHex(r, g, b))).Bold(true).Render(s)
}

func lerp(a, b int, t float64) int { return int(float64(a) + (float64(b)-float64(a))*t) }

func maxRunes(lines []string) int {
	w := 0
	for _, l := range lines {
		if n := len([]rune(l)); n > w {
			w = n
		}
	}
	return w
}
