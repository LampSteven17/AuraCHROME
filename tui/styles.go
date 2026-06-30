package main

import "github.com/charmbracelet/lipgloss"

// Aerochrome false-color palette: IR magenta, sky cyan, sun amber.
var (
	cMagenta = lipgloss.Color("#ff4fd8")
	cCyan    = lipgloss.Color("#36e3ff")
	cAmber   = lipgloss.Color("#ffb454")
	cDim     = lipgloss.Color("#6b7280")
	cFg      = lipgloss.Color("#e8e8ea")
	cGreen   = lipgloss.Color("#7cffb2")
	cRed     = lipgloss.Color("#ff6b6b")
	cInk     = lipgloss.Color("#0b0b0e")

	titleStyle  = lipgloss.NewStyle().Foreground(cMagenta).Bold(true)
	taglineStyle = lipgloss.NewStyle().Foreground(cCyan).Italic(true)
	stepStyle   = lipgloss.NewStyle().Foreground(cAmber).Bold(true)
	labelStyle  = lipgloss.NewStyle().Foreground(cCyan).Bold(true)
	valStyle    = lipgloss.NewStyle().Foreground(cFg)
	dimStyle    = lipgloss.NewStyle().Foreground(cDim)
	selChip     = lipgloss.NewStyle().Foreground(cInk).Background(cAmber).Bold(true).Padding(0, 1)
	offChip     = lipgloss.NewStyle().Foreground(cFg).Padding(0, 1)
	cursorStyle = lipgloss.NewStyle().Foreground(cMagenta).Bold(true)
	boxStyle    = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cMagenta).Padding(1, 3)
	errStyle    = lipgloss.NewStyle().Foreground(cRed).Bold(true)
	okStyle     = lipgloss.NewStyle().Foreground(cGreen).Bold(true)
	helpStyle   = lipgloss.NewStyle().Foreground(cDim)
)

func rgbHex(r, g, b int) string {
	const hexdigits = "0123456789abcdef"
	clamp := func(v int) int {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return v
	}
	r, g, b = clamp(r), clamp(g), clamp(b)
	bs := []byte{'#',
		hexdigits[r>>4], hexdigits[r&0xf],
		hexdigits[g>>4], hexdigits[g&0xf],
		hexdigits[b>>4], hexdigits[b&0xf],
	}
	return string(bs)
}
