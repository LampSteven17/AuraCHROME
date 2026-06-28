package main

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Compact block wordmark for the splash. Kept ASCII-only so it renders
// everywhere; colored with the magenta->cyan gradient at draw time.
var wordmark = []string{
	" ___  _   _ ____  ___   ___ _   _ ____  ___  __  __ ____ ",
	"/ _ \\| | | |  _ \\/ _ \\ / __| | | |  _ \\/ _ \\|  \\/  | __ )",
	"| |_| | | | | |_) | |_| | |  | |_| | |_) | | | | |\\/| |  _ \\",
	"|  _  | |_| |  _ <|  _  | |__|  _  |  _ <| |_| | |  | | |_) |",
	"|_| |_|\\___/|_| \\_\\_| |_|\\___|_| |_|_| \\_\\\\___/|_|  |_|____/",
}

func splashView(width int) string {
	var b strings.Builder
	for _, line := range wordmark {
		b.WriteString(gradientText(line))
		b.WriteString("\n")
	}
	art := b.String()
	tag := taglineStyle.Render("false-color film engine")
	hint := dimStyle.Render("Aerochrome / EIR  ·  RAW → 16-bit TIFF  ·  CPU / CUDA")
	press := helpStyle.Render("press any key to begin")

	body := lipgloss.JoinVertical(lipgloss.Center, art, "", tag, hint, "", press)
	framed := boxStyle.Render(body)
	if width <= 0 {
		return framed
	}
	return lipgloss.Place(width, lipgloss.Height(framed)+2, lipgloss.Center, lipgloss.Center, framed)
}
