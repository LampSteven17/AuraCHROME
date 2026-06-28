// Command aurachrome-tui is a Bubble Tea wizard front-end for the Aurachrome
// false-color film engine. It collects input/output/look/grain/device, then
// drives the Python `aurachrome` CLI (via `poetry run`) and renders its
// JSON progress stream live.
//
// Build:  go build -o aurachrome-tui ./tui      (from the repo root)
// Run:    ./aurachrome-tui                       (needs `poetry` + the engine)
package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "aurachrome-tui:", err)
		os.Exit(1)
	}
}
