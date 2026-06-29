// Command aurachrome is a Bubble Tea wizard front-end for the Aurachrome
// false-color film engine. It collects input/output and the export options, then
// drives the Python engine (via `poetry run aurachrome`) and renders its JSON
// progress stream live.
//
// Build:  go build -o aurachrome ./tui      (from the repo root)
// Run:    ./aurachrome                       (needs `poetry` + the engine)
package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "aurachrome:", err)
		os.Exit(1)
	}
}
