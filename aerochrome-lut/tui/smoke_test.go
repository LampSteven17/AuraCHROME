package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// step concrete model forward, keeping the concrete type.
func advance(t *testing.T, m model, msg tea.Msg) model {
	t.Helper()
	nm, _ := m.Update(msg)
	mm, ok := nm.(model)
	if !ok {
		t.Fatalf("Update returned non-model %T", nm)
	}
	return mm
}

func key(s string) tea.KeyMsg {
	if len(s) == 1 {
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case "right":
		return tea.KeyMsg{Type: tea.KeyRight}
	case "left":
		return tea.KeyMsg{Type: tea.KeyLeft}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
}

func nonEmpty(t *testing.T, m model, label string) {
	t.Helper()
	if strings.TrimSpace(m.View()) == "" {
		t.Fatalf("%s: empty view", label)
	}
}

func TestWizardFlowHeadless(t *testing.T) {
	// a folder with one candidate file so input validation passes
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "shot.tif"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := initialModel()
	m = advance(t, m, tea.WindowSizeMsg{Width: 100, Height: 40})
	nonEmpty(t, m, "splash")

	// splash -> input
	m = advance(t, m, key("x"))
	if m.step != stepInput {
		t.Fatalf("expected stepInput, got %v", m.step)
	}
	m.inInput.SetValue(dir)
	nonEmpty(t, m, "input")

	// input -> output
	m = advance(t, m, key("enter"))
	if m.step != stepOutput {
		t.Fatalf("expected stepOutput, got %v", m.step)
	}
	if !strings.Contains(m.inOutput.Value(), "aurachrome_out") {
		t.Fatalf("output not defaulted: %q", m.inOutput.Value())
	}
	nonEmpty(t, m, "output")

	// output -> options
	m = advance(t, m, key("enter"))
	if m.step != stepOptions {
		t.Fatalf("expected stepOptions, got %v", m.step)
	}
	// exercise option controls. Rows: Look·Format·Size·Grain·IR·Device·Jobs.
	m = advance(t, m, key("right")) // Look -> punchy
	if m.lookIdx != 1 {
		t.Fatalf("look not advanced: %d", m.lookIdx)
	}
	m = advance(t, m, key("down"))  // -> Format
	m = advance(t, m, key("right")) // tiff16 -> jpeg
	m = advance(t, m, key("down"))  // -> Size
	m = advance(t, m, key("right")) // full -> 4096px (adds --longedge 4096)
	m = advance(t, m, key("down"))  // -> Grain
	m = advance(t, m, key("left"))  // toggle grain off
	if m.grain {
		t.Fatal("grain should be off")
	}
	m = advance(t, m, key("down"))  // -> IR
	m = advance(t, m, key("right")) // auto -> neural
	m = advance(t, m, key("down"))  // -> Device
	m = advance(t, m, key("right")) // auto -> cpu
	m = advance(t, m, key("down"))  // -> Jobs
	m = advance(t, m, key("right"))
	if m.jobs != 5 {
		t.Fatalf("jobs not incremented: %d", m.jobs)
	}
	if m.formatIdx != 1 || m.resIdx != 1 || m.irIdx != 1 {
		t.Fatalf("opts: format=%d size=%d ir=%d", m.formatIdx, m.resIdx, m.irIdx)
	}
	nonEmpty(t, m, "options")

	// verify the engine args we'd spawn (without spawning)
	cfg := m.config()
	got := strings.Join(cfg.args(), " ")
	for _, want := range []string{"-i " + dir, "--preset punchy", "--format jpeg",
		"--no-grain", "--longedge 4096", "--cpu", "--jobs 5", "--progress-json"} {
		if !strings.Contains(got, want) {
			t.Fatalf("args missing %q in %q", want, got)
		}
	}
	// IR=neural must NOT force --no-neural (only grvi does)
	if strings.Contains(got, "--no-neural") {
		t.Fatalf("unexpected --no-neural with IR=neural: %q", got)
	}

	// simulate the engine stream into the run view
	m.step = stepRun
	m = advance(t, m, engStartMsg{total: 3, device: "gpu", ir: "neural NIR", outdir: "/out"})
	m = advance(t, m, engImageMsg{done: 1, total: 3, file: "a.tif", preset: "punchy"})
	m = advance(t, m, engImageMsg{done: 2, total: 3, file: "b.tif", preset: "punchy"})
	nonEmpty(t, m, "run")
	m = advance(t, m, engDoneMsg{outdir: "/out"})
	if m.step != stepDone || !m.finished {
		t.Fatalf("expected done/finished, got step=%v finished=%v", m.step, m.finished)
	}
	nonEmpty(t, m, "done-ok")

	// error path renders too
	m.err = errors.New("boom")
	nonEmpty(t, m, "done-err")
}
