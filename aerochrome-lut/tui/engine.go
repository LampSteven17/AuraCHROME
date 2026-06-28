package main

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

// ---- engine invocation -----------------------------------------------------

// runConfig is everything the wizard collected.
type runConfig struct {
	input  string
	output string
	look   string // classic|punchy|muted|portrait|all
	grain  bool
	device string // auto|cpu|gpu
	jobs   int
}

// engineArgs builds the CLI args for `aurachrome` from the wizard config.
func (c runConfig) args() []string {
	a := []string{"-i", c.input, "-o", c.output, "--preset", c.look, "--progress-json"}
	if !c.grain {
		a = append(a, "--no-grain")
	}
	switch c.device {
	case "cpu":
		a = append(a, "--cpu")
	case "gpu":
		a = append(a, "--gpu")
	}
	a = append(a, "--jobs", strconv.Itoa(c.jobs))
	return a
}

// engineCommand resolves how to launch the Python engine. Override the whole
// command with AURACHROME_ENGINE ("poetry run aurachrome"), and the working dir
// with AURACHROME_REPO. Defaults to `poetry run aurachrome` from the repo root.
func engineCommand(cfg runConfig) (*exec.Cmd, context.CancelFunc) {
	base := []string{"poetry", "run", "aurachrome"}
	if env := strings.TrimSpace(os.Getenv("AURACHROME_ENGINE")); env != "" {
		base = strings.Fields(env)
	}
	full := append(append([]string{}, base[1:]...), cfg.args()...)
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, base[0], full...)
	cmd.Dir = repoRoot()
	return cmd, cancel
}

// defaultRepo is baked in at build time (`-ldflags -X main.defaultRepo=...`, see
// the Makefile) so an installed binary finds the engine from anywhere.
var defaultRepo string

// repoRoot resolves the engine's working dir: $AURACHROME_REPO, then an upward
// search for pyproject.toml (when run inside the tree), then the baked-in path,
// then the cwd.
func repoRoot() string {
	if env := strings.TrimSpace(os.Getenv("AURACHROME_REPO")); env != "" {
		return env
	}
	if dir, err := os.Getwd(); err == nil {
		for {
			if _, err := os.Stat(filepath.Join(dir, "pyproject.toml")); err == nil {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	if defaultRepo != "" {
		return defaultRepo
	}
	cwd, _ := os.Getwd()
	return cwd
}

// ---- streaming messages ----------------------------------------------------

type engStartMsg struct {
	total  int
	device string
	outdir string
}
type engImageMsg struct {
	done, total int
	file, preset string
}
type engDoneMsg struct{ outdir string }
type engErrMsg struct{ err error }
type engReadyMsg struct {
	ch     chan tea.Msg
	cancel context.CancelFunc
}

// startEngine launches the process and a reader goroutine that turns each JSON
// progress line into a tea.Msg on a channel.
func startEngine(cfg runConfig) tea.Cmd {
	return func() tea.Msg {
		cmd, cancel := engineCommand(cfg)
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			cancel()
			return engErrMsg{err}
		}
		var stderr strings.Builder
		cmd.Stderr = &stderr
		if err := cmd.Start(); err != nil {
			cancel()
			return engErrMsg{err}
		}
		ch := make(chan tea.Msg, 128)
		go func() {
			sc := bufio.NewScanner(stdout)
			sc.Buffer(make([]byte, 1<<20), 1<<20)
			for sc.Scan() {
				var ev map[string]any
				if json.Unmarshal(sc.Bytes(), &ev) != nil {
					continue
				}
				switch ev["event"] {
				case "start":
					ch <- engStartMsg{total: toInt(ev["total"]), device: toStr(ev["device"]), outdir: toStr(ev["outdir"])}
				case "image":
					ch <- engImageMsg{done: toInt(ev["done"]), total: toInt(ev["total"]), file: toStr(ev["file"]), preset: toStr(ev["preset"])}
				case "done":
					ch <- engDoneMsg{outdir: toStr(ev["outdir"])}
				}
			}
			if err := cmd.Wait(); err != nil {
				ch <- engErrMsg{err: &engineError{err: err, stderr: strings.TrimSpace(stderr.String())}}
			}
			close(ch)
		}()
		return engReadyMsg{ch: ch, cancel: cancel}
	}
}

// waitForEngine blocks for the next message from the engine channel.
func waitForEngine(ch chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return nil
		}
		return msg
	}
}

type engineError struct {
	err    error
	stderr string
}

func (e *engineError) Error() string {
	if e.stderr != "" {
		tail := e.stderr
		if len(tail) > 400 {
			tail = "…" + tail[len(tail)-400:]
		}
		return e.err.Error() + "\n" + tail
	}
	return e.err.Error()
}

func toInt(v any) int {
	if f, ok := v.(float64); ok {
		return int(f)
	}
	return 0
}
func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
