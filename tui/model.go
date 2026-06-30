package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// splashTickMsg drives the animated title card. The ticker re-arms only while
// the splash is on screen (see Update), so it costs nothing past the first step.
type splashTickMsg struct{}

func splashTick() tea.Cmd {
	return tea.Tick(time.Second/15, func(time.Time) tea.Msg { return splashTickMsg{} })
}

type step int

const (
	stepSplash step = iota
	stepInput
	stepOutput
	stepOptions
	stepConfirm
	stepRun
	stepDone
)

var (
	looks       = []string{"classic", "punchy", "muted", "portrait", "redscale", "hie", "all"}
	formats     = []string{"tiff16", "jpeg", "both"}
	resLabels   = []string{"full", "4096px", "2048px"}
	resPx       = []int{0, 4096, 2048} // long-edge px; 0 = full resolution
	irModes     = []string{"auto", "neural", "grvi"}
	devices     = []string{"auto", "cpu", "gpu"}
	grainLevels = []string{"off", "subtle", "standard", "heavy"} // chromatic grain amount
	rawExts     = map[string]bool{".arw": true, ".cr2": true, ".cr3": true, ".nef": true,
		".raf": true, ".rw2": true, ".dng": true, ".orf": true, ".raw": true,
		".tif": true, ".tiff": true, ".png": true, ".jpg": true, ".jpeg": true}
)

// lastOpt is the index of the final selectable row in the Options step.
const lastOpt = 6

type model struct {
	step          step
	width, height int
	splashPhase   int
	picking       bool // native folder dialog is open
	typing        bool // input step: manual-path fallback field is shown

	gpuAvail   bool   // a CUDA device is present (from the engine probe)
	gpuName    string // its display name
	gpuProbed  bool   // probe has reported back
	confirmMsg string // shown on stepConfirm

	inInput  textinput.Model
	inOutput textinput.Model

	optCursor int
	lookIdx   int
	formatIdx int
	resIdx    int
	grainIdx  int
	irIdx     int
	devIdx    int
	jobs      int

	prog     progress.Model
	spin     spinner.Model
	engCh    chan tea.Msg
	cancel   context.CancelFunc
	total    int
	done     int
	device   string
	ir       string
	outdir   string
	warn     string
	logLines []string

	err      error
	finished bool
}

func initialModel() model {
	in := textinput.New()
	in.Placeholder = "~/shoot/raws  (folder of RAWs, or a single file)"
	in.CharLimit = 1024
	in.Width = 56

	out := textinput.New()
	out.Placeholder = "defaults to <input>/aurachrome_tif"
	out.CharLimit = 1024
	out.Width = 56

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(cMagenta)

	return model{
		step:     stepSplash,
		inInput:  in,
		inOutput: out,
		grainIdx: indexOf(grainLevels, "standard"), // the per-preset tuned default
		devIdx:   indexOf(devices, "auto"),         // auto picks GPU when present, else CPU
		jobs:     4,
		prog:     progress.New(progress.WithDefaultGradient()),
		spin:     sp,
	}
}

func (m model) Init() tea.Cmd { return tea.Batch(textinput.Blink, splashTick(), probeGPU()) }

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		w := msg.Width - 16
		if w > 64 {
			w = 64
		}
		if w < 20 {
			w = 20
		}
		m.prog.Width = w
		return m, nil

	case tea.KeyMsg:
		if msg.String() == "ctrl+c" {
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		}
		return m.handleKey(msg)

	// ---- engine streaming ----
	case engReadyMsg:
		m.engCh = msg.ch
		m.cancel = msg.cancel
		return m, waitForEngine(m.engCh)
	case engStartMsg:
		m.total, m.device, m.ir, m.outdir = msg.total, msg.device, msg.ir, msg.outdir
		m.warn = msg.warning
		return m, waitForEngine(m.engCh)
	case engImageMsg:
		m.done = msg.done
		m.total = msg.total
		line := okStyle.Render("✓ ") + valStyle.Render(msg.file) +
			dimStyle.Render("  ("+msg.preset+")")
		m.logLines = append(m.logLines, line)
		if len(m.logLines) > 8 {
			m.logLines = m.logLines[len(m.logLines)-8:]
		}
		return m, waitForEngine(m.engCh)
	case engDoneMsg:
		m.finished = true
		m.outdir = msg.outdir
		m.step = stepDone
		return m, waitForEngine(m.engCh)
	case engErrMsg:
		m.err = msg.err
		m.step = stepDone
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd

	case splashTickMsg:
		if m.step != stepSplash {
			return m, nil // stop animating once we leave the splash
		}
		m.splashPhase++
		return m, splashTick()

	case pickedMsg:
		m.picking = false
		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}
		if msg.path != "" {
			m.inInput.SetValue(msg.path)
			m.err = nil
			m.typing = false
			m.inInput.Blur()
		}
		return m, nil

	case gpuProbeMsg:
		m.gpuProbed = true
		m.gpuAvail = msg.avail
		m.gpuName = msg.name
		return m, nil
	}

	// text inputs own other messages while focused
	var cmd tea.Cmd
	switch m.step {
	case stepInput:
		m.inInput, cmd = m.inInput.Update(msg)
	case stepOutput:
		m.inOutput, cmd = m.inOutput.Update(msg)
	}
	return m, cmd
}

func (m model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := msg.String()
	switch m.step {
	case stepSplash:
		m.step = stepInput
		m.inInput.Focus()
		return m, textinput.Blink

	case stepInput:
		// Manual-path fallback (revealed with "t", e.g. on a headless host).
		if m.typing {
			switch key {
			case "enter":
				return m.advanceFromInput()
			case "esc":
				m.typing = false
				m.inInput.Blur()
				return m, nil
			default:
				var cmd tea.Cmd
				m.inInput, cmd = m.inInput.Update(msg)
				return m, cmd
			}
		}
		// Picker-first: enter opens the native dialog; once a folder is chosen,
		// enter continues. No typing required.
		switch key {
		case "enter":
			if m.inputReady() {
				return m.advanceFromInput()
			}
			return m.openPicker()
		case "r", "c", "ctrl+o":
			return m.openPicker()
		case "t":
			m.typing = true
			m.inInput.Focus()
			return m, textinput.Blink
		}
		return m, nil

	case stepOutput:
		switch key {
		case "enter":
			if strings.TrimSpace(m.inOutput.Value()) == "" {
				m.inOutput.SetValue(defaultOutput(expand(m.inInput.Value())))
			}
			m.step = stepOptions
			m.inOutput.Blur()
			return m, nil
		case "esc":
			m.step = stepInput
			m.inOutput.Blur()
			m.inInput.Focus()
			return m, textinput.Blink
		default:
			var cmd tea.Cmd
			m.inOutput, cmd = m.inOutput.Update(msg)
			return m, cmd
		}

	case stepOptions:
		switch key {
		case "up", "k":
			m.optCursor = m.nextRow(m.optCursor, -1)
		case "down", "j", "tab":
			m.optCursor = m.nextRow(m.optCursor, 1)
		case "left", "h":
			m.adjust(-1)
		case "right", "l":
			m.adjust(1)
		case "esc":
			m.step = stepOutput
			m.inOutput.Focus()
			return m, textinput.Blink
		case "enter":
			if msg, ok := m.cpuConfirm(); ok {
				m.confirmMsg = msg
				m.step = stepConfirm
				return m, nil
			}
			return m.startRender()
		}
		return m, nil

	case stepConfirm:
		switch key {
		case "y", "Y", "enter":
			return m.startRender()
		case "n", "N", "esc":
			m.step = stepOptions
			return m, nil
		}
		return m, nil

	case stepRun:
		if key == "q" {
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		}
		return m, nil

	case stepDone:
		switch key {
		case "n":
			nm := initialModel()
			nm.width, nm.height = m.width, m.height
			nm.prog.Width = m.prog.Width
			nm.step = stepInput
			nm.inInput.SetValue(m.inInput.Value())
			nm.inInput.Focus()
			return nm, textinput.Blink
		case "q", "enter", "esc":
			return m, tea.Quit
		}
	}
	return m, nil
}

// inputReady reports whether the chosen folder holds at least one usable file.
func (m model) inputReady() bool {
	v := strings.TrimSpace(m.inInput.Value())
	return v != "" && countInputs(expand(v)) > 0
}

// advanceFromInput validates the selected path and moves to the Output step.
func (m model) advanceFromInput() (tea.Model, tea.Cmd) {
	path := expand(strings.TrimSpace(m.inInput.Value()))
	if countInputs(path) == 0 {
		m.err = fmt.Errorf("no RAW/image files found at %q", path)
		return m, nil
	}
	m.err = nil
	m.typing = false
	if strings.TrimSpace(m.inOutput.Value()) == "" {
		m.inOutput.SetValue(defaultOutput(path))
	}
	m.step = stepOutput
	m.inInput.Blur()
	m.inOutput.Focus()
	return m, textinput.Blink
}

// openPicker launches the native folder dialog (unless one is already open).
func (m model) openPicker() (tea.Model, tea.Cmd) {
	if m.picking {
		return m, nil
	}
	m.picking = true
	m.err = nil
	return m, pickFolder(expand(strings.TrimSpace(m.inInput.Value())))
}

// startRender transitions to the render step and launches the engine.
func (m model) startRender() (tea.Model, tea.Cmd) {
	m.step = stepRun
	m.done, m.total, m.err, m.finished = 0, 0, nil, false
	m.warn = ""
	m.logLines = nil
	return m, tea.Batch(startEngine(m.config()), m.spin.Tick)
}

// cpuConfirm decides whether to interrupt with a "really render on CPU?" prompt.
// GPU is the recommended device, so we confirm when the user is about to leave
// it on the table: CPU picked while a GPU exists, or GPU picked with none found.
// Until the probe reports back we don't second-guess the user.
func (m model) cpuConfirm() (string, bool) {
	if !m.gpuProbed {
		return "", false
	}
	switch devices[m.devIdx] {
	case "cpu":
		if m.gpuAvail {
			return "A GPU (" + m.gpuName + ") is available and renders much faster. Continue on CPU anyway?", true
		}
	case "gpu":
		if !m.gpuAvail {
			return "GPU selected, but no CUDA device was found — this will render on CPU. Continue?", true
		}
	}
	return "", false
}

// jobsRowIdx is the Options row controlling CPU worker count.
const jobsRowIdx = 6

// rowDisabled reports whether an Options row is inert. CPU worker count only
// applies when the device is explicitly CPU; on auto (GPU when present) and gpu
// the Jobs row is greyed out and skipped.
func (m model) rowDisabled(row int) bool {
	return row == jobsRowIdx && devices[m.devIdx] != "cpu"
}

// nextRow steps the cursor in direction d, skipping disabled rows and stopping
// at the ends (no wrap). Returns cur if no enabled row lies that way.
func (m model) nextRow(cur, d int) int {
	for n := cur + d; n >= 0 && n <= lastOpt; n += d {
		if !m.rowDisabled(n) {
			return n
		}
	}
	return cur
}

func (m *model) adjust(d int) {
	switch m.optCursor {
	case 0:
		m.lookIdx = wrap(m.lookIdx+d, len(looks))
	case 1:
		m.formatIdx = wrap(m.formatIdx+d, len(formats))
	case 2:
		m.resIdx = wrap(m.resIdx+d, len(resLabels))
	case 3:
		m.grainIdx = wrap(m.grainIdx+d, len(grainLevels))
	case 4:
		m.irIdx = wrap(m.irIdx+d, len(irModes))
	case 5:
		m.devIdx = wrap(m.devIdx+d, len(devices))
	case 6:
		m.jobs += d
		if m.jobs < 1 {
			m.jobs = 1
		}
		if m.jobs > 16 {
			m.jobs = 16
		}
	}
}

func (m model) config() runConfig {
	return runConfig{
		input:    expand(strings.TrimSpace(m.inInput.Value())),
		output:   expand(strings.TrimSpace(m.inOutput.Value())),
		look:     looks[m.lookIdx],
		format:   formats[m.formatIdx],
		longedge: resPx[m.resIdx],
		grain:    grainLevels[m.grainIdx],
		ir:       irModes[m.irIdx],
		device:   devices[m.devIdx],
		jobs:     m.jobs,
	}
}

// ---- helpers ----
func indexOf(xs []string, want string) int {
	for i, x := range xs {
		if x == want {
			return i
		}
	}
	return 0
}

func wrap(i, n int) int {
	if i < 0 {
		return n - 1
	}
	if i >= n {
		return 0
	}
	return i
}

func expand(p string) string {
	if strings.HasPrefix(p, "~") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}

func defaultOutput(input string) string {
	info, err := os.Stat(input)
	if err == nil && info.IsDir() {
		return filepath.Join(input, "aurachrome_out")
	}
	return filepath.Join(filepath.Dir(input), "aurachrome_out")
}

func countInputs(path string) int {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	if !info.IsDir() {
		if rawExts[strings.ToLower(filepath.Ext(path))] {
			return 1
		}
		return 0
	}
	entries, _ := os.ReadDir(path)
	n := 0
	for _, e := range entries {
		if !e.IsDir() && rawExts[strings.ToLower(filepath.Ext(e.Name()))] {
			n++
		}
	}
	return n
}
