package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

func (m model) View() string {
	switch m.step {
	case stepSplash:
		return splashView(m.width, m.splashPhase)
	case stepInput:
		return m.frame("Input", 1, m.inputView())
	case stepOutput:
		return m.frame("Output", 2, m.outputView())
	case stepOptions:
		return m.frame("Options", 3, m.optionsView())
	case stepConfirm:
		return m.frame("Confirm", 3, m.confirmView())
	case stepRun:
		return m.frame("Render", 4, m.runView())
	case stepDone:
		return m.frame("Done", 4, m.doneView())
	}
	return ""
}

// frame wraps step content with a header (wordmark-lite + step counter) and footer.
func (m model) frame(title string, n int, body string) string {
	head := titleStyle.Render("aurachrome") + "  " +
		stepStyle.Render(fmt.Sprintf("step %d/4", n)) + "  " +
		labelStyle.Render(title)
	help := helpStyle.Render(m.footer())
	content := lipgloss.JoinVertical(lipgloss.Left, head, "", body, "", help)
	framed := boxStyle.Render(content)
	if m.width <= 0 {
		return framed
	}
	return lipgloss.Place(m.width, lipgloss.Height(framed), lipgloss.Center, lipgloss.Top, framed)
}

func (m model) footer() string {
	switch m.step {
	case stepInput:
		if m.typing {
			return "enter: use path   ·   esc: back   ·   ctrl+c: quit"
		}
		return "enter: choose folder / continue   ·   r: re-choose   ·   ctrl+c: quit"
	case stepConfirm:
		return "y: render on CPU   ·   n/esc: back"
	case stepOutput:
		return "enter: next   ·   esc: back   ·   ctrl+c: quit"
	case stepOptions:
		return "↑/↓: field   ·   ←/→: change   ·   enter: render   ·   esc: back"
	case stepRun:
		return "rendering…   ·   q: cancel"
	case stepDone:
		return "n: new batch   ·   q: quit"
	}
	return "ctrl+c: quit"
}

func (m model) inputView() string {
	title := labelStyle.Render("Select the folder with your photos") + "\n\n"

	// Manual-path fallback field.
	if m.typing {
		s := title + m.inInput.View() + "\n" +
			dimStyle.Render("enter: use this path   ·   esc: back to the picker")
		if m.err != nil {
			s += "\n\n" + errStyle.Render("x "+m.err.Error())
		}
		return s
	}

	// Dialog is open.
	if m.picking {
		return title + stepStyle.Render("opening folder picker…") + "\n" +
			dimStyle.Render("a window should appear over the terminal — pick a folder there")
	}

	// A folder is chosen: show it + file count + how to proceed.
	if v := strings.TrimSpace(m.inInput.Value()); v != "" {
		s := title + valStyle.Render(v) + "\n"
		if n := countInputs(expand(v)); n > 0 {
			s += okStyle.Render(fmt.Sprintf("found %d file(s)", n)) + "\n\n" +
				stepStyle.Render("press enter to continue") +
				dimStyle.Render("    ·    r: choose a different folder")
		} else {
			s += errStyle.Render("no photos found in this folder") + "\n\n" +
				dimStyle.Render("r: choose a different folder")
		}
		return s
	}

	// Nothing chosen yet — the default landing state.
	s := title + stepStyle.Render("press enter") +
		dimStyle.Render(" to open the folder picker")
	if m.err != nil {
		s += "\n\n" + errStyle.Render("x "+m.err.Error()) +
			"\n" + dimStyle.Render("t: type a path manually instead")
	}
	return s
}

func (m model) confirmView() string {
	return stepStyle.Render("! "+m.confirmMsg) + "\n\n" +
		labelStyle.Render("y") + dimStyle.Render(": render on CPU    ") +
		labelStyle.Render("n") + dimStyle.Render(": go back and change the device")
}

func (m model) outputView() string {
	return labelStyle.Render("Write the output to:") + "\n\n" + m.inOutput.View() +
		"\n\n" + dimStyle.Render("sRGB · format chosen next · flows into Lightroom → Photoshop")
}

func (m model) optionsView() string {
	rows := []string{
		m.choiceRow(0, "Look", looks, m.lookIdx),
		m.choiceRow(1, "Format", formats, m.formatIdx),
		m.choiceRow(2, "Size", resLabels, m.resIdx),
		m.choiceRow(3, "Grain", grainLevels, m.grainIdx),
		m.choiceRow(4, "IR", irModes, m.irIdx),
		m.choiceRow(5, "Device", devices, m.devIdx),
		m.jobsRow(6),
	}
	return strings.Join(rows, "\n")
}

func (m model) cursor(row int) string {
	if m.optCursor == row {
		return cursorStyle.Render("❯ ")
	}
	return "  "
}

func (m model) choiceRow(row int, label string, opts []string, idx int) string {
	var b strings.Builder
	b.WriteString(m.cursor(row))
	b.WriteString(labelStyle.Render(fmt.Sprintf("%-8s", label)))
	for i, o := range opts {
		if i == idx {
			b.WriteString(selChip.Render(o))
		} else {
			b.WriteString(offChip.Render(o))
		}
	}
	return b.String()
}

func (m model) jobsRow(row int) string {
	if m.rowDisabled(row) {
		note := "· auto-managed — set Device to cpu to tune workers"
		if devices[m.devIdx] == "gpu" {
			note = "· skipped — GPU renders on-device"
		}
		return "  " + dimStyle.Render(fmt.Sprintf("%-8s", "Jobs")) +
			dimStyle.Render(fmt.Sprintf("%d", m.jobs)) + "  " +
			dimStyle.Render(note)
	}
	return m.cursor(row) + labelStyle.Render(fmt.Sprintf("%-8s", "Jobs")) +
		selChip.Render(fmt.Sprintf("%d", m.jobs)) + "  " + dimStyle.Render("CPU workers")
}

func (m model) runView() string {
	frac := 0.0
	if m.total > 0 {
		frac = float64(m.done) / float64(m.total)
	}
	head := m.spin.View() + " " + valStyle.Render("rendering")
	if m.device != "" {
		head += dimStyle.Render("  [" + m.device + "]")
	}
	if m.ir != "" {
		head += dimStyle.Render("  IR: " + m.ir)
	}
	bar := m.prog.ViewAs(frac)
	count := dimStyle.Render(fmt.Sprintf("%d / %d images", m.done, m.total))
	log := strings.Join(m.logLines, "\n")
	rows := []string{head}
	if m.warn != "" {
		rows = append(rows, stepStyle.Render("! "+m.warn))
	}
	rows = append(rows, "", bar, count, "", log)
	return lipgloss.JoinVertical(lipgloss.Left, rows...)
}

func (m model) doneView() string {
	if m.err != nil {
		return errStyle.Render("× conversion failed") + "\n\n" +
			dimStyle.Render(m.err.Error())
	}
	return okStyle.Render(fmt.Sprintf("✓ rendered %d image(s)", m.done)) + "\n\n" +
		labelStyle.Render("output: ") + valStyle.Render(m.outdir) + "\n\n" +
		dimStyle.Render("import the TIFFs into Lightroom → Photoshop")
}
