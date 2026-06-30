package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"unicode/utf16"

	tea "github.com/charmbracelet/bubbletea"
)

// pickedMsg carries the result of the native folder dialog back into Update.
// A zero value (no path, no err) means the user cancelled.
type pickedMsg struct {
	path string
	err  error
}

// folderScript pops Windows Explorer's "Select Folder" dialog and writes the
// chosen path to stdout. It runs on the Windows side via powershell.exe (WSL
// interop). A hidden TopMost owner form keeps the dialog in front of the
// terminal; AURA_START (a Windows path) seeds the initial location.
const folderScript = `
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$sig = @'
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
'@
$nm = Add-Type -MemberDefinition $sig -Name Fg -Namespace Aura -PassThru
# A tiny TopMost owner that we force to the foreground first; the dialog inherits
# it. Launched from a WSL/background process, Windows blocks focus-stealing, so we
# also send an ALT keystroke (the well-known workaround) to unlock the change.
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Width = 1; $owner.Height = 1
$owner.StartPosition = 'CenterScreen'
$owner.Show() | Out-Null
$nm::ShowWindow($owner.Handle, 5) | Out-Null
try { (New-Object -ComObject WScript.Shell).SendKeys('%') } catch {}
$nm::SetForegroundWindow($owner.Handle) | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = 'Select your RAW photo folder  -  Aurachrome'
$dlg.ShowNewFolderButton = $false
if ($env:AURA_START) { $dlg.SelectedPath = $env:AURA_START }
$res = $dlg.ShowDialog($owner)
$owner.Close()
if ($res -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dlg.SelectedPath) }
`

// pickFolder runs the native dialog as a tea.Cmd. startWSL (if set) is the path
// currently in the field, used to open the dialog at that location.
func pickFolder(startWSL string) tea.Cmd {
	return func() tea.Msg {
		if _, err := exec.LookPath("powershell.exe"); err != nil {
			return pickedMsg{err: fmt.Errorf("native folder picker needs Windows (powershell.exe not found)")}
		}
		cmd := exec.Command("powershell.exe", "-NoProfile", "-STA", "-EncodedCommand", encodePS(folderScript))
		if startWSL != "" {
			if w, err := exec.Command("wslpath", "-w", startWSL).Output(); err == nil {
				cmd.Env = append(os.Environ(), "AURA_START="+strings.TrimSpace(string(w)))
			}
		}
		out, err := cmd.Output()
		if err != nil {
			return pickedMsg{err: fmt.Errorf("folder picker failed: %w", err)}
		}
		win := strings.TrimSpace(string(out))
		if win == "" {
			return pickedMsg{} // cancelled
		}
		return pickedMsg{path: toWSLPath(win)}
	}
}

// toWSLPath converts a Windows path (C:\a\b) to its WSL mount (/mnt/c/a/b),
// preferring wslpath and falling back to a manual rewrite.
func toWSLPath(win string) string {
	win = strings.TrimSpace(win)
	if win == "" {
		return ""
	}
	if out, err := exec.Command("wslpath", "-u", win).Output(); err == nil {
		if p := strings.TrimSpace(string(out)); p != "" {
			return p
		}
	}
	p := strings.ReplaceAll(win, `\`, "/")
	if len(p) >= 2 && p[1] == ':' {
		drive := strings.ToLower(p[:1])
		return "/mnt/" + drive + p[2:]
	}
	return p
}

// encodePS renders a script as the UTF-16LE base64 that powershell.exe expects
// for -EncodedCommand, sidestepping all shell quoting concerns.
func encodePS(script string) string {
	u := utf16.Encode([]rune(script))
	b := make([]byte, len(u)*2)
	for i, r := range u {
		b[i*2] = byte(r)
		b[i*2+1] = byte(r >> 8)
	}
	return base64.StdEncoding.EncodeToString(b)
}
