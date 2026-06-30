package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Live bridge test: spawns the real Python engine via `poetry run aurachrome`
// and drains its JSON progress stream. Skipped unless AURA_ENGINE_TEST=1 (it
// needs poetry + the installed engine). Run:
//
//	AURA_ENGINE_TEST=1 AURA_TEST_INPUT=/path/to/one.jpg go test -run Bridge -v
func TestEngineBridgeLive(t *testing.T) {
	if os.Getenv("AURA_ENGINE_TEST") == "" {
		t.Skip("set AURA_ENGINE_TEST=1 (needs poetry + engine)")
	}
	in := os.Getenv("AURA_TEST_INPUT")
	if in == "" {
		t.Skip("set AURA_TEST_INPUT to a RAW/image file or folder")
	}
	out := t.TempDir()
	cfg := runConfig{input: in, output: out, look: "classic", format: "tiff16",
		grain: "standard", ir: "auto", device: "cpu", jobs: 1}

	msg := startEngine(cfg)()
	rdy, ok := msg.(engReadyMsg)
	if !ok {
		t.Fatalf("expected engReadyMsg, got %T: %+v", msg, msg)
	}

	var started, done bool
	var images int
	for {
		next := waitForEngine(rdy.ch)()
		if next == nil {
			break
		}
		switch v := next.(type) {
		case engStartMsg:
			started = true
		case engImageMsg:
			images++
		case engDoneMsg:
			done = true
		case engErrMsg:
			t.Fatalf("engine error: %v", v.err)
		}
	}
	if !started || !done || images == 0 {
		t.Fatalf("incomplete stream: started=%v done=%v images=%d", started, done, images)
	}
	// a TIFF should exist
	matches, _ := filepath.Glob(filepath.Join(out, "*.tif"))
	if len(matches) == 0 {
		t.Fatal("no TIFF written")
	}
}
