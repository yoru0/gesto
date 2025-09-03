package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	if err := run(); err != nil {
		log.Fatalf("Application failed: %v", err)
	}
}

func run() error {
	cfg := defaultConfig()

	if _, err := os.Stat(cfg.ScriptPath); err != nil {
		return fmt.Errorf("recognizer.py not found at %s (adjust ScriptPath in config.go)", cfg.ScriptPath)
	}

	log.Printf("Python: %s", cfg.PythonExe)
	log.Printf("Script: %s", cfg.ScriptPath)
	log.Printf("Workdir: %s", filepath.Dir(cfg.ScriptPath))

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	inputSender := NewWindowsInputSender()
	gestureHandler := NewGestureHandler(inputSender, cfg)

	log.Println("Starting gesture recognition...")
	return RunRecognizer(ctx, cfg, gestureHandler.HandleEvent, gestureHandler.HandleError)
}

// GestureHandler handles gesture events with rate limiting and input sending
type GestureHandler struct {
	inputSender InputSender
	config      Config
	lastSent    time.Time
}

// NewGestureHandler creates a new gesture handler
func NewGestureHandler(inputSender InputSender, config Config) *GestureHandler {
	return &GestureHandler{
		inputSender: inputSender,
		config:      config,
	}
}

// HandleEvent processes gesture events
func (h *GestureHandler) HandleEvent(ev Event) {
	if err := ev.Validate(); err != nil {
		log.Printf("Invalid event: %v", err)
		return
	}

	if ev.Confidence < h.config.MinConf {
		log.Printf("Low confidence gesture ignored: %s (%.2f)", ev.Gesture, ev.Confidence)
		return
	}

	if time.Since(h.lastSent) < h.config.LocalCooldown {
		log.Printf("Gesture rate limited: %s", ev.Gesture)
		return
	}

	var mediaKey MediaKey
	var description string

	switch ev.Gesture {
	case "play_pause":
		mediaKey = MediaKeyPlayPause
		description = "▶⏸  play/pause"
	case "next":
		mediaKey = MediaKeyNext
		description = "⏭  next"
	case "prev":
		mediaKey = MediaKeyPrev
		description = "⏮  previous"
	default:
		log.Printf("Unknown gesture: %s", ev.Gesture)
		return
	}

	if err := h.inputSender.SendMediaKey(mediaKey); err != nil {
		log.Printf("Failed to send media key %s: %v", description, err)
		return
	}

	h.lastSent = time.Now()
	log.Printf("%s (conf=%.2f, hand=%s)", description, ev.Confidence, ev.Hand)
}

// HandleError processes error messages from the Python script
func (h *GestureHandler) HandleError(line string) {
	log.Printf("[python] %s", line)
}
