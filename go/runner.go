package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
)

type Event struct {
	Gesture    string  `json:"gesture"`
	Confidence float64 `json:"confidence"`
	Hand       string  `json:"hand"`
	TS         int64   `json:"ts"`
}

func RunRecognizer(ctx context.Context, cfg Config, onEvent func(Event), onStderr func(string)) error {
	log.Printf("Starting Python recognizer: %s %s", cfg.PythonExe, cfg.ScriptPath)

	cmd := exec.CommandContext(ctx, cfg.PythonExe, cfg.ScriptPath)
	cmd.Dir = filepath.Dir(cfg.ScriptPath)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	defer stdout.Close()

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}
	defer stderr.Close()

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start recognizer process: %w", err)
	}

	// Handle stderr in a separate goroutine
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Stderr handler panic: %v", r)
			}
		}()

		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			if onStderr != nil {
				onStderr(sc.Text())
			}
		}
		if err := sc.Err(); err != nil && onStderr != nil {
			onStderr(fmt.Sprintf("stderr scanner error: %v", err))
		}
	}()

	// Handle stdout events
	sc := bufio.NewScanner(stdout)
	for sc.Scan() {
		select {
		case <-ctx.Done():
			log.Println("Context cancelled, stopping stdout reader")
			return ctx.Err()
		default:
		}

		line := sc.Bytes()
		var ev Event
		if err := json.Unmarshal(line, &ev); err != nil {
			if onStderr != nil {
				onStderr(fmt.Sprintf("JSON parse error: %s (%v)", string(line), err))
			}
			continue
		}
		if onEvent != nil {
			onEvent(ev)
		}
	}

	if err := sc.Err(); err != nil && onStderr != nil {
		onStderr(fmt.Sprintf("stdout scanner error: %v", err))
	}

	// Wait for the process to finish
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("recognizer process failed: %w", err)
	}

	log.Println("Recognizer process completed successfully")
	return nil
}
