package main

import (
	"os"
	"path/filepath"
	"time"
)

type Config struct {
	PythonExe     string
	ScriptPath    string
	MinConf       float64
	LocalCooldown time.Duration
}

func defaultConfig() Config {
	wd, _ := os.Getwd()
	repo := wd
	if filepath.Base(wd) == "go" {
		repo = filepath.Dir(wd)
	}

	venvPython := filepath.Join(repo, "venv", "Scripts", "python.exe")
	pythonExe := venvPython
	if _, err := os.Stat(venvPython); err != nil {
		pythonExe = "py"
	}
	pyScript := filepath.Join(repo, "py", "recognizer.py")

	return Config{
		PythonExe:     pythonExe,
		ScriptPath:    pyScript,
		MinConf:       0.60,
		LocalCooldown: 250 * time.Millisecond,
	}
}
