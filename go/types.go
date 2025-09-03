package main

import "fmt"

// MediaKey represents a media control key
type MediaKey uint16

const (
	MediaKeyPlayPause MediaKey = 0xB3
	MediaKeyNext      MediaKey = 0xB0
	MediaKeyPrev      MediaKey = 0xB1
)

// InputSender interface for sending input events
type InputSender interface {
	SendMediaKey(key MediaKey) error
}

// Validate checks if the event is valid
func (e *Event) Validate() error {
	if e.Gesture == "" {
		return fmt.Errorf("gesture cannot be empty")
	}
	if e.Confidence < 0 || e.Confidence > 1 {
		return fmt.Errorf("confidence must be between 0 and 1, got %f", e.Confidence)
	}
	if e.Hand != "Left" && e.Hand != "Right" {
		return fmt.Errorf("hand must be 'Left' or 'Right', got %s", e.Hand)
	}
	return nil
}
