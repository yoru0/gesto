package main

import (
	"fmt"
	"log"
	"syscall"
	"unsafe"
)

var (
	user32        = syscall.NewLazyDLL("user32.dll")
	procSendInput = user32.NewProc("SendInput")
)

const (
	InputKeyboard  = 1
	KeyEventFKeyUp = 0x0002
)

// WindowsInputSender implements the InputSender interface for Windows
type WindowsInputSender struct{}

// NewWindowsInputSender creates a new Windows input sender
func NewWindowsInputSender() *WindowsInputSender {
	return &WindowsInputSender{}
}

type keyboardInput struct {
	WVk         uint16
	WScan       uint16
	DwFlags     uint32
	Time        uint32
	DwExtraInfo uintptr
}

type input struct {
	Type uint32
	_    uint32 // padding for 64-bit alignment
	Ki   keyboardInput
	_    [8]byte // padding to match the size of the largest union member
}

// SendMediaKey sends a media key press event
func (w *WindowsInputSender) SendMediaKey(key MediaKey) error {
	return sendKey(uint16(key))
}

func sendKey(vk uint16) error {
	down := input{
		Type: InputKeyboard,
		Ki: keyboardInput{
			WVk: vk,
		},
	}

	up := input{
		Type: InputKeyboard,
		Ki: keyboardInput{
			WVk:     vk,
			DwFlags: KeyEventFKeyUp,
		},
	}

	inputs := []input{down, up}

	n, _, err := procSendInput.Call(
		uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&inputs[0])),
		unsafe.Sizeof(inputs[0]),
	)
	if n == 0 {
		log.Printf("SendInput failed: %v", err)
		return fmt.Errorf("SendInput failed: %v", err)
	}
	return nil
}
