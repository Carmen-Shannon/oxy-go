//go:build js

package common

import (
	"fmt"
	"syscall/js"

	"github.com/Carmen-Shannon/webgpu/jsx"
)

// ReadFile reads the named resource and returns its contents via the browser
// Fetch API under GOOS=js, where no local filesystem exists.
// See https://developer.mozilla.org/docs/Web/API/Fetch_API and
// https://pkg.go.dev/github.com/Carmen-Shannon/webgpu/jsx
//
// Parameters:
//   - path: the URL path to fetch
//
// Returns:
//   - []byte: the response body
//   - error: error if the fetch fails
func ReadFile(path string) ([]byte, error) {
	resp, ok := jsx.Await(js.Global().Call("fetch", path))
	if !ok || !resp.Truthy() {
		return nil, fmt.Errorf("failed to fetch %q", path)
	}
	if !resp.Get("ok").Bool() {
		return nil, fmt.Errorf("fetch %q failed with status %s", path, resp.Get("status").String())
	}

	buf, ok := jsx.Await(resp.Call("arrayBuffer"))
	if !ok || !buf.Truthy() {
		return nil, fmt.Errorf("failed to read body of %q", path)
	}

	size := buf.Get("byteLength").Int()
	data := make([]byte, size)
	js.CopyBytesToGo(data, js.Global().Get("Uint8Array").New(buf))
	return data, nil
}
