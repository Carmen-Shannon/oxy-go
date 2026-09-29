//go:build !js

package common

import "os"

// ReadFile reads the named file and returns its contents using the native
// filesystem implementation.
//
// Parameters:
//   - path: the file path to read
//
// Returns:
//   - []byte: the file contents
//   - error: error if the read fails
func ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}
