package common

// ToPtr returns a pointer to the given value.
//
// Parameters:
//   - v: the value to take the address of
//
// Returns:
//   - *T: a pointer to v
func ToPtr[T any](v T) *T {
	return &v
}

// Coalesce returns the first non-zero value from the provided values, or the zero value if all are zero.
//
// Parameters:
//   - values: a variadic list of values to check for non-zero status
//
// Returns:
//   - T: the first non-zero value from the input, or the zero value if all are zero
func Coalesce[T comparable](values ...T) T {
	var zero T
	for _, v := range values {
		if v != zero {
			return v
		}
	}
	return zero
}
