package lifecycle

// Hook defines a generic function type that constrains the allowed function signatures for Lifecycle hooks
//
// Returns:
//   - error: an error if the hook execution fails, otherwise nil
type Hook func() error
