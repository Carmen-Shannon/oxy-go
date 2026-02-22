package common

// Delegate defines the contract for embedding delegation support into any interface.
// The type parameter T should be the interface type that the delegate will back,
// enabling method calls to be routed through a delegate instance.
//
// In production code, the delegate is typically set to the instance itself during construction.
// In test code, the delegate can be replaced with a mock to intercept and control specific method calls.
type Delegate[T any] interface {
	// SetDelegate sets the delegate instance that method calls will be routed through.
	// In production code this is typically called during construction to set the delegate to the instance itself.
	// In test code this is used to replace the delegate with a mock instance.
	//
	// Parameters:
	//   - delegate: the instance to set as the delegate
	SetDelegate(delegate T)
}

// DelegateImpl is the implementation of the Delegate interface.
// It should be embedded in any struct that needs delegation support.
// The type parameter T should be the interface type that the struct implements,
// so that the Delegate field can hold any implementation of that interface including mocks.
//
// Example usage:
//
//	type fooImpl struct {
//	    common.DelegateImpl[Foo]
//	}
//
//	func NewFoo() Foo {
//	    f := &fooImpl{}
//	    f.SetDelegate(f)
//	    return f
//	}
//
//	func (f *fooImpl) MethodA() error {
//	    return f.Delegate.MethodB()
//	}
type DelegateImpl[T any] struct {
	Delegate T
}

func (d *DelegateImpl[T]) SetDelegate(delegate T) {
	d.Delegate = delegate
}
