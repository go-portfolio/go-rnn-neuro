package activation

import "math"

// Функция активации tanh
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// Производная tanh
func TanhPrime(x float64) float64 {
	return 1 - x*x
}
