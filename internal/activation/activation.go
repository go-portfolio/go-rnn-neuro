package activation

import "math"

// Функция активации tanh
// tanh(x) возвращает гиперболический тангенс значения x.
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// Производная tanh
// Производная функции гиперболического тангенса равна (1 - x^2).
func TanhPrime(x float64) float64 {
	return 1 - x*x
}
