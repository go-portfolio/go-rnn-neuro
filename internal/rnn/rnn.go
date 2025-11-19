package rnn

import (
	"math/rand"
	"time"

	"github.com/go-portfolio/go-rnn-neuro/internal/activation"
)

// Структура для рекуррентной нейросети (RNN)
// Содержит веса и смещения для сети, а также размеры слоев.
type RNN struct {
	Wx, Wh, Wy                     []float64 // Веса для входа, скрытых состояний и выхода
	bh, by                         float64   // Смещения для скрытых состояний и выхода
	inputDim, hiddenDim, outputDim int       // Размерности входа, скрытого слоя и выхода
}

// Инициализация RNN
// Создается новая рекуррентная нейросеть с заданными размерами слоев.
func NewRNN(inputDim, hiddenDim, outputDim int) *RNN {
	rand.Seed(time.Now().UnixNano())
	return &RNN{
		Wx:        randomArray(inputDim * hiddenDim),
		Wh:        randomArray(hiddenDim * hiddenDim),
		Wy:        randomArray(hiddenDim * outputDim),
		bh:        rand.Float64(),
		by:        rand.Float64(),
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
	}
}

// Генерация случайного массива весов
func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.NormFloat64()
	}
	return arr
}

// Прямой проход RNN для одного временного шага
func (rnn *RNN) Forward(x []float64, hPrev []float64) ([]float64, []float64) {
	var hNew []float64
	hNew = make([]float64, rnn.hiddenDim) // Новый скрытый слой

	// Вычисление скрытого состояния для текущего шага
	for i := 0; i < rnn.hiddenDim; i++ {
		hNew[i] = rnn.bh
		for j := 0; j < rnn.inputDim; j++ {
			hNew[i] += rnn.Wx[j+rnn.inputDim*i] * x[j]
		}
		for j := 0; j < rnn.hiddenDim; j++ {
			hNew[i] += rnn.Wh[j+rnn.hiddenDim*i] * hPrev[j]
		}
		hNew[i] = activation.Tanh(hNew[i])
	}

	// Вычисление выходных значений
	var y []float64
	y = make([]float64, rnn.outputDim)
	for i := 0; i < rnn.outputDim; i++ {
		y[i] = rnn.by
		for j := 0; j < rnn.hiddenDim; j++ {
			y[i] += rnn.Wy[j+rnn.hiddenDim*i] * hNew[j]
		}
	}

	return y, hNew
}
