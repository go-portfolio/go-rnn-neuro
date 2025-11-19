package rnn

import (
	"math/rand"
	"time"

	"github.com/go-portfolio/go-rnn-neuro/internal/textutils" // Пакет с функциями для работы с текстом
)

// Структура RNN
type RNN struct {
	Wx, Wh, Wy                     []float64
	bh, by                         float64
	inputDim, hiddenDim, outputDim int
}

// Инициализация RNN
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

// Генерация случайного массива
func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.NormFloat64()
	}
	return arr
}

// Прямой проход RNN для одного временного шага
func (rnn *RNN) forward(x []float64, hPrev []float64) ([]float64, []float64) {
	var hNew []float64
	hNew = make([]float64, rnn.hiddenDim)

	// Скрытое состояние
	for i := 0; i < rnn.hiddenDim; i++ {
		hNew[i] = rnn.bh
		for j := 0; j < rnn.inputDim; j++ {
			hNew[i] += rnn.Wx[j+rnn.inputDim*i] * x[j]
		}
		for j := 0; j < rnn.hiddenDim; j++ {
			hNew[i] += rnn.Wh[j+rnn.hiddenDim*i] * hPrev[j]
		}
		hNew[i] = tanh(hNew[i])
	}

	// Выход
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

// Функция активации tanh
func tanh(x float64) float64 {
	return (1 - x*x) / (1 + x*x) // Гиперболический тангенс
}

// Функция обучения
func (rnn *RNN) Train(inputs [][]float64, outputs [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(inputs); i++ {
			input := inputs[i]
			output := outputs[i]
			// Здесь можно добавить логику обновления весов (обратное распространение ошибки)
			rnn.updateWeights(input, output, learningRate)
		}
	}
}

// Метод обновления весов (должен быть реализован)
func (rnn *RNN) updateWeights(input []float64, output []float64, learningRate float64) {
	// Логика обновления весов
	// Используйте градиентный спуск для обновления Wx, Wh и Wy
}

// Прогнозирование следующего слова
func (rnn *RNN) PredictNextWord(inputText string, vocab map[string]int, seqLength int) string {
	indices := textutils.TextToIndices(inputText, vocab)
	hPrev := make([]float64, rnn.hiddenDim)

	// Прогнозируем для входной последовательности
	for _, index := range indices {
		x := make([]float64, rnn.inputDim)
		if index >= 0 {
			x[index] = 1.0 // One-hot вектор
		}

		output, _ := rnn.forward(x, hPrev)
		// Получаем индекс с наибольшим выходом
		predictedIdx := maxIndex(output)
		for word, idx := range vocab {
			if idx == predictedIdx {
				return word
			}
		}
	}
	return ""
}

// Функция для поиска индекса с максимальным значением
func maxIndex(output []float64) int {
	maxIdx := 0
	maxVal := output[0]
	for i, value := range output {
		if value > maxVal {
			maxIdx = i
			maxVal = value
		}
	}
	return maxIdx
}
