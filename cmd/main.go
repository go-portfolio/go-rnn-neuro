package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

// Функция активации tanh
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// Производная tanh
func tanhPrime(x float64) float64 {
	return 1 - x*x
}

// Структура для рекуррентной нейросети
type RNN struct {
	Wx, Wh, Wy  []float64
	bh, by       float64
	inputDim, hiddenDim, outputDim int
}

// Инициализация RNN
func NewRNN(inputDim, hiddenDim, outputDim int) *RNN {
	rand.Seed(time.Now().UnixNano())
	return &RNN{
		Wx:       randomArray(inputDim * hiddenDim),
		Wh:       randomArray(hiddenDim * hiddenDim),
		Wy:       randomArray(hiddenDim * outputDim),
		bh:       rand.Float64(),
		by:       rand.Float64(),
		inputDim: inputDim,
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

	// Выход из сети
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

// Функция для преобразования текста в индексы слов
func textToIndices(text string, vocab map[string]int) []int {
	words := strings.Fields(text)
	indices := make([]int, len(words))
	for i, word := range words {
		if idx, exists := vocab[word]; exists {
			indices[i] = idx
		} else {
			indices[i] = -1 // Если слово нет в словаре
		}
	}
	return indices
}

// Функция для построения словаря из текста
func buildVocab(texts []string) map[string]int {
	vocab := make(map[string]int)
	index := 0
	for _, text := range texts {
		words := strings.Fields(text)
		for _, word := range words {
			if _, exists := vocab[word]; !exists {
				vocab[word] = index
				index++
			}
		}
	}
	return vocab
}

// Основная функция
func main() {
	// Пример текста для обучения
	texts := []string{
		"i am learning go",
		"go is awesome",
		"i love programming in go",
	}

	// Строим словарь
	vocab := buildVocab(texts)
	fmt.Println("Vocabulary:", vocab)

	// Параметры RNN
	inputDim := len(vocab) // Размер словаря
	hiddenDim := 5
	outputDim := len(vocab) // Число уникальных слов

	// Инициализация RNN
	rnn := NewRNN(inputDim, hiddenDim, outputDim)

	// Тестовый вход: последовательность слов
	inputText := "i am"
	indices := textToIndices(inputText, vocab)
	hPrev := make([]float64, hiddenDim)

	// Процесс анализа: проходим через текст
	for _, index := range indices {
		x := make([]float64, inputDim)
		if index >= 0 {
			x[index] = 1.0 // Преобразуем индекс в one-hot вектор
		}

		// Прямой проход RNN
		output, hNew := rnn.forward(x, hPrev)
		fmt.Printf("Output: %v\n", output)

		// Обновляем скрытое состояние
		hPrev = hNew
	}
}
