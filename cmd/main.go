package main

import (
	"fmt"

	"github.com/go-portfolio/go-rnn-neuro/internal/rnn"
	"github.com/go-portfolio/go-rnn-neuro/internal/textutils"
)

func main() {
	// Пример текста для обучения
	texts := []string{
		"i am learning go",
		"go is awesome",
		"i love programming in go",
	}

	// Строим словарь из текста
	vocab := textutils.BuildVocab(texts)
	fmt.Println("Vocabulary:", vocab)

	// Параметры RNN
	inputDim := len(vocab)  // Размерность входа = размер словаря
	hiddenDim := 5          // Размерность скрытого слоя
	outputDim := len(vocab) // Размерность выхода = размер словаря (количество уникальных слов)

	// Инициализация RNN
	rnnModel := rnn.NewRNN(inputDim, hiddenDim, outputDim)

	// Тестовый вход: последовательность слов
	inputText := "i am"                                  // Пример входного текста
	indices := textutils.TextToIndices(inputText, vocab) // Преобразуем текст в индексы
	hPrev := make([]float64, hiddenDim)                  // Начальное скрытое состояние

	// Процесс анализа: проходим через текст
	for _, index := range indices {
		x := make([]float64, inputDim) // Вектор входных данных
		if index >= 0 {
			x[index] = 1.0 // Преобразуем индекс в one-hot вектор
		}

		// Прямой проход через RNN для получения выхода и нового скрытого состояния
		output, hNew := rnnModel.Forward(x, hPrev)
		fmt.Printf("Output: %v\n", output) // Выводим выходное значение

		// Обновляем скрытое состояние для следующего шага
		hPrev = hNew
	}
}
