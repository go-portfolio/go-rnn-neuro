package dataset

import (
	"strings"

	"github.com/go-portfolio/go-rnn-neuro/internal/textutils"
)

// Функция для построения словаря из текста
func BuildVocab(texts []string) map[string]int {
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

// Подготовка данных для обучения (преобразование индексов в one-hot векторы)
func PrepareData(texts []string, seqLength int, vocab map[string]int) ([][]float64, [][]float64) {
	var inputs [][]float64
	var outputs [][]float64

	for _, text := range texts {
		indices := textutils.TextToIndices(text, vocab)

		// Формируем последовательности входов и выходов
		for i := 0; i < len(indices)-seqLength; i++ {
			seqIn := make([]float64, len(vocab))
			seqOut := make([]float64, len(vocab))

			// Входные данные: one-hot представление для последовательности слов
			for j := 0; j < seqLength; j++ {
				index := indices[i+j]
				if index >= 0 {
					seqIn[index] = 1.0
				}
			}

			// Выходные данные: one-hot представление для следующего слова
			nextIndex := indices[i+seqLength]
			if nextIndex >= 0 {
				seqOut[nextIndex] = 1.0
			}

			// Добавляем в список
			inputs = append(inputs, seqIn)
			outputs = append(outputs, seqOut)
		}
	}

	return inputs, outputs
}
