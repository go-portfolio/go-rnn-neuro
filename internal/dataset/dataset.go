package dataset

import (
	"strings"
	"sync"

	"github.com/go-portfolio/go-rnn-neuro/internal/textutils"
)

// BuildVocab строит словарь (word -> index) из массива текстов
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

// PrepareData преобразует тексты в обучающие пары (inputs, outputs) в формате one-hot
// seqLength — длина входной последовательности слов
func PrepareData(texts []string, seqLength int, vocab map[string]int) ([][]float64, [][]float64) {
	var inputs [][]float64
	var outputs [][]float64

	var wg sync.WaitGroup
	var mu sync.Mutex // mutex для безопасного добавления в slices

	for _, text := range texts {
		text := text // захватываем переменную
		wg.Add(1)
		go func() {
			defer wg.Done()
			indices := textutils.TextToIndices(text, vocab)

			localInputs := [][]float64{}
			localOutputs := [][]float64{}

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

				localInputs = append(localInputs, seqIn)
				localOutputs = append(localOutputs, seqOut)
			}

			// Добавляем локальные результаты в общий slice
			mu.Lock()
			inputs = append(inputs, localInputs...)
			outputs = append(outputs, localOutputs...)
			mu.Unlock()
		}()
	}

	wg.Wait()
	return inputs, outputs
}
