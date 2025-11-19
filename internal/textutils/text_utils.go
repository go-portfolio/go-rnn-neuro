package textutils

import "strings"

// Функция для преобразования текста в индексы слов
func TextToIndices(text string, vocab map[string]int) []int {
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
