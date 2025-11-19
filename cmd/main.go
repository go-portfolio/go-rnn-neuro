package main

import (
	"fmt"

	"github.com/go-portfolio/go-rnn-neuro/internal/dataset"
	"github.com/go-portfolio/go-rnn-neuro/internal/rnn"
)

func main() {
	// Пример данных для обучения
	texts := []string{
		"i am learning go",
		"go is awesome",
		"i love programming in go",
	}

	// Строим словарь
	vocab := dataset.BuildVocab(texts)
	fmt.Println("Vocabulary:", vocab)

	// Размер входа (размер словаря), скрытого слоя и выхода
	inputDim := len(vocab)
	hiddenDim := 5
	outputDim := len(vocab)

	// Инициализация модели RNN
	model := rnn.NewRNN(inputDim, hiddenDim, outputDim)

	// Подготовка данных для обучения
	seqLength := 3 // Длина последовательности входных данных
	inputs, outputs := dataset.PrepareData(texts, seqLength, vocab)

	// Обучение модели
	model.Train(inputs, outputs, 100, 0.01)

	// Прогнозирование следующего слова
	inputText := "i am"
	nextWord := model.PredictNextWord(inputText, vocab, seqLength)
	fmt.Printf("Next word prediction: %s\n", nextWord)
}
