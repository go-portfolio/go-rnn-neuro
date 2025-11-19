package rnn

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/go-portfolio/go-rnn-neuro/internal/textutils"
)

// ------------------------- RNN STRUCT -------------------------
// RNN — структура рекуррентной нейронной сети
type RNN struct {
	Wx, Wh, Wy                     []float64 // весовые матрицы для входа, скрытого состояния и выхода
	bh, by                         []float64 // смещения скрытого и выходного слоя
	inputDim, hiddenDim, outputDim int       // размеры входного, скрытого и выходного слоёв
}

// ------------------------- INITIALIZATION -------------------------
// NewRNN создаёт новую RNN с рандомными весами и смещениями
func NewRNN(inputDim, hiddenDim, outputDim int) *RNN {
	rand.Seed(time.Now().UnixNano())
	return &RNN{
		Wx:        randomArray(inputDim * hiddenDim),
		Wh:        randomArray(hiddenDim * hiddenDim),
		Wy:        randomArray(hiddenDim * outputDim),
		bh:        randomArray(hiddenDim),
		by:        randomArray(outputDim),
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
	}
}

// ------------------------- TRAIN -------------------------
// Train обучает RNN на данных с помощью BPTT и SGD
func (rnn *RNN) Train(inputs [][]float64, outputs [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		// Проходим по всем примерам
		for i := 0; i < len(inputs); i++ {
			input := inputs[i]
			target := outputs[i]

			// Проверка корректности размеров входа и выхода
			if len(input) == 0 || len(target) != rnn.outputDim || rnn.inputDim == 0 || len(input)%rnn.inputDim != 0 {
				continue
			}

			seqLen := len(input) / rnn.inputDim
			if seqLen <= 0 {
				continue
			}

			// ------------------------- FORWARD -------------------------
			// hs — скрытые состояния для всех шагов (инициализированы нулями)
			hs := make([][]float64, seqLen+1)
			for t := 0; t <= seqLen; t++ {
				hs[t] = make([]float64, rnn.hiddenDim)
			}

			// ys — выходные векторы для каждого шага
			ys := make([][]float64, seqLen)

			// Прямой проход по времени
			for t := 0; t < seqLen; t++ {
				x := input[t*rnn.inputDim : (t+1)*rnn.inputDim]
				y, hNew := rnn.forwardParallel(x, hs[t]) // параллельный forward
				ys[t] = y
				hs[t+1] = hNew
			}

			// ------------------------- BPTT -------------------------
			// Инициализация градиентов
			dWx := make([]float64, len(rnn.Wx))
			dWh := make([]float64, len(rnn.Wh))
			dWy := make([]float64, len(rnn.Wy))
			dbh := make([]float64, len(rnn.bh))
			dby := make([]float64, len(rnn.by))
			dhNext := make([]float64, rnn.hiddenDim) // dh от следующего шага

			// Градиент ошибки на последнем шаге (softmax + CE)
			lastY := ys[seqLen-1]
			dy := make([]float64, rnn.outputDim)
			for k := 0; k < rnn.outputDim; k++ {
				dy[k] = lastY[k] - target[k]
			}

			// Градиенты по выходным весам Wy и смещениям by
			for iOut := 0; iOut < rnn.outputDim; iOut++ {
				if iOut >= len(dby) {
					continue
				}
				dby[iOut] += dy[iOut]
				for jHidden := 0; jHidden < rnn.hiddenDim; jHidden++ {
					if jHidden+rnn.hiddenDim*iOut < len(dWy) && jHidden < len(hs[seqLen]) {
						dWy[jHidden+rnn.hiddenDim*iOut] += dy[iOut] * hs[seqLen][jHidden]
					}
				}
			}

			// Вычисление dhNext = Wy^T * dy
			for jHidden := 0; jHidden < rnn.hiddenDim; jHidden++ {
				sum := 0.0
				for iOut := 0; iOut < rnn.outputDim; iOut++ {
					if jHidden+rnn.hiddenDim*iOut < len(rnn.Wy) {
						sum += rnn.Wy[jHidden+rnn.hiddenDim*iOut] * dy[iOut]
					}
				}
				dhNext[jHidden] += sum
			}

			// Обратный проход по времени
			for t := seqLen - 1; t >= 0; t-- {
				h := hs[t+1]
				hPrev := hs[t]
				x := input[t*rnn.inputDim : (t+1)*rnn.inputDim]

				dh := make([]float64, rnn.hiddenDim)
				copy(dh, dhNext)

				// Производная tanh
				dhraw := make([]float64, rnn.hiddenDim)
				for k := 0; k < rnn.hiddenDim; k++ {
					dhraw[k] = (1 - h[k]*h[k]) * dh[k]
					dbh[k] += dhraw[k]
				}

				// Градиенты по Wx и Wh
				for iNew := 0; iNew < rnn.hiddenDim; iNew++ {
					for jPrev := 0; jPrev < rnn.hiddenDim; jPrev++ {
						if jPrev+rnn.hiddenDim*iNew < len(dWh) {
							dWh[jPrev+rnn.hiddenDim*iNew] += dhraw[iNew] * hPrev[jPrev]
						}
					}
					for jIn := 0; jIn < rnn.inputDim; jIn++ {
						if jIn+rnn.inputDim*iNew < len(dWx) {
							dWx[jIn+rnn.inputDim*iNew] += dhraw[iNew] * x[jIn]
						}
					}
				}

				// dhNext для предыдущего шага
				newDhNext := make([]float64, rnn.hiddenDim)
				for jPrev := 0; jPrev < rnn.hiddenDim; jPrev++ {
					sum := 0.0
					for iNew := 0; iNew < rnn.hiddenDim; iNew++ {
						if jPrev+rnn.hiddenDim*iNew < len(rnn.Wh) {
							sum += rnn.Wh[jPrev+rnn.hiddenDim*iNew] * dhraw[iNew]
						}
					}
					newDhNext[jPrev] = sum
				}
				dhNext = newDhNext
			}

			// ------------------------- GRADIENT CLIPPING -------------------------
			clip := func(arr []float64, limit float64) {
				for k := range arr {
					if arr[k] > limit {
						arr[k] = limit
					} else if arr[k] < -limit {
						arr[k] = -limit
					}
				}
			}
			clip(dWx, 5.0)
			clip(dWh, 5.0)
			clip(dWy, 5.0)
			clip(dbh, 5.0)
			clip(dby, 5.0)

			// ------------------------- UPDATE WEIGHTS -------------------------
			for k := range rnn.Wx {
				rnn.Wx[k] -= learningRate * dWx[k]
			}
			for k := range rnn.Wh {
				rnn.Wh[k] -= learningRate * dWh[k]
			}
			for k := range rnn.Wy {
				rnn.Wy[k] -= learningRate * dWy[k]
			}
			for k := range rnn.bh {
				rnn.bh[k] -= learningRate * dbh[k]
			}
			for k := range rnn.by {
				rnn.by[k] -= learningRate * dby[k]
			}
		}
	}
}

// ------------------------- FORWARD (PARALLEL) -------------------------
// ------------------------- ACTIVATIONS -------------------------

// Активация скрытого слоя (tanh)
func activateHidden(z float64) float64 {
	return math.Tanh(z)
}

// Активация выходного слоя (softmax)
func activateOutput(logits []float64) []float64 {
	max := logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}

	expSum := 0.0
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		expSum += out[i]
	}

	for i := range out {
		out[i] /= expSum
	}
	return out
}

// ------------------------- FORWARD -------------------------

// forwardParallel выполняет прямой проход с явными активациями
func (rnn *RNN) forwardParallel(x []float64, hPrev []float64) ([]float64, []float64) {
	hNew := make([]float64, rnn.hiddenDim)
	y := make([]float64, rnn.outputDim)
	var wg sync.WaitGroup

	// ------------------------- HIDDEN LAYER -------------------------
	for i := 0; i < rnn.hiddenDim; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Суммируем вход и рекуррентное состояние + смещение
			z := rnn.bh[i]
			for j := 0; j < rnn.inputDim; j++ {
				z += rnn.Wx[j+rnn.inputDim*i] * x[j]
			}
			for j := 0; j < rnn.hiddenDim; j++ {
				z += rnn.Wh[j+rnn.hiddenDim*i] * hPrev[j]
			}
			// Применяем явную функцию активации скрытого слоя
			hNew[i] = activateHidden(z)
		}()
	}
	wg.Wait()

	// ------------------------- OUTPUT LAYER -------------------------
	for i := 0; i < rnn.outputDim; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			z := rnn.by[i]
			for j := 0; j < rnn.hiddenDim; j++ {
				z += rnn.Wy[j+rnn.hiddenDim*i] * hNew[j]
			}
			y[i] = z
		}()
	}
	wg.Wait()

	// Применяем явную функцию активации выхода (softmax)
	y = activateOutput(y)
	return y, hNew
}



// ------------------------- SOFTMAX (PARALLEL) -------------------------
func softmaxParallel(logits []float64) []float64 {
	max := logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}

	expSum := 0.0
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		expSum += out[i]
	}

	var wg sync.WaitGroup
	for i := range out {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			out[i] /= expSum
		}()
	}
	wg.Wait()
	return out
}

// ------------------------- HELPERS -------------------------
func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.NormFloat64() * 0.1 // случайная инициализация
	}
	return arr
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func maxIndex(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, v := range arr {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// ------------------------- PREDICTION -------------------------
// PredictNextWord предсказывает следующее слово по входной последовательности
func (rnn *RNN) PredictNextWord(inputText string, vocab map[string]int, seqLength int) string {
	indices := textutils.TextToIndices(inputText, vocab)
	hPrev := make([]float64, rnn.hiddenDim)
	var output []float64

	for _, index := range indices {
		x := make([]float64, rnn.inputDim)
		if index >= 0 && index < rnn.inputDim {
			x[index] = 1.0 // one-hot
		}
		output, hPrev = rnn.forwardParallel(x, hPrev)
	}

	if len(output) == 0 {
		return ""
	}

	predIdx := maxIndex(output)
	for word, idx := range vocab {
		if idx == predIdx {
			return word
		}
	}
	return ""
}
