package rnn

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/go-portfolio/go-rnn-neuro/internal/textutils"
)

// ------------------------- RNN STRUCT -------------------------
type RNN struct {
	Wx, Wh, Wy                     []float64
	bh, by                         []float64
	inputDim, hiddenDim, outputDim int
}

// ------------------------- INITIALIZATION -------------------------
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
func (rnn *RNN) Train(inputs [][]float64, outputs [][]float64, epochs int, lr float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("========== Epoch %d ==========\n", epoch+1)
		for i := 0; i < len(inputs); i++ {
			input := inputs[i]
			target := outputs[i]

			if len(input) == 0 || len(target) != rnn.outputDim || rnn.inputDim == 0 || len(input)%rnn.inputDim != 0 {
				continue
			}

			// Forward pass
			ys, hs := rnn.forwardSequence(input)
			lastY := ys[len(ys)-1]

			// Градиенты выходного слоя
			dWy, dby := rnn.computeOutputGradients(lastY, target, hs[len(hs)-1])

			// Backpropagation Through Time
			dWx, dWh, dbh, _ := rnn.backpropThroughTime(hs, input, dWy)

			// Clip gradients
			clipGradients(dWx, 5.0)
			clipGradients(dWh, 5.0)
			clipGradients(dWy, 5.0)
			clipGradients(dbh, 5.0)
			clipGradients(dby, 5.0)

			// Обновление весов
			rnn.updateWeights(dWx, dWh, dWy, dbh, dby, lr)
		}
	}
}

// ------------------------- FORWARD -------------------------
func (rnn *RNN) forwardSequence(input []float64) ([][]float64, [][]float64) {
	seqLen := len(input) / rnn.inputDim
	hs := make([][]float64, seqLen+1)
	for t := 0; t <= seqLen; t++ {
		hs[t] = make([]float64, rnn.hiddenDim)
	}
	ys := make([][]float64, seqLen)

	for t := 0; t < seqLen; t++ {
		x := input[t*rnn.inputDim : (t+1)*rnn.inputDim]
		y, hNew := rnn.forwardParallel(x, hs[t])
		ys[t] = y
		hs[t+1] = hNew

		// Логирование первых и последних шагов
		if seqLen <= 4 || t < 2 || t >= seqLen-2 {
			fmt.Printf("[Forward] Step %d\n", t)
			fmt.Printf("  x: %v\n", x)
			fmt.Printf("  hNew: %v\n", hNew)
			fmt.Printf("  y: %v\n", y)
		}
	}
	return ys, hs
}

func (rnn *RNN) forwardParallel(x []float64, hPrev []float64) ([]float64, []float64) {
	hNew := make([]float64, rnn.hiddenDim)
	y := make([]float64, rnn.outputDim)
	var wg sync.WaitGroup

	// Вычисление скрытого слоя параллельно
	for i := 0; i < rnn.hiddenDim; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			z := rnn.bh[i]
			for j := 0; j < rnn.inputDim; j++ {
				z += rnn.Wx[j+rnn.inputDim*i] * x[j]
			}
			for j := 0; j < rnn.hiddenDim; j++ {
				z += rnn.Wh[j+rnn.hiddenDim*i] * hPrev[j]
			}
			hNew[i] = activateHidden(z)
		}()
	}
	wg.Wait()

	// Вычисление выходного слоя параллельно
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

	y = activateOutput(y)
	return y, hNew
}

// ------------------------- ACTIVATION -------------------------
func activateHidden(z float64) float64 {
	return math.Tanh(z)
}

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

// ------------------------- BACKPROP -------------------------
func (rnn *RNN) computeOutputGradients(lastY, target []float64, hLast []float64) ([]float64, []float64) {
	dWy := make([]float64, len(rnn.Wy))
	dby := make([]float64, len(rnn.by))
	dy := make([]float64, rnn.outputDim)

	for k := 0; k < rnn.outputDim; k++ {
		dy[k] = lastY[k] - target[k]
		dby[k] += dy[k]
		for j := 0; j < rnn.hiddenDim; j++ {
			dWy[j+rnn.hiddenDim*k] += dy[k] * hLast[j]
		}
	}

	return dWy, dby
}

func (rnn *RNN) backpropThroughTime(hs [][]float64, input []float64, dhNextInit []float64) ([]float64, []float64, []float64, []float64) {
	dWx := make([]float64, len(rnn.Wx))
	dWh := make([]float64, len(rnn.Wh))
	dbh := make([]float64, len(rnn.bh))
	dhNext := dhNextInit

	seqLen := len(hs) - 1
	for t := seqLen - 1; t >= 0; t-- {
		h := hs[t+1]
		hPrev := hs[t]
		x := input[t*rnn.inputDim : (t+1)*rnn.inputDim]

		dh := make([]float64, rnn.hiddenDim)
		copy(dh, dhNext)

		dhraw := make([]float64, rnn.hiddenDim)
		for k := 0; k < rnn.hiddenDim; k++ {
			dhraw[k] = (1 - h[k]*h[k]) * dh[k]
			dbh[k] += dhraw[k]
		}

		for iNew := 0; iNew < rnn.hiddenDim; iNew++ {
			for jPrev := 0; jPrev < rnn.hiddenDim; jPrev++ {
				dWh[jPrev+rnn.hiddenDim*iNew] += dhraw[iNew] * hPrev[jPrev]
			}
			for jIn := 0; jIn < rnn.inputDim; jIn++ {
				dWx[jIn+rnn.inputDim*iNew] += dhraw[iNew] * x[jIn]
			}
		}

		newDhNext := make([]float64, rnn.hiddenDim)
		for jPrev := 0; jPrev < rnn.hiddenDim; jPrev++ {
			sum := 0.0
			for iNew := 0; iNew < rnn.hiddenDim; iNew++ {
				sum += rnn.Wh[jPrev+rnn.hiddenDim*iNew] * dhraw[iNew]
			}
			newDhNext[jPrev] = sum
		}
		dhNext = newDhNext
	}

	return dWx, dWh, dbh, dhNext
}

// ------------------------- HELPERS -------------------------
func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.NormFloat64() * 0.1
	}
	return arr
}

func clipGradients(arr []float64, limit float64) {
	for k := range arr {
		if arr[k] > limit {
			arr[k] = limit
		} else if arr[k] < -limit {
			arr[k] = -limit
		}
	}
}

func (rnn *RNN) updateWeights(dWx, dWh, dWy, dbh, dby []float64, lr float64) {
	for k := range rnn.Wx {
		rnn.Wx[k] -= lr * dWx[k]
	}
	for k := range rnn.Wh {
		rnn.Wh[k] -= lr * dWh[k]
	}
	for k := range rnn.Wy {
		rnn.Wy[k] -= lr * dWy[k]
	}
	for k := range rnn.bh {
		rnn.bh[k] -= lr * dbh[k]
	}
	for k := range rnn.by {
		rnn.by[k] -= lr * dby[k]
	}
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
func (rnn *RNN) PredictNextWord(inputText string, vocab map[string]int, seqLength int) string {
	indices := textutils.TextToIndices(inputText, vocab)
	hPrev := make([]float64, rnn.hiddenDim)
	var output []float64

	for _, index := range indices {
		x := make([]float64, rnn.inputDim)
		if index >= 0 && index < rnn.inputDim {
			x[index] = 1.0
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
