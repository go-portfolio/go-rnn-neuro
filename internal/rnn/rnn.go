package rnn

import (
	"fmt"       // Для вывода логов в консоль
	"math"      // Для математических функций: exp, tanh
	"math/rand" // Для генерации случайных чисел при инициализации весов
	"sync"      // Для параллельных вычислений с помощью горутин
	"time"      // Для получения текущего времени и инициализации seed

	"github.com/go-portfolio/go-rnn-neuro/internal/textutils" // Вспомогательные функции для текста
)

// ------------------------- RNN STRUCT -------------------------
type RNN struct {
	Wx, Wh, Wy                     []float64 // Веса: вход->скрытый, скрытый->скрытый, скрытый->выход
	bh, by                         []float64 // Смещения скрытого слоя и выходного слоя
	inputDim, hiddenDim, outputDim int       // Размерности входного, скрытого и выходного слоев
}

// ------------------------- INITIALIZATION -------------------------
func NewRNN(inputDim, hiddenDim, outputDim int) *RNN {
	// Инициализируем генератор случайных чисел текущим временем
	rand.Seed(time.Now().UnixNano())

	// Создаем объект RNN с инициализацией всех весов и смещений случайными маленькими числами
	return &RNN{
		Wx:        randomArray(inputDim * hiddenDim), // Веса между входным и скрытым слоями
		Wh:        randomArray(hiddenDim * hiddenDim), // Веса между скрытым и скрытым слоями
		Wy:        randomArray(hiddenDim * outputDim), // Веса между скрытым и выходным слоями
		bh:        randomArray(hiddenDim),            // Смещения скрытого слоя
		by:        randomArray(outputDim),           // Смещения выходного слоя
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
	}
}

// ------------------------- TRAIN -------------------------
func (rnn *RNN) Train(inputs [][]float64, outputs [][]float64, epochs int, lr float64) {
	// Цикл по эпохам обучения
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("========== Epoch %d ==========\n", epoch+1)

		// Цикл по каждому обучающему примеру
		for i := 0; i < len(inputs); i++ {
			input := inputs[i]   // Текущий входной вектор (одна последовательность слов)
			target := outputs[i] // Целевой вектор выхода (следующее слово в one-hot формате)

			// Проверка корректности входных данных
			if len(input) == 0 || len(target) != rnn.outputDim || rnn.inputDim == 0 || len(input)%rnn.inputDim != 0 {
				continue // Пропускаем некорректные данные
			}

			// ------------------------- FORWARD PASS -------------------------
			ys, hs := rnn.forwardSequence(input) // Вычисляем все скрытые состояния и выходы
			lastY := ys[len(ys)-1]              // Выход последнего шага последовательности

			// ------------------------- OUTPUT GRADIENTS -------------------------
			dWy, dby := rnn.computeOutputGradients(lastY, target, hs[len(hs)-1])
			// dWy — градиенты весов скрытый->выход
			// dby — градиенты смещений выходного слоя

			// ------------------------- BACKPROP THROUGH TIME -------------------------
			dWx, dWh, dbh, _ := rnn.backpropThroughTime(hs, input, dWy)
			// dWx — градиенты входных весов
			// dWh — градиенты скрытых весов
			// dbh — градиенты смещений скрытого слоя

			// Ограничение градиентов (clipping) чтобы избежать взрыва градиентов
			clipGradients(dWx, 5.0)
			clipGradients(dWh, 5.0)
			clipGradients(dWy, 5.0)
			clipGradients(dbh, 5.0)
			clipGradients(dby, 5.0)

			// Обновление всех весов с учетом learning rate
			rnn.updateWeights(dWx, dWh, dWy, dbh, dby, lr)
		}
	}
}

// ------------------------- FORWARD -------------------------
func (rnn *RNN) forwardSequence(input []float64) ([][]float64, [][]float64) {
	seqLen := len(input) / rnn.inputDim // Количество шагов (слов) в последовательности

	// Инициализация массива скрытых состояний: hs[t] — скрытое состояние на шаге t
	hs := make([][]float64, seqLen+1)
	for t := 0; t <= seqLen; t++ {
		hs[t] = make([]float64, rnn.hiddenDim) // Все нули в начале
	}

	ys := make([][]float64, seqLen) // Массив выходов на каждом шаге t

	// Цикл по шагам последовательности
	for t := 0; t < seqLen; t++ {
		x := input[t*rnn.inputDim : (t+1)*rnn.inputDim] // One-hot вектор входа на шаге t
		y, hNew := rnn.forwardParallel(x, hs[t])       // Forward pass для одного шага
		ys[t] = y
		hs[t+1] = hNew

		// Логирование первых и последних шагов для отладки
		if seqLen <= 4 || t < 2 || t >= seqLen-2 {
			fmt.Printf("[Forward] Step %d\n", t)
			fmt.Printf("  x: %v\n", x)       // Вектор входа
			fmt.Printf("  hNew: %v\n", hNew) // Скрытое состояние после шага
			fmt.Printf("  y: %v\n", y)       // Выход сети
		}
	}

	return ys, hs // Возвращаем массив выходов и скрытых состояний
}

// computeHiddenNeuron вычисляет активацию одного нейрона скрытого слоя.
// x — текущий входной вектор (one-hot последовательность),
// hPrev — скрытое состояние на предыдущем шаге,
// i — индекс нейрона скрытого слоя
func (rnn *RNN) computeHiddenNeuron(x []float64, hPrev []float64, i int) float64 {
    // Начальное значение z — это смещение для i-го нейрона
    z := rnn.bh[i]

    // Вклад входного слоя: сумма входов, умноженных на соответствующие веса Wx
    for j := 0; j < rnn.inputDim; j++ {
        z += rnn.Wx[j+rnn.inputDim*i] * x[j]
    }

    // Вклад скрытого слоя: сумма предыдущих скрытых состояний, умноженных на веса Wh
    for j := 0; j < rnn.hiddenDim; j++ {
        z += rnn.Wh[j+rnn.hiddenDim*i] * hPrev[j]
    }

    // Пропускаем через функцию активации tanh и возвращаем новое скрытое состояние нейрона
    return activateHidden(z)
}

// computeHiddenLayer вычисляет все скрытые состояния слоя на текущем шаге
// с использованием параллельных горутин для каждого нейрона
func (rnn *RNN) computeHiddenLayer(x []float64, hPrev []float64) []float64 {
    // Создаём слайс для новых скрытых состояний
    hNew := make([]float64, rnn.hiddenDim)
    var wg sync.WaitGroup // sync.WaitGroup для ожидания завершения всех горутин

    // Параллельный запуск вычисления каждого нейрона скрытого слоя
    for i := 0; i < rnn.hiddenDim; i++ {
        i := i // захватываем локальную копию переменной для горутины
        wg.Add(1)
        go func() {
            defer wg.Done() // уменьшаем счетчик wg по завершению горутины
            hNew[i] = rnn.computeHiddenNeuron(x, hPrev, i) // вычисляем активацию нейрона
        }()
    }

    // Ждём, пока все горутины закончат работу
    wg.Wait()

    return hNew // возвращаем весь вектор скрытых состояний
}

// forwardParallel вычисляет один шаг прямого прохода RNN
// x — текущий входной вектор
// hPrev — скрытое состояние предыдущего шага
// возвращает: y — выходной вектор (после softmax), hNew — новое скрытое состояние
func (rnn *RNN) forwardParallel(x []float64, hPrev []float64) ([]float64, []float64) {
    // Вычисляем скрытый слой на текущем шаге
    hNew := rnn.computeHiddenLayer(x, hPrev)

    // Создаём слайс для выходного слоя
    y := make([]float64, rnn.outputDim)
    var wg sync.WaitGroup // для параллельного вычисления выходных нейронов

    // Параллельное вычисление выходного слоя
    for i := 0; i < rnn.outputDim; i++ {
        i := i // захватываем локальную копию переменной для горутины
        wg.Add(1)
        go func() {
            defer wg.Done()
            z := rnn.by[i] // начинаем с bias для i-го выходного нейрона

            // Вклад скрытого слоя: сумма произведений скрытых состояний на веса Wy
            for j := 0; j < rnn.hiddenDim; j++ {
                z += rnn.Wy[j+rnn.hiddenDim*i] * hNew[j]
            }
            y[i] = z // сохраняем линейную комбинацию перед активацией
        }()
    }

    // Ждём, пока все горутины закончат вычисление выхода
    wg.Wait()

    // Применяем softmax для получения вероятностей на выходе
    y = activateOutput(y)

    return y, hNew // возвращаем выход и новое скрытое состояние
}


// ------------------------- ACTIVATION -------------------------
func activateHidden(z float64) float64 {
	return math.Tanh(z) // Tanh для скрытого слоя
}

func activateOutput(logits []float64) []float64 {
	max := logits[0] // Максимум для численной стабильности softmax
	for _, v := range logits {
		if v > max {
			max = v
		}
	}

	expSum := 0.0
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v - max) // exp(z-max)
		expSum += out[i]            // Сумма экспонент
	}

	for i := range out {
		out[i] /= expSum // Делим на сумму, чтобы получить вероятности
	}

	return out
}

// ------------------------- BACKPROP -------------------------
func (rnn *RNN) computeOutputGradients(lastY, target []float64, hLast []float64) ([]float64, []float64) {
	dWy := make([]float64, len(rnn.Wy)) // Градиенты Wy
	dby := make([]float64, len(rnn.by)) // Градиенты смещений
	dy := make([]float64, rnn.outputDim) // Ошибка на выходе

	for k := 0; k < rnn.outputDim; k++ {
		dy[k] = lastY[k] - target[k]   // Разность предсказания и целевого выхода
		dby[k] += dy[k]                // Градиент смещения
		for j := 0; j < rnn.hiddenDim; j++ {
			dWy[j+rnn.hiddenDim*k] += dy[k] * hLast[j] // Градиент весов скрытый->выход
		}
	}

	return dWy, dby
}

// ------------------------- BACKPROP THROUGH TIME -------------------------
func (rnn *RNN) backpropThroughTime(hs [][]float64, input []float64, dhNextInit []float64) ([]float64, []float64, []float64, []float64) {
	dWx := make([]float64, len(rnn.Wx)) // Градиенты входных весов
	dWh := make([]float64, len(rnn.Wh)) // Градиенты скрытых весов
	dbh := make([]float64, len(rnn.bh)) // Градиенты смещений скрытого слоя
	dhNext := dhNextInit                // Ошибка для следующего шага

	seqLen := len(hs) - 1
	for t := seqLen - 1; t >= 0; t-- { // Обратный цикл по шагам последовательности
		h := hs[t+1]                     // Скрытое состояние на шаге t+1
		hPrev := hs[t]                    // Скрытое состояние на шаге t
		x := input[t*rnn.inputDim : (t+1)*rnn.inputDim] // Вектор входа на шаге t

		dh := make([]float64, rnn.hiddenDim) // Ошибка на текущем шаге
		copy(dh, dhNext)                     // Копируем предыдущую ошибку

		dhraw := make([]float64, rnn.hiddenDim) // Градиенты до активации
		for k := 0; k < rnn.hiddenDim; k++ {
			dhraw[k] = (1 - h[k]*h[k]) * dh[k] // Производная tanh
			dbh[k] += dhraw[k]                // Градиент смещения скрытого слоя
		}

		// Градиенты весов
		for iNew := 0; iNew < rnn.hiddenDim; iNew++ {
			for jPrev := 0; jPrev < rnn.hiddenDim; jPrev++ {
				dWh[jPrev+rnn.hiddenDim*iNew] += dhraw[iNew] * hPrev[jPrev]
			}
			for jIn := 0; jIn < rnn.inputDim; jIn++ {
				dWx[jIn+rnn.inputDim*iNew] += dhraw[iNew] * x[jIn]
			}
		}

		// Передача ошибки на предыдущий шаг
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
		arr[i] = rand.NormFloat64() * 0.1 // Малые случайные числа
	}
	return arr
}

func clipGradients(arr []float64, limit float64) {
	for k := range arr { // Ограничение градиентов
		if arr[k] > limit {
			arr[k] = limit
		} else if arr[k] < -limit {
			arr[k] = -limit
		}
	}
}

func (rnn *RNN) updateWeights(dWx, dWh, dWy, dbh, dby []float64, lr float64) {
	for k := range rnn.Wx {
		rnn.Wx[k] -= lr * dWx[k] // Обновление весов входного слоя
	}
	for k := range rnn.Wh {
		rnn.Wh[k] -= lr * dWh[k] // Обновление весов скрытого слоя
	}
	for k := range rnn.Wy {
		rnn.Wy[k] -= lr * dWy[k] // Обновление весов выходного слоя
	}
	for k := range rnn.bh {
		rnn.bh[k] -= lr * dbh[k] // Обновление смещений скрытого слоя
	}
	for k := range rnn.by {
		rnn.by[k] -= lr * dby[k] // Обновление смещений выходного слоя
	}
}

func maxIndex(arr []float64) int { // Индекс максимального элемента
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
	indices := textutils.TextToIndices(inputText, vocab) // Преобразуем текст в индексы
	hPrev := make([]float64, rnn.hiddenDim)             // Инициализация скрытого состояния
	var output []float64

	for _, index := range indices { // Проходим по каждому слову
		x := make([]float64, rnn.inputDim)
		if index >= 0 && index < rnn.inputDim {
			x[index] = 1.0 // One-hot кодирование
		}
		output, hPrev = rnn.forwardParallel(x, hPrev) // Forward pass
	}

	if len(output) == 0 { // Если нет выхода
		return ""
	}

	predIdx := maxIndex(output) // Индекс максимального значения
	for word, idx := range vocab { // Поиск слова по индексу
		if idx == predIdx {
			return word
		}
	}
	return ""
}
