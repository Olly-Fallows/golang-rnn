package rnn

import (
  "sync"
)

func Sum(input []float64) (float64, error) {
  var total float64
  for _, v := range input {
    total += v
  }
  return total, nil
}

func Weighted_Sum(wg *sync.WaitGroup, input []float64, weights []float64, output *float64) {
  defer wg.Done()
  var r float64

  for a, v := range input {
    r += v*weights[a]
  }

  *output = r
}

func Parrallel_Weighted_Sum(input []float64, weights [][]float64) ([]float64, error) {
  output := make([]float64, len(weights))
  var wg sync.WaitGroup

  for a, _ := range output {
    wg.Add(1)
    go Weighted_Sum(&wg, input, weights[a], &output[a])
  }

  wg.Wait()
  return output, nil
}

func RELU (x float64) float64 {
  return x
}

func DRELU (x float64) float64 {
  return x
}
