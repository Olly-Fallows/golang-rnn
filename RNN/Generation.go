package rnn

import (
  "math/rand"
  "time"
)

func MakeRNN(layers []int) (*Network, error) {
  rand.Seed(time.Now().UTC().UnixNano())

  l := make([]Layer, len(layers))
  d := len(layers)
  for a, _ := range layers {
    var weights [][]float64
    if a != 0 {
      weights = make([][]float64, layers[a])
      for b, _ := range weights {
        weights[b] = make([]float64, layers[a-1])
        for c, _ := range weights[b] {
          weights[b][c] = (1/float64(len(weights[b])))*(rand.Float64()-0.5)
        }
      }
    } else {
      weights = nil
    }

    biases := make([]float64, layers[a])
    for b, _ := range biases{
      biases[b] = (1/float64(len(biases)))*(rand.Float64()-0.5)
    }

    alpha := make([][]float64, layers[a])
    for b, _ := range alpha {
      alpha[b] = make([]float64, layers[a])
      for c, _ := range alpha[b] {
        alpha[b][c] = (1/float64(len(alpha[b])))*(rand.Float64()-0.5)
      }
    }

    l[a] = Layer{
      Weights: weights,
      Biases: biases,
      Alphas: alpha,
      Values: make([]float64, layers[a]),
      Size: layers[a],
    }
  }

  return &Network{
    Layers: l,
    Depth: d,
  }, nil
}

func MakeNormRNN(layers []int) (*Network, error) {
  rand.Seed(time.Now().UTC().UnixNano())

  l := make([]Layer, len(layers))
  d := len(layers)
  for a, _ := range layers {
    var weights [][]float64
    if a != 0 {
      weights = make([][]float64, layers[a])
      for b, _ := range weights {
        weights[b] = make([]float64, layers[a-1])
        for c, _ := range weights[b] {
          weights[b][c] = 0.1
        }
      }
    } else {
      weights = nil
    }

    biases := make([]float64, layers[a])
    for b, _ := range biases{
      biases[b] = 0.1
    }

    alpha := make([][]float64, layers[a])
    for b, _ := range alpha {
      alpha[b] = make([]float64, layers[a])
      for c, _ := range alpha[b] {
        alpha[b][c] = 0.1
      }
    }

    l[a] = Layer{
      Weights: weights,
      Biases: biases,
      Alphas: alpha,
      Values: make([]float64, layers[a]),
      Size: layers[a],
    }
  }

  return &Network{
    Layers: l,
    Depth: d,
  }, nil
}
