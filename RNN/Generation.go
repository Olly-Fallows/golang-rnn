package rnn

import (
  "math/rand"
  "time"
  "os"
  "encoding/csv"
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

func LoadRNN(path string) (*Network, error) {
  var m *rnn.Network
  // Load template file
  f, err := os.Open(path)
  if err != nil {
    fmt.Printf("[E] Failed to read template file with error: %v\n", err)
    return nil, err
  }
  defer f.Close()

  lines, err := csv.NewReader(f).ReadAll()
  if err != nil {
    fmt.Printf("[E] Failed to parse template file with error: %v\n", err)
    return nil, err
  } else if len(lines) == 0 {
    fmt.Printf("[E] Template was empty.\n")
    return nil, err
  }

  layers := make([]Layer, len(lines[0]))
  for a, l := range lines[0] {
    size, err := strconv.Atoi(l)
    if err != nil {
      fmt.Printf("[E] Failed to convert \"%v\" to int.\n", l)
      return nil, err
    }
    layers[a].Size = size
    layers[a].Biases = make([]float64, size)
    layers[a].Values = make([]float64, size)
    layers[a].Alphas = make([][]float64, size)
    for b=0; b<size; b++ {
      layers[a].Alphas = make([]float64, size)
    }
    if a > 0 {
      layers[a].Weights = make([][]float64, size)
      for b=0; b<layers[a-1].Size; b++ {
        layers[a].Weights = make([]float64, layers[a-1].Size)
      }
    }
  }

  l := 0
  stage := 0
  for a, line := range lines[1:] {
    if stage == 0 {
      // Biases
      for b, d := range line {
        val, err := strconv.Atoi(l)
        if err != nil {
          fmt.Printf("[E] Failed to convert \"%v\" to int.\n", l)
          return nil, err
        }
        layers[l].Biases[b] = val
      }
    } else if stage < layers[l].Size+1 {
      // Alphas
      for b, d := range line {
        val, err := strconv.Atoi(l)
        if err != nil {
          fmt.Printf("[E] Failed to convert \"%v\" to int.\n", l)
          return nil, err
        }
        layers[l].Alphas[stage-1][b] = val
      }
    } else if a > 0 && stage < (layers[l].Size*2)+1 {
      // Weights
      for b, d := range line {
        val, err := strconv.Atoi(l)
        if err != nil {
          fmt.Printf("[E] Failed to convert \"%v\" to int.\n", l)
          return nil, err
        }
        layers[l].Weights[stage-(layers[l].Size+1)][b] = val
      }
    } else {
      stage = -1
      l += 1
    }
    stage += 1
  }
  
  return &Network {
    Layers: layers,
    Depth: len(layers),
  }, nil
}
