package rnn

import (
  _ "fmt"
  "errors"
  "sync"

  "github.com/Olly-Fallows/golang-rnn/Logging"
)

func (n *Network) Execute(input []float64) ([]float64, error) {
  logging.Trace("Network Execution", "Starting execution")

  // Check input is the correct size
  if len(input) != n.Layers[0].Size {
    logging.Err("Network Execution", "Input length and input layer length don't match, with values: %v and %v", len(input), n.Layers[0].Size)
    return nil, errors.New("Input length mismatch")
  }

  // Set layer starting values.
  starting_value := func(wg *sync.WaitGroup, l Layer) {
    defer wg.Done()
    v := make([]float64, l.Size)
    for a, _ := range l.Values {
      v[a] = l.Biases[a]
      for b, _ := range l.Alphas[a] {
        v[a] += l.Alphas[a][b] * l.Values[b]
      }
      v[a] = RELU(v[a])
    }
    for a, _ := range l.Values {
      l.Values[a] = v[a]
    }
  }
  var wg sync.WaitGroup
  for a, _ := range n.Layers {
    wg.Add(1)
    go starting_value(&wg, n.Layers[a])
  }
  wg.Wait()

  // Set input layer to input values
  for a, _ := range n.Layers[0].Values {
    n.Layers[0].Values[a] += input[a]
  }

  // Process layers
  for a := 1; a < n.Depth ; a++ {
    pl := n.Layers[a-1]
    cl := n.Layers[a]

    f, err := Parrallel_Weighted_Sum(pl.Values, cl.Weights)
    if err != nil {
      logging.Err("Network Execution", "Parrallel Weighted Sum fucked up")
      return nil, err
    }

    for a, _ := range cl.Values {
      cl.Values[a] += f[a]
      cl.Values[a] = RELU(cl.Values[a])
    }
  }

  // Get output layer
  return n.Layers[n.Depth-1].Values, nil
}
