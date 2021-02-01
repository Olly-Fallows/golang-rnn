package rnn

import (
  "github.com/Olly-Fallows/golang-rnn/Logging"
)

type Layer struct {
  Weights [][]float64
  Biases []float64
  Alphas [][]float64
  Values []float64
  Size int
}

type Network struct {
  Layers []Layer
  Depth int
}


func (n *Network) Clean() error {
  logging.Trace("Network Clean", "Started clean function")
  for _, l := range n.Layers {
    for b, _ := range l.Values {
      l.Values[b] = 0
    }
  }
  return nil
}
