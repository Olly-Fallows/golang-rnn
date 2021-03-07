package rnn

import (
  "github.com/Olly-Fallows/golang-rnn/Logging"
  "fmt"
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


func (n *Network) ToCSV() (string) {
  str := ""
  for a, l := range n.Layers {
    if a > 0 {
      str = fmt.Sprintf("%v%v", str, ",")
    }
    str = fmt.Sprintf("%v%v", str, l.Size)
  }
  for a, l := range n.Layers {
    str = fmt.Sprintf("%v%v", str, "\n")
    for b, vals := range l.Biases {
      if b > 0 {
        str = fmt.Sprintf("%v%v", str, ",")
      }
      str = fmt.Sprintf("%v%v", str, vals[b])
    }
    for b, vals := range l.Alphas {
      str = fmt.Sprintf("%v%v", str, "\n")
      for c, v := range vals {
        if c > 0 {
          str = fmt.Sprintf("%v%v", str, ",")
        }
        str = fmt.Sprintf("%v%v", str, v[c])
      }
    }
    for b, vals := range l.Weights {
      str = fmt.Sprintf("%v%v", str, "\n")
      for c, v := range vals {
        if c > 0 {
          str = fmt.Sprintf("%v%v", str, ",")
        }
        str = fmt.Sprintf("%v%v", str, v[c])
      }
    }
  }
  return str
}
