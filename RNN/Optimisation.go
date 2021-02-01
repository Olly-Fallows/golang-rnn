package rnn

import (
  "github.com/Olly-Fallows/golang-rnn/Logging"
)

func (n *Network) CalculateDecent (input []float64, output []float64) ([][][]float64, [][][]float64, [][]float64, [][]float64, error){
  logging.Trace("Network CalculateDecent", "Started calculated decent variables")
  dw := make([][][]float64, n.Depth)
  da := make([][][]float64, n.Depth)
  db := make([][]float64, n.Depth)
  dv := make([][]float64, n.Depth)

  for a, l := range n.Layers {
    dw[a] = make([][]float64, len(l.Weights))
    for b, w := range l.Weights {
      dw[a][b] = make([]float64, len(w))
    }
    da[a] = make([][]float64, len(l.Alphas))
    for b, w := range l.Alphas {
      da[a][b] = make([]float64, len(w))
    }
    db[a] = make([]float64, len(l.Biases))
    dv[a] = make([]float64, len(l.Biases))
  }

  old_values := make([][]float64, n.Depth)
  for a, _ := range old_values {
    old_values[a] = make([]float64, len(n.Layers[a].Values))
    for b, _ := range old_values[a] {
      old_values[a][b] = n.Layers[a].Values[b]
    }
  }

  out, err := n.Execute(input)
  if err != nil {
    return nil, nil, nil, nil, err
  }

  for a := n.Depth-1 ; a >= 0 ; a-- {
    if a == n.Depth-1 {
      // Last Layer
      for b, _ := range out {
        dv[a][b] = out[b]-output[b]
        db[a][b] = DRELU(dv[a][b])
      }
    } else {
      // Other Layers
      // Propigate cost derivative
      for b, _ := range dv[a] {
        for c, _ := range dv[a+1] {
          dv[a][b] += dv[a+1][c] * n.Layers[a+1].Weights[c][b]
        }
        db[a][b] = DRELU(dv[a][b])
      }
    }
    // Calculate weight derivative
    for b, _ := range dw[a] {
      for c, _ := range dw[a][b] {
        dw[a][b][c] += DRELU(dv[a][b]) * n.Layers[a-1].Values[c]
      }
    }
    // Calculate alphas derivative
    for b, _ := range da[a] {
      for c, _ := range da[a][b] {
        da[a][b][c] += DRELU(dv[a][b]) * old_values[a][c]
      }
    }
  }

  return dw, da, db, dv, nil
}

func (n *Network) Backprop(input [][]float64, output [][]float64, offset int, step float64) error {
  for a := 0 ; a < len(input) ; a++ {
    n.Clean()
    if a >= offset {
      for b := 0 ; b < a ; b++ {
        n.Execute(input[b])
      }
      dw, da, db, _, _ := n.CalculateDecent(input[a], output[a])
      for a, _ := range dw {
        for b, _ := range dw[a] {
          for c, _ := range dw[a][b] {
            n.Layers[a].Weights[b][c] -= dw[a][b][c]*step
          }
        }
      }
      for a, _ := range da {
        for b, _ := range da[a] {
          for c, _ := range da[a][b] {
            n.Layers[a].Alphas[b][c] -= da[a][b][c]*step
          }
        }
      }
      for a, _ := range db {
        for b, _ := range db[a] {
          n.Layers[a].Biases[b] -= db[a][b]*step
        }
      }
    }
  }
  return nil
}
