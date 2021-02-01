package main

import (
  "fmt"
  "github.com/Olly-Fallows/golang-rnn/RNN"
  "github.com/Olly-Fallows/golang-rnn/Logging"
)

func main() {

  logging.SetLogLevel(logging.Info_level)

  input := [][][]float64{
    [][]float64{
      []float64{1},
    },
    [][]float64{
      []float64{0},
    },
  }
  output := [][][]float64{
    [][]float64{
      []float64{1},
    },
    [][]float64{
      []float64{0},
    },
  }

  layers := []int{1,3,1}

  rnn, _ := rnn.MakeRNN(layers)

  for a, _ := range input {
    val, _ := rnn.Execute(input[a][0])
    fmt.Printf("-1: %v -> %v\n", input[a][0], val)
  }
  for i := 0 ; i < 50000 ; i++ {
    for a, _ := range input {
      rnn.Backprop(input[a], output[a], 0, 0.01)
    }
    for a, _ := range input {
      val, _ := rnn.Execute(input[a][0])
      fmt.Printf("%v: %v -> %v\n", i, input[a][0], val)
    }
    fmt.Printf("\n")
  }
}
