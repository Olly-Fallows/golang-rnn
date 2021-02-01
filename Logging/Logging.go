package logging

import (
  "fmt"
)

const (
  Trace_level   = iota
  Debug_level   = iota
  Info_level    = iota
  Warn_level    = iota
  Err_level     = iota
)

var level int = Warn_level

func SetLogLevel(l int) {
  level = l
}

func Trace(function string, msg string, a...interface{}){
  if level > Trace_level {
    return
  }
  fmt.Printf("[%v]: ", function)
  fmt.Printf(msg, a...)
  fmt.Printf("\n")
}

func Debug(function string, msg string, a...interface{}){
  if level > Debug_level {
    return
  }
  fmt.Printf("[%v]: ", function)
  fmt.Printf(msg, a...)
  fmt.Printf("\n")
}

func Info(function string, msg string, a...interface{}){
  if level > Info_level {
    return
  }
  fmt.Printf("[%v]: ", function)
  fmt.Printf(msg, a...)
  fmt.Printf("\n")
}

func Warn(function string, msg string, a...interface{}){
  if level > Warn_level {
    return
  }
  fmt.Printf("[%v]: ", function)
  fmt.Printf(msg, a...)
  fmt.Printf("\n")
}

func Err(function string, msg string, a...interface{}){
  if level > Err_level {
    return
  }
  fmt.Printf("[%v]: ", function)
  fmt.Printf(msg, a...)
  fmt.Printf("\n")
}

func Fatal(function string, msg string, a...interface{}){
  fmt.Printf("[%v]: ", function)
  fmt.Printf(msg, a...)
  fmt.Printf("\n")
}
