package main

import (
	"flag"
	"path/filepath"
	"strconv"

	"github.com/rsc.io/pdf"
)

func main() {
	flag.Parse()
	args := flag.Args()
	var path []string
	toStdout, err := strconv.ParseBool(args[0])
	if err != nil {
		toStdout = false
		path = args
	} else {
		path = args[1:]
	}
	for i, p := range path {
		path[i] = filepath.FromSlash(p)
	}
	pdf.JSON(path, toStdout)
}
