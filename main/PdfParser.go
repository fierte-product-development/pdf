package main

import (
	"flag"
	"os"
	"path/filepath"
	"strconv"

	"github.com/rsc.io/pdf"
)

func main() {
	flag.Parse()
	args := flag.Args()
	var path []string
	toFile, err := strconv.ParseBool(args[0])
	if err != nil {
		toFile = true
		path = args
	} else {
		path = args[1:]
	}
	for i, p := range path {
		path[i] = filepath.FromSlash(p)
	}
	os.Stdout.Write(pdf.JSON(path, toFile))
}
