package main

import (
	"flag"
	"strconv"
	"strings"

	"github.com/rsc.io/pdf"
)

func main() {
	flag.Parse()
	args := flag.Args()
	log, _ := strconv.ParseBool(args[0])
	path := args[1:]
	for i, p := range path {
		path[i] = strings.ReplaceAll(p, "\\", "/")
	}
	pdf.JSON(path, log)
}
